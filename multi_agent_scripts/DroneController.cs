using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine.InputSystem;
using System.Collections.Generic;
using System.IO;
using System;

public class DroneController : Agent
{
    private Rigidbody rb;
    public float thrust = 15f;
    public float maxTiltAngle = 30f;
    public float tiltSpeed = 8f;
    public float maxYawSpeed = 120f;
    public float hoverForce = 9.81f;
    public float drag = 2f;
    public float angularDrag = 5f;
    public float maxVelocity = 15f;
    public bool test = false;
    public bool recordEvaluationMetrics = false;
    public Animator[] fanAnimators;

    float yaw, pitch, roll, throttle;

    public float actionSmoothing = 10f;
    private float smoothThrottle, smoothRoll, smoothPitch, smoothYaw;

    public bool hoverMode = true;

    private Vector3 startPosition;
    private Quaternion startRotation;

    private DroneEnvironmentConfig envConfig;

    [HideInInspector] public int agentIndex = 0;
    private int targetIndex = -1;

    private int episodeCount = 0;
    private float episodeStartTime;
    private int stepCount = 0;

    private float episodeReward = 0f;
    private float cumulativeReward = 0f;
    private List<float> distanceToTargetHistory = new List<float>();
    private float minDistanceAchieved = float.MaxValue;

    private int successCount = 0;
    private int crashCount = 0;
    private int timeoutCount = 0;
    private int outOfBoundsCount = 0;

    private float totalFlightTime = 0f;
    private float avgVelocity = 0f;
    private float maxVelocityReached = 0f;
    private float avgTiltAngle = 0f;
    private List<float> velocitySamples = new List<float>();
    private List<float> tiltSamples = new List<float>();

    private int currentCurriculumStage = 0;
    private float currentTargetDistance = 0f;
    private int currentObstacleCount = 0;

    public float droneHP = 500f;
    public const float MAX_HP = 500f;
    private float softLandingThreshold = 2f;
    private float hardLandingThreshold = 5f;

    private float previousDistance = Mathf.Infinity;
    private float closestDistanceEver = Mathf.Infinity;
    private int backtrackCounter = 0;

    private HashSet<Vector3Int> visitedCells = new HashSet<Vector3Int>();
    private int hoverCounter = 0;
    private Vector3 lastHoverCheck;
    private const float GRID_CELL_SIZE = 3f;
    private string endReason = "start";

    private float groundDistance = 20f;
    private bool isNearGround = false;

    private Vector3 lastPos;
    private float distanceTraveled = 0f;

    public int goalsReached = 0;
    private int groundCollisionCount = 0;
    private float shortestPath = 0f;
    private int groundMask;

    private GameObject[] cachedObstacles;
    private string logFilePath;
    private bool isVictimVisible = false;
    private bool wasVictimVisibleLastStep = false;

    // Evaluation metrics logging
    [Header("Test Mode Metrics")]
    public int testEpisodeLimit = 100;
    private int testEpisodesRun = 0;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        rb.linearDamping = drag;
        rb.angularDamping = angularDrag;
        groundMask = LayerMask.GetMask("Ground");
        envConfig = FindFirstObjectByType<DroneEnvironmentConfig>();
        startPosition = transform.position;
        startRotation = transform.rotation;

        if (recordEvaluationMetrics && agentIndex==0)
        {
            logFilePath = Path.Combine(Application.dataPath, "..", "Drone_Test_Results.csv");
            if (!File.Exists(logFilePath))
            {
                try {
                    File.WriteAllText(logFilePath, "Episode,VictimsRescued,TotalVictims,StepsTaken,PathEfficiency,DistanceTraveled,EndReason,DroneHP\n");
                } catch(Exception e) { Debug.LogError("CSV Init Error: " + e.Message); }
            }
            Debug.Log($"[Test Mode] Logging results to: {logFilePath}");
        }
    }

    private void LogTestRun()
    {
        if (!recordEvaluationMetrics || agentIndex != 0 || episodeCount == 0 || string.IsNullOrEmpty(logFilePath)) return;

        int totalVictims = envConfig != null ? envConfig.victimCount : 1;
        int teamRescued = envConfig != null ? envConfig.TotalVictimsFound() : goalsReached;
        float efficiency = shortestPath > 0 ? shortestPath / Mathf.Max(distanceTraveled, 0.001f) : 0f;

        string agentGoals = "";
        string agentHPs = "";
        if (envConfig != null)
        {
            for (int i = 0; i < envConfig.AgentCount; i++)
            {
                var info = envConfig.GetAgentInfo(i);
                agentGoals += info.goals.ToString();
                agentHPs += info.hp.ToString("F1");
                if (i < envConfig.AgentCount - 1) { agentGoals += ","; agentHPs += ","; }
            }
        }

        string logData = $"{episodeCount},{teamRescued},{totalVictims},{stepCount}," +
                        $"{efficiency:F3},{endReason},{agentGoals},{agentHPs}\n";
        try { File.AppendAllText(logFilePath, logData); }
        catch (Exception e) { Debug.LogError("CSV Write Error: " + e.Message); }
    }

    private void EndEpisodeClean(string reason)
    {
        endReason = reason;
        if (envConfig != null) envConfig.ReleaseClaim(agentIndex);
        EndEpisode();
    }

    public override void OnEpisodeBegin()
    {
        LogTestRun();

        // Increment episode counter and check test limit (only agent 0 tracks this)
        if (agentIndex == 0)
        {
            episodeCount++;
            if (recordEvaluationMetrics)
            {
                testEpisodesRun++;
                if (testEpisodesRun > testEpisodeLimit)
                {
                    Debug.Log($"[Test Complete] Ran {testEpisodesRun - 1} episodes. Results saved to: {logFilePath}");
                    #if UNITY_EDITOR
                        UnityEditor.EditorApplication.isPlaying = false;
                    #else
                        UnityEngine.Application.Quit();
                    #endif
                    return;
                }
            }
        }

        if (envConfig != null)
        {
            envConfig.RequestReset();
        }

        episodeStartTime = Time.time;
        stepCount = 0;
        episodeReward = 0f;
        droneHP = MAX_HP;
        minDistanceAchieved = float.MaxValue;
        distanceToTargetHistory.Clear();
        velocitySamples.Clear();
        tiltSamples.Clear();

        smoothThrottle = 0f;
        smoothRoll = 0f;
        smoothPitch = 0f;
        smoothYaw = 0f;

        previousDistance = Mathf.Infinity;
        closestDistanceEver = Mathf.Infinity;
        backtrackCounter = 0;
        visitedCells.Clear();
        hoverCounter = 0;
        lastHoverCheck = Vector3.zero;
        distanceTraveled = 0f;
        endReason = "start";
        goalsReached = 0;
        groundCollisionCount = 0;
        shortestPath = 0f;
        targetIndex = -1;
    }

    public void PostEnvironmentReset()
    {
        if (envConfig != null)
        {
            cachedObstacles = envConfig.activeObstacles;
            currentTargetDistance = envConfig.targetDistance;
            currentObstacleCount = envConfig.obstacleCount;

            targetIndex = envConfig.ClaimNearestAvailableVictim(agentIndex, transform.position);
            GameObject claimedVictim = envConfig.GetVictim(targetIndex);
            if (claimedVictim != null)
            {
                // FIX: use world position
                previousDistance = Vector3.Distance(transform.position, claimedVictim.transform.position);
                closestDistanceEver = previousDistance;
                shortestPath = CalculateShortestPath();
            }
        }
        lastPos = transform.position;
        lastHoverCheck = transform.position;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (envConfig == null || envConfig.isInitializing)
        {
            // Base obs: 20
            sensor.AddObservation(Vector3.zero); // 3
            sensor.AddObservation(Vector3.zero); // 3
            sensor.AddObservation(Vector3.zero); // 3
            sensor.AddObservation(Quaternion.identity); // 4
            sensor.AddObservation(0f); // 1
            sensor.AddObservation(0f); // 1
            sensor.AddObservation(0f); // 1
            sensor.AddObservation(0f); // 1
            sensor.AddObservation(0f); // 1
            sensor.AddObservation(0f); // 1
            sensor.AddObservation(0f); // 1

            // Teammate obs (3 teammates defaults)
            int teammateCount = envConfig != null ? envConfig.AgentCount - 1 : 2;
            for (int t = 0; t < teammateCount; t++)
            {
                sensor.AddObservation(Vector3.zero); // 3
                sensor.AddObservation(Vector3.zero); // 3
                sensor.AddObservation(0f); // 1
            }
            return;
        }

        Vector3 normalizedPos = (transform.position - envConfig.arenaCenter);
        normalizedPos.x /= envConfig.arenaSize.x;
        normalizedPos.y /= envConfig.arenaSize.y;
        normalizedPos.z /= envConfig.arenaSize.z;
        sensor.AddObservation(normalizedPos); // 3

        sensor.AddObservation(transform.InverseTransformDirection(rb.linearVelocity) / maxVelocity); // 3

        float maxYawRad = maxYawSpeed * Mathf.Deg2Rad;
        sensor.AddObservation(rb.angularVelocity / maxYawRad); // 3

        sensor.AddObservation(transform.localRotation); // 4

        GameObject claimedVictim = envConfig.GetVictim(targetIndex);
        if (claimedVictim != null && claimedVictim.activeSelf)
        {
            // FIX: use world position for direction and distance
            Vector3 toVictim = claimedVictim.transform.position - transform.position;
            Vector3 localDir = transform.InverseTransformDirection(toVictim).normalized;

            sensor.AddObservation(localDir.x); // 1
            sensor.AddObservation(localDir.y); // 1
            sensor.AddObservation(localDir.z); // 1

            float dist = toVictim.magnitude;
            sensor.AddObservation(Mathf.Clamp01(dist / 50f)); // 1

            isVictimVisible = true;
        }
        else
        {
            sensor.AddObservation(Vector3.zero);
            sensor.AddObservation(0f);
            isVictimVisible = false;
        }

        int totalVictims = envConfig.victimCount;
        int teamFound = envConfig.TotalVictimsFound();
        sensor.AddObservation((float)teamFound / Mathf.Max(totalVictims, 1)); // 1

        int remaining = envConfig.RemainingVictims();
        sensor.AddObservation(remaining / (float)Mathf.Max(totalVictims, 1)); // 1

        RaycastHit groundHit;
        if (Physics.Raycast(transform.position, Vector3.down, out groundHit, 20f, groundMask))
        {
            groundDistance = groundHit.distance;
        }
        else
        {
            groundDistance = 20f;
        }

        isNearGround = groundDistance < 3f;
        sensor.AddObservation(groundDistance / 20f); // 1

        int activeTeammateCount = envConfig.AgentCount - 1;
        for (int t = 0; t < activeTeammateCount; t++)
        {
            var info = envConfig.GetTeammateInfo(agentIndex, t);
            if (info.isActive)
            {
                // FIX: use world position for teammate relative position
                Vector3 relPos = info.position - transform.position;
                relPos.x /= envConfig.arenaSize.x;
                relPos.y /= envConfig.arenaSize.y;
                relPos.z /= envConfig.arenaSize.z;
                sensor.AddObservation(relPos); // 3
                sensor.AddObservation(transform.InverseTransformDirection(info.velocity) / maxVelocity); // 3
                sensor.AddObservation(info.hp / MAX_HP); // 1
            }
            else
            {
                sensor.AddObservation(Vector3.zero); // 3
                sensor.AddObservation(Vector3.zero); // 3
                sensor.AddObservation(0f); // 1
            }
        }
    }

    void FixedUpdate()
    {
        if (envConfig != null && envConfig.isInitializing) return;

        Vector3 currentEuler = transform.rotation.eulerAngles;
        float newYaw = currentEuler.y + (yaw * maxYawSpeed * Time.fixedDeltaTime);
        Quaternion targetRotation = Quaternion.Euler(pitch * maxTiltAngle, newYaw, -roll * maxTiltAngle);
        rb.MoveRotation(Quaternion.Slerp(transform.rotation, targetRotation, Time.fixedDeltaTime * tiltSpeed));

        float tiltAngle = Vector3.Angle(Vector3.up, transform.up);
        float tiltCompensation = 1f / Mathf.Max(Mathf.Cos(tiltAngle * Mathf.Deg2Rad), 0.2f);
        float gravityCounter = Physics.gravity.magnitude * tiltCompensation;
        float totalThrust = gravityCounter + (throttle * thrust);
        rb.AddForce(transform.up * totalThrust, ForceMode.Acceleration);

        if (rb.linearVelocity.magnitude > maxVelocity)
        {
            rb.linearVelocity = rb.linearVelocity.normalized * maxVelocity;
        }
    }

    void Update()
    {
        if (envConfig != null && envConfig.isInitializing) return;

        if (Input.GetKeyDown(KeyCode.H))
        {
            hoverMode = !hoverMode;
            Debug.Log("Hover Mode: " + (hoverMode ? "ON" : "OFF"));
        }

        if (Input.GetKeyDown(KeyCode.R))
        {
            EndEpisodeClean("manual_reset");
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (envConfig != null && envConfig.isInitializing) return;

        if (targetIndex >= 0)
        {
            GameObject claimed = envConfig.GetVictim(targetIndex);
            if (claimed == null || !claimed.activeSelf || !envConfig.IsVictimClaimed(targetIndex, agentIndex))
            {
                envConfig.ReleaseClaim(agentIndex);
                // FIX: use world position
                targetIndex = envConfig.ClaimNearestAvailableVictim(agentIndex, transform.position);
                if (targetIndex >= 0)
                {
                    GameObject newClaimed = envConfig.GetVictim(targetIndex);
                    // FIX: use world position
                    previousDistance = Vector3.Distance(transform.position, newClaimed.transform.position);
                    closestDistanceEver = previousDistance;
                    backtrackCounter = 0;
                }
            }
        }
        else
        {
            // FIX: use world position
            targetIndex = envConfig.ClaimNearestAvailableVictim(agentIndex, transform.position);
            if (targetIndex >= 0)
            {
                GameObject newClaimed = envConfig.GetVictim(targetIndex);
                // FIX: use world position
                previousDistance = Vector3.Distance(transform.position, newClaimed.transform.position);
                closestDistanceEver = previousDistance;
                backtrackCounter = 0;
            }
        }

        float rawThrottle = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float rawRoll = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        float rawPitch = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);
        float rawYaw = Mathf.Clamp(actions.ContinuousActions[3], -1f, 1f);

        float actionDelta = Mathf.Abs(rawThrottle - smoothThrottle) +
                            Mathf.Abs(rawRoll - smoothRoll) +
                            Mathf.Abs(rawPitch - smoothPitch) +
                            Mathf.Abs(rawYaw - smoothYaw);

        if (actionDelta > 0.5f)
        {
            float penalty = actionDelta * 0.002f;
            AddReward(-penalty);
            episodeReward -= penalty;
        }

        smoothThrottle = Mathf.Lerp(smoothThrottle, rawThrottle, Time.fixedDeltaTime * actionSmoothing);
        smoothRoll = Mathf.Lerp(smoothRoll, rawRoll, Time.fixedDeltaTime * actionSmoothing);
        smoothPitch = Mathf.Lerp(smoothPitch, rawPitch, Time.fixedDeltaTime * actionSmoothing);
        smoothYaw = Mathf.Lerp(smoothYaw, rawYaw, Time.fixedDeltaTime * actionSmoothing);

        throttle = smoothThrottle;
        roll = smoothRoll;
        pitch = smoothPitch;
        yaw = smoothYaw;

        stepCount++;

        CalculateRewards();
        GiveScanningRewards();
        CheckTerminationConditions();
    }

    private void GiveScanningRewards()
    {
        bool victimsRemain = envConfig != null && envConfig.RemainingVictims() > 0;

        if (victimsRemain)
        {
            float horizontalSpeed = new Vector3(rb.linearVelocity.x, 0, rb.linearVelocity.z).magnitude;

            if (horizontalSpeed > 1f)
            {
                AddReward(0.001f);
            }
            else if (rb.linearVelocity.magnitude < 0.1f)
            {
                AddReward(-0.001f);
            }
        }

        wasVictimVisibleLastStep = isVictimVisible;
    }

    void CalculateRewards()
    {
        float reward = 0f;

        bool hasVictims = envConfig != null && envConfig.RemainingVictims() > 0;
        if (!hasVictims)
        {
            float altitudeError = Mathf.Abs(transform.position.y - envConfig.spawnHeight);
            if (altitudeError < 0.5f) reward += 0.15f;
            else if (altitudeError < 1.0f) reward += 0.05f;
            else if (altitudeError < 2.0f) reward += 0.01f;
            else reward -= 0.02f;

            float velocity = rb.linearVelocity.magnitude;
            if (velocity < 1f) reward += 0.005f;
        }

        float altitude = transform.position.y;
        if (altitude <= 1f) reward -= 0.08f;

        float uprightness = Vector3.Dot(transform.up, Vector3.up);

        Vector3 euler = transform.localRotation.eulerAngles;
        float tiltX = NormalizeAngle(euler.x);
        float tiltZ = NormalizeAngle(euler.z);
        float combinedTilt = Mathf.Sqrt(tiltX * tiltX + tiltZ * tiltZ);

        if (combinedTilt > maxTiltAngle + 15f) reward -= 0.4f;

        if (rb.linearVelocity.y < -2.5f) reward -= 0.01f;

        if (isNearGround && rb.linearVelocity.y < -1.5f)
        {
            reward -= 0.05f;
            if (rb.linearVelocity.y < -3f) reward -= 0.2f;
        }

        if (isNearGround && rb.linearVelocity.y > -1.0f && rb.linearVelocity.y < 0.5f)
            reward += 0.005f;

        if (altitude < 0.5f) reward -= 0.01f;

        float horizontalSpeed = new Vector3(rb.linearVelocity.x, 0, rb.linearVelocity.z).magnitude;
        float yawRate = Mathf.Abs(rb.angularVelocity.y);
        bool isSpinningInPlace = yawRate > 0.3f && horizontalSpeed < 0.3f;

        if (isSpinningInPlace && envConfig != null && envConfig.RemainingVictims() > 0)
        {
            reward -= 0.01f;
        }

        bool obstacleInPath = false;
        GameObject claimedVictimObj = envConfig != null ? envConfig.GetVictim(targetIndex) : null;
        Transform nearestVictimObs = claimedVictimObj != null ? claimedVictimObj.transform : null;

        if (nearestVictimObs != null)
        {
            float closestObsDist = Mathf.Infinity;
            Vector3 agentPos = transform.position;
            // FIX: use world position
            Vector3 targetDir = (nearestVictimObs.position - agentPos).normalized;
            float targetDist = Vector3.Distance(agentPos, nearestVictimObs.position);

            if (cachedObstacles != null)
            {
                foreach (GameObject obs in cachedObstacles)
                {
                    if (obs != null && obs.activeSelf)
                    {
                        float dist = Vector3.Distance(agentPos, obs.transform.position);
                        if (dist < closestObsDist) closestObsDist = dist;

                        if (dist < targetDist)
                        {
                            Vector3 obsDir = (obs.transform.position - agentPos).normalized;
                            float dot = Vector3.Dot(targetDir, obsDir);
                            if (dot > 0.7f) obstacleInPath = true;
                        }
                    }
                }
            }

            // FIX: use world position for alignment check
            Vector3 toTarget = (nearestVictimObs.position - transform.position);
            toTarget.y = 0f;

            if (toTarget.magnitude > 0.5f)
            {
                Vector3 flatForward = transform.forward;
                flatForward.y = 0f;
                flatForward.Normalize();

                float targetAlignment = Vector3.Dot(flatForward, toTarget.normalized);
                if (targetAlignment < 0.7f)
                {
                    float targetMisalignment = 1.0f - targetAlignment;
                    reward -= 0.005f * targetMisalignment;
                }
            }

            if (closestObsDist < 4f)
            {
                float proximityPenalty = (4f - closestObsDist) / 4f * 0.05f;
                reward -= proximityPenalty;

                float speed = rb.linearVelocity.magnitude;
                if (speed > 3f)
                {
                    float speedPenalty = (speed - 3f) / 10f * 0.03f;
                    reward -= speedPenalty;
                }
            }
        }

        if (nearestVictimObs != null)
        {
            // FIX: use world position for all distance tracking
            float currentDistance = Vector3.Distance(transform.position, nearestVictimObs.position);
            distanceToTargetHistory.Add(currentDistance);

            if (currentDistance < minDistanceAchieved) minDistanceAchieved = currentDistance;

            if (currentDistance > 0f)
            {
                float progressMultiplier = obstacleInPath ? 0.3f : 1.0f;
                float delta = previousDistance - currentDistance;

                if (delta > 0.01f)
                {
                    int victimsLeft = envConfig.RemainingVictims();
                    float urgencyMultiplier = 1f + (victimsLeft * 0.1f);

                    reward += Mathf.Clamp(delta * 2f, 0f, 0.5f) * progressMultiplier * urgencyMultiplier;
                    backtrackCounter = 0;

                    if (currentDistance < closestDistanceEver)
                        closestDistanceEver = currentDistance;
                }
                else if (delta < -0.01f)
                {
                    float backtrackDelta = -delta;
                    if (obstacleInPath)
                    {
                        backtrackCounter = 0;
                    }
                    else
                    {
                        reward -= Mathf.Clamp(backtrackDelta * 1.5f, 0f, 0.4f);
                        backtrackCounter++;
                        if (backtrackCounter > 20) reward -= 0.02f;
                    }
                }
                else
                {
                    backtrackCounter = Mathf.Max(0, backtrackCounter - 1);
                }

                previousDistance = currentDistance;
            }

            bool rescuedClaimed = false;

            if (envConfig.TryRescueAnyVictim(transform.position, out int rescuedIdx))
            {
                goalsReached++;
                rescuedClaimed = (rescuedIdx == targetIndex);

                AddReward(reward);
                episodeReward += reward;
                reward = 0f;

                int remaining = envConfig.RemainingVictims();
                float baseReward = 5.0f;
                float timeBonus = Mathf.Max(0f, (envConfig.episodeLength - stepCount) / envConfig.episodeLength) * 1f;
                float difficultyBonus = envConfig.obstacleCount > 0 ? 2.0f : 0f;

                float rescueReward = baseReward + timeBonus + difficultyBonus;

                reward += rescueReward;
                AddReward(reward);
                episodeReward += reward;

                Debug.Log($"VICTIM RESCUED! Team={envConfig.TotalVictimsFound()}/{envConfig.victimCount} | Episode {episodeCount}, Steps: {stepCount}");

                envConfig.ReleaseClaim(agentIndex);
                // FIX: use world position
                targetIndex = envConfig.ClaimNearestAvailableVictim(agentIndex, transform.position);
                if (targetIndex >= 0)
                {
                    GameObject nextV = envConfig.GetVictim(targetIndex);
                    // FIX: use world position
                    previousDistance = Vector3.Distance(transform.position, nextV.transform.position);
                    closestDistanceEver = previousDistance;
                }
                backtrackCounter = 0;

                if (envConfig.AllVictimsFound())
                {
                    successCount++;
                    float efficiency = shortestPath / Mathf.Max(distanceTraveled, 0.001f);
                    float completionBonus = 10f;
                    float efficiencyBonus = efficiency * 5f;
                    AddReward(completionBonus + efficiencyBonus);
                    episodeReward += completionBonus + efficiencyBonus;

                    endReason = "team_completion";
                    Debug.Log($"ALL VICTIMS RESCUED! Team Bonus! Episode {episodeCount}");
                    EndEpisodeClean("team_completion");
                    return;
                }
            }
        }

        Vector3Int currentCell = new Vector3Int(
            Mathf.RoundToInt(transform.position.x / GRID_CELL_SIZE),
            Mathf.RoundToInt(transform.position.y / GRID_CELL_SIZE),
            Mathf.RoundToInt(transform.position.z / GRID_CELL_SIZE)
        );

        if (!visitedCells.Contains(currentCell))
        {
            visitedCells.Add(currentCell);
            float explorationReward = 0.05f / (1f + visitedCells.Count * 0.05f);
            reward += explorationReward;
        }

        if (envConfig != null && envConfig.RegisterExploredCell(currentCell))
        {
            reward += 0.02f;
        }

        if (envConfig != null)
        {
            for (int t = 0; t < envConfig.AgentCount - 1; t++)
            {
                var info = envConfig.GetTeammateInfo(agentIndex, t);
                if (info.isActive)
                {
                    // FIX: use world position for teammate proximity
                    float teammateDist = Vector3.Distance(transform.position, info.position);
                    if (teammateDist < 3f)
                    {
                        reward -= 0.005f * (3f - teammateDist) / 3f;
                    }
                }
            }
        }

        bool isNavigating = envConfig != null && envConfig.RemainingVictims() > 0;
        if (isNavigating && stepCount % 10 == 0)
        {
            // FIX: use world position for hover detection
            float movement = Vector3.Distance(transform.position, lastHoverCheck);

            if (movement < 0.2f)
            {
                hoverCounter++;
                if (hoverCounter > 15)
                {
                    reward -= 0.1f;
                }
                if (hoverCounter > 30)
                {
                    reward -= 1f;
                    endReason = "idle_hovering";
                    AddReward(reward);
                    episodeReward += reward;
                    Debug.Log($"Idle Hovering detected! Episode {episodeCount}");
                    if (!test) EndEpisodeClean("idle_hovering");
                    return;
                }
            }
            else
            {
                hoverCounter = 0;
            }

            lastHoverCheck = transform.position;
        }

        // FIX: use world position for distance traveled
        float newDist = Vector3.Distance(transform.position, lastPos);
        if (newDist < 0.001f) reward -= 0.01f;
        distanceTraveled += newDist;
        lastPos = transform.position;

        reward -= 0.005f;

        float spd = rb.linearVelocity.magnitude;
        if (spd > 12f) reward -= 0.01f * (spd - 12f);

        AddReward(reward);
        episodeReward += reward;

        var recorder = Academy.Instance.StatsRecorder;
        recorder.Add("AngleStability", uprightness);
        recorder.Add("GroundDistance", groundDistance);
        recorder.Add("VerticalVelocity", rb.linearVelocity.y);
        recorder.Add("DroneHP", droneHP);
        if (agentIndex == 0 && envConfig != null)
        {
            recorder.Add("GoalsReached", envConfig.TotalVictimsFound(), Unity.MLAgents.StatAggregationMethod.MostRecent);
        }
        recorder.Add("GroundCollisions", groundCollisionCount, Unity.MLAgents.StatAggregationMethod.MostRecent);
        recorder.Add("ExploredCells", visitedCells.Count, Unity.MLAgents.StatAggregationMethod.MostRecent);
        recorder.Add("BacktrackCount", backtrackCounter, Unity.MLAgents.StatAggregationMethod.MostRecent);
        recorder.Add("ClosestEver", closestDistanceEver, Unity.MLAgents.StatAggregationMethod.MostRecent);
        recorder.Add("PathEfficiency", shortestPath / Mathf.Max(distanceTraveled, 0.001f));
        recorder.Add("MissionProgress", (float)goalsReached / Mathf.Max(envConfig != null ? envConfig.victimCount : 1, 1));
    }

    void OnCollisionEnter(Collision other)
    {
        if (other.gameObject.CompareTag("Ground"))
        {
            groundCollisionCount++;
            float impactSpeed = other.relativeVelocity.magnitude;

            if (hoverMode)
            {
                AddReward(-3f);
                episodeReward -= 3f;
                crashCount++;
                endReason = "ground_contact_hover";
                Debug.Log($"Ground contact in hover mode! Episode {episodeCount}");
            }

            if (impactSpeed >= hardLandingThreshold)
            {
                float damage = impactSpeed * impactSpeed * 1.0f;
                droneHP -= damage;
                float penalty = damage / MAX_HP * 2f;
                AddReward(-penalty);
                episodeReward -= penalty;

                if (droneHP <= 0f)
                {
                    AddReward(-2f);
                    episodeReward -= 2f;
                    crashCount++;
                    endReason = "hard_crash_destroyed";
                    EndEpisodeClean("hard_crash_destroyed");
                    return;
                }
            }
            else
            {
                float damage = impactSpeed * 2f + 2f;
                droneHP -= damage;
                AddReward(-0.5f);
                episodeReward -= 0.5f;

                if (droneHP <= 0f)
                {
                    AddReward(-1f);
                    episodeReward -= 1f;
                    crashCount++;
                    endReason = "rough_landing_destroyed";
                    EndEpisodeClean("rough_landing_destroyed");
                    return;
                }
            }
        }

        if (other.gameObject.CompareTag("Obstacle"))
        {
            float impactSpeed = other.relativeVelocity.magnitude;
            float damage = (impactSpeed * impactSpeed * 1.5f) + 10f;
            droneHP -= damage;
            float damagePenalty = (damage / MAX_HP * 3f) + 3f;
            AddReward(-damagePenalty);
            episodeReward -= damagePenalty;

            if (droneHP <= 0f)
            {
                AddReward(-10f);
                episodeReward -= 10f;
                crashCount++;
                endReason = "drone_destroyed_obstacle";
                EndEpisodeClean("drone_destroyed_obstacle");
                return;
            }
        }
    }

    void OnCollisionStay(Collision other)
    {
        if (other.gameObject.CompareTag("Obstacle"))
        {
            float stayDamage = 1.0f;
            droneHP -= stayDamage;
            AddReward(-0.05f);
            episodeReward -= 0.05f;

            if (droneHP <= 0f)
            {
                AddReward(-10f);
                episodeReward -= 10f;
                crashCount++;
                endReason = "grinding_obstacle_destroyed";
                EndEpisodeClean("grinding_obstacle_destroyed");
                return;
            }
        }

        if (other.gameObject.CompareTag("Ground") && !hoverMode)
        {
            AddReward(-0.01f);
            episodeReward -= 0.01f;
        }
    }

    void CheckTerminationConditions()
    {
        if (test) return;
        if (envConfig == null) return;

        // FIX: use world position for bounds check
        if (envConfig.IsOutOfBounds(transform.position))
        {
            AddReward(-10f);
            episodeReward -= 10f;
            outOfBoundsCount++;
            endReason = "out_of_bounds";
            EndEpisodeClean("out_of_bounds");
            return;
        }

        if (transform.position.y < 0.2f)
        {
            AddReward(-5f);
            episodeReward -= 5f;
            crashCount++;
            endReason = "crashed_below_ground";
            EndEpisodeClean("crashed_below_ground");
            return;
        }

        if (stepCount >= envConfig.episodeLength)
        {
            int remaining = envConfig != null ? envConfig.RemainingVictims() : 0;
            float timeoutPenalty = -2f - (remaining * 1f);
            AddReward(timeoutPenalty);
            episodeReward += timeoutPenalty;
            timeoutCount++;
            endReason = "timeout";
            Debug.Log($"TIMEOUT! Episode {episodeCount}, Steps: {stepCount}, Remaining victims: {remaining}, Penalty: {timeoutPenalty:F1}");
            EndEpisodeClean("timeout");
            return;
        }
    }

    void CollectStepStatistics()
    {
        float currentVel = rb.linearVelocity.magnitude;
        velocitySamples.Add(currentVel);
        if (currentVel > maxVelocityReached) maxVelocityReached = currentVel;

        Vector3 euler = transform.localRotation.eulerAngles;
        float tiltMagnitude = Mathf.Sqrt(
            Mathf.Pow(NormalizeAngle(euler.x), 2) +
            Mathf.Pow(NormalizeAngle(euler.z), 2)
        );
        tiltSamples.Add(tiltMagnitude);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;

        if (Input.GetKey(KeyCode.W))
            continuousActions[0] = 1f;
        else if (Input.GetKey(KeyCode.S))
            continuousActions[0] = -1f;
        else
            continuousActions[0] = 0f;

        float speed = Mathf.Abs(continuousActions[0]) != 0 ? 2f : 1f;
        foreach (Animator anim in fanAnimators)
            anim.SetFloat("fanSpeed", speed);

        continuousActions[1] = Input.GetAxis("Horizontal");

        if (Input.GetKey(KeyCode.UpArrow))
            continuousActions[2] = 1f;
        else if (Input.GetKey(KeyCode.DownArrow))
            continuousActions[2] = -1f;
        else
            continuousActions[2] = 0f;

        if (Input.GetKey(KeyCode.Q))
            continuousActions[3] = -1f;
        else if (Input.GetKey(KeyCode.E))
            continuousActions[3] = 1f;
        else
            continuousActions[3] = 0f;
    }

    public void ResetDrone()
    {
        Debug.Log("Drone reset");
        throttle = 0;
        roll = 0;
        pitch = 0;
        yaw = 0;

        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        rb.Sleep();
        rb.position = startPosition;
        rb.rotation = startRotation;
        rb.WakeUp();
    }

    float NormalizeAngle(float angle)
    {
        while (angle > 180f) angle -= 360f;
        while (angle < -180f) angle += 360f;
        return angle;
    }

    float CalculateShortestPath()
    {
        if (envConfig == null) return 0f;
        GameObject[] victims = envConfig.activeVictims;
        if (victims == null || victims.Length == 0) return 0f;

        float total = 0f;
        // FIX: use world position as starting point
        Vector3 current = transform.position;
        List<GameObject> remaining = new List<GameObject>();
        foreach (var v in victims) if (v != null) remaining.Add(v);

        while (remaining.Count > 0)
        {
            float nearest = Mathf.Infinity;
            GameObject nearestVic = null;
            foreach (var v in remaining)
            {
                if (v == null) continue;
                // FIX: use world position
                float d = Vector3.Distance(current, v.transform.position);
                if (d < nearest) { nearest = d; nearestVic = v; }
            }
            if (nearestVic == null) break;
            total += nearest;
            // FIX: use world position
            current = nearestVic.transform.position;
            remaining.Remove(nearestVic);
        }
        return total;
    }

    void OnGUI()
    {
        GUIStyle style = new GUIStyle();
        style.fontSize = 14;
        style.normal.textColor = Color.white;

        float currentDist = 0f;
        GameObject claimedVic = envConfig != null ? envConfig.GetVictim(targetIndex) : null;
        Transform nearestGui = claimedVic != null ? claimedVic.transform : null;
        if (nearestGui != null && envConfig != null && !envConfig.isInitializing)
            // FIX: use world position for GUI display
            currentDist = Vector3.Distance(transform.position, nearestGui.position);

        int victimsLeft = envConfig != null ? envConfig.RemainingVictims() : 0;
        int xOffset = 20 + (agentIndex * 260);

        GUI.Label(new Rect(xOffset, 30, 250, 30), $"[Agent {agentIndex}] Rwd: {GetCumulativeReward():F2}", style);
        GUI.Label(new Rect(xOffset, 50, 250, 30), $"Dist: {currentDist:F1} | Rec: {closestDistanceEver:F1}", style);
        GUI.Label(new Rect(xOffset, 70, 250, 30), $"Explored: {visitedCells.Count} | Team: {(envConfig != null ? envConfig.TotalVictimsFound() : 0)}/{victimsLeft}", style);
        GUI.Label(new Rect(xOffset, 90, 250, 30), $"HP: {droneHP:F0} | {endReason}", style);
        GUI.Label(new Rect(xOffset, 110, 250, 30), $"Target: {targetIndex}", style);

        if (nearestGui != null)
        {
            // FIX: use world position for angle display
            float angle = Vector3.Angle(transform.forward, nearestGui.position - transform.position);
            GUI.Label(new Rect(xOffset, 130, 250, 30), $"Angle: {angle:F0}°", style);
        }
    }
}