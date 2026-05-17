using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine.InputSystem;
using System.Collections;
using System.Collections.Generic;

public class DroneEnvironmentConfig : MonoBehaviour
{
    [Header("Curriculum Parameters (Auto-Updated by ML-Agents)")]
    public float targetDistance = 0f;
    public int obstacleCount = 0;

    public float episodeLength = 5000f;
    public float spawnHeight = 10f;
    public float successRadius = 1f;

    [Header("References")]
    public Transform drone;
    public GameObject victimPrefab;      // Victim prefab (replaces single target)
    public GameObject obstaclePrefab;

    public DroneController droneController;

    [Header("Victim Settings")]
    public int victimCount = 10;          // Number of victims to spawn each episode

    [Header("Environment Bounds")]
    public Vector3 arenaSize = new Vector3(50f, 20f, 50f);
    public Vector3 arenaCenter = Vector3.zero;

    private GameObject[] activeObstacles;
    private GameObject[] activeVictims;      // Spawned victim instances
    private EnvironmentParameters envParams;
    private Rigidbody droneRb;
    private GameObject nearestVictim;
    public bool isInitializing = true;

    void Start()
    {
        // Get ML-Agents environment parameters
        envParams = Academy.Instance.EnvironmentParameters;

        if (drone != null)
        {
            droneRb = drone.GetComponent<Rigidbody>();
        }

        // Initial setup
        UpdateEnvironment();
    }



    /// <summary>
    /// Called at episode start to update environment based on curriculum
    /// </summary>

    void SpawnVictims()
    {
        // Destroy previous victims
        if (activeVictims != null)
        {
            foreach (GameObject v in activeVictims)
            {
                if (v != null) Destroy(v);
            }
        }

        activeVictims = new GameObject[victimCount];

        if (victimPrefab == null || targetDistance == 0f)
        {
            return;
        }

        for (int i = 0; i < victimCount; i++)
        {
            Vector3 spawnPos = GetValidVictimPosition();
            activeVictims[i] = Instantiate(victimPrefab, spawnPos, Quaternion.identity);
            activeVictims[i].transform.SetParent(transform);
            activeVictims[i].tag = "Victim"; 
        }
    }

    Vector3 GetValidVictimPosition()
    {
        float minSpacing = 3f;
        int maxAttempts = 30;

        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            // Random direction on unit sphere, keep above ground
            Vector3 randomDir = Random.onUnitSphere;
            randomDir.y = Mathf.Abs(randomDir.y) + 0.1f;
            randomDir.Normalize();

            // Random distance up to targetDistance
            float dist = Random.Range(targetDistance * 0.4f, targetDistance);
            Vector3 candidate = arenaCenter + randomDir * dist;

            // Clamp inside arena
            candidate.x = Mathf.Clamp(candidate.x, arenaCenter.x - arenaSize.x / 2f + 1f, arenaCenter.x + arenaSize.x / 2f - 1f);
            candidate.y = Mathf.Clamp(candidate.y, 1.5f, arenaSize.y - 1f);
            candidate.z = Mathf.Clamp(candidate.z, arenaCenter.z - arenaSize.z / 2f + 1f, arenaCenter.z + arenaSize.z / 2f - 1f);

            // Check spacing from already placed victims
            bool tooClose = false;
            for (int j = 0; j < activeVictims.Length; j++)
            {
                if (activeVictims[j] != null && Vector3.Distance(candidate, activeVictims[j].transform.position) < minSpacing)
                {
                    tooClose = true;
                    break;
                }
            }
            if (tooClose) continue;

            return candidate;
        }

        // Fallback: place at random inside arena
        return new Vector3(
            Random.Range(arenaCenter.x - arenaSize.x / 2f + 2f, arenaCenter.x + arenaSize.x / 2f - 2f),
            Random.Range(2f, arenaSize.y - 2f),
            Random.Range(arenaCenter.z - arenaSize.z / 2f + 2f, arenaCenter.z + arenaSize.z / 2f - 2f)
        );
    }

    public GameObject[] GetActiveVictims()
    {
        if (activeVictims == null) return new GameObject[0];
        // Return only non-null, active ones
        var list = new System.Collections.Generic.List<GameObject>();
        foreach (var v in activeVictims)
        {
            if (v != null && v.activeSelf) list.Add(v);
        }
        return list.ToArray();
    }

    public int RemainingVictims()
    {
        int count = 0;
        if (activeVictims == null) return 0;
        foreach (var v in activeVictims)
        {
            if (v != null && v.activeSelf) count++;
        }
        return count;
    }

    /// <summary>
    /// Returns the nearest active victim transform, or null if none.
    /// </summary>
    public Transform GetNearestVictim(Vector3 fromPosition)
    {
        Transform nearest = null;
        float nearestDist = Mathf.Infinity;
        if (activeVictims == null) return null;

        foreach (var v in activeVictims)
        {
            if (v == null || !v.activeSelf) continue;
            float d = Vector3.Distance(fromPosition, v.transform.position);
            if (d < nearestDist)
            {
                nearestDist = d;
                nearest = v.transform;
                nearestVictim = v;
            }
        }
        return nearest;
    }

    public bool IsVictimReached(Vector3 dronePosition, out GameObject rescuedVictim)
    {
        rescuedVictim = null;
        if (activeVictims == null) return false;
        foreach (var v in activeVictims)
        {
            if (v == null || !v.activeSelf) continue;
            
            if (Vector3.Distance(dronePosition, v.transform.position) <= successRadius)
            {
                
                rescuedVictim = v;
                return true;  // DO NOT deactivate here — let DroneController call RescueVictim()
            }
            if (v != nearestVictim) {
                // Debug.DrawLine(
                //     dronePosition,
                //     v.transform.position,
                //     Color.magenta
                // );
            }else{
                // Debug.DrawLine(
                //     dronePosition,
                //     v.transform.position,
                //     Color.white
                // );
            }
        }
        return false;
    }

    public void RescueVictim(GameObject victim)
    {
        if (victim != null && victim.activeSelf)
        {
            Debug.Log($"[RescueVictim] Rescued: {victim.name}");
            victim.SetActive(false);
        }
    }

    bool IsTooCloseToKeyObjects(Vector3 pos, float minDist)
    {
        // Check drone spawn
        if (drone != null && Vector3.Distance(pos, drone.position) < minDist)
            return true;

        // Check all active victims
        if (activeVictims != null)
        {
            foreach (var v in activeVictims)
            {
                if (v != null && v.activeSelf && Vector3.Distance(pos, v.transform.position) < minDist)
                    return true;
            }
        }

        return false;
    }

    void SpawnObstacles()
    {
        if (activeObstacles != null)
        {
            foreach (GameObject obs in activeObstacles)
                if (obs != null) Destroy(obs);
        }

        if (obstaclePrefab == null || obstacleCount == 0)
        {
            activeObstacles = new GameObject[0];
            return;
        }

        activeObstacles = new GameObject[obstacleCount];
        const float minDistance = 3f;
        const int maxAttempts = 50;

        for (int i = 0; i < obstacleCount; i++)
        {
            Vector3 obstaclePos = Vector3.zero;
            bool placed = false;

            for (int attempt = 0; attempt < maxAttempts; attempt++)
            {
                obstaclePos = new Vector3(
                    Random.Range(arenaCenter.x - arenaSize.x / 2f + 1f, arenaCenter.x + arenaSize.x / 2f - 1f),
                    -2f,
                    Random.Range(arenaCenter.z - arenaSize.z / 2f + 1f, arenaCenter.z + arenaSize.z / 2f - 1f)
                );

                if (!IsTooCloseToKeyObjects(obstaclePos, minDistance))
                {
                    placed = true;
                    break;
                }
            }

            if (!placed)
                Debug.LogWarning($"[SpawnObstacles] No valid pos for obstacle {i} after {maxAttempts} attempts, placing at fallback.");

            Quaternion randomYRotation = Quaternion.Euler(0, Random.Range(0f, 360f), 0);
            activeObstacles[i] = Instantiate(obstaclePrefab, obstaclePos, randomYRotation);
            activeObstacles[i].transform.SetParent(transform);
            activeObstacles[i].tag = "Obstacle"; // ← required for ray sensor + reward shaping
        }
    }

    public void SpawnDrone()
    {
        drone.position = new Vector3(
            Random.Range(arenaCenter.x - 5f, arenaCenter.x + 5f),
            spawnHeight,
            Random.Range(arenaCenter.z - 5f, arenaCenter.z + 5f)
        );
    }

    IEnumerator LaunchDroneAfterVictimsSettled()
    {
        float timeout = 5f;
        float elapsed = 0f;

        while (elapsed < timeout)
        {
            bool allSettled = true;

            foreach (GameObject victim in activeVictims)
            {
                Rigidbody victimRb = victim.GetComponent<Rigidbody>();
                if (victimRb != null &&
                    (!victimRb.IsSleeping() || victimRb.linearVelocity.magnitude > 0.05f))
                {
                    allSettled = false;
                    break;
                }
            }

            if (allSettled)
                break;

            yield return new WaitForFixedUpdate();
            elapsed += Time.fixedDeltaTime;
        }

        yield return new WaitForFixedUpdate();

        SpawnDrone();
        isInitializing = false;
        droneController.ResetDrone();
    }

    void DeactivateDrone()
    {
        drone.position = new Vector3(0, -100, 0); // Move off-screen
        droneRb.Sleep();
    }


    public bool IsOutOfBounds(Vector3 position)
    {
        return Mathf.Abs(position.x - arenaCenter.x) > arenaSize.x / 2 ||
               position.y < 0 || position.y > arenaSize.y ||
               Mathf.Abs(position.z - arenaCenter.z) > arenaSize.z / 2;
    }

    public void UpdateEnvironment()
    { 
        isInitializing = true;
        // Read curriculum parameters from ML-Agents
        targetDistance = envParams.GetWithDefault("target_distance", 2f);
        obstacleCount = Mathf.RoundToInt(envParams.GetWithDefault("obstacle_count", 8f));


        // Update environment
        
        DeactivateDrone();
        SpawnVictims();
        SpawnObstacles();
        if(targetDistance > 0f)StartCoroutine(LaunchDroneAfterVictimsSettled());
        else {
            isInitializing = false;
            SpawnDrone();
        }
        Debug.Log($"[Curriculum] Stage: Distance={targetDistance}, Obstacles={obstacleCount} | Fixed: EpisodeLen={episodeLength}, SpawnH={spawnHeight}, SuccessR={successRadius}");
    }

    void OnDrawGizmos()
    {
        // Draw arena bounds
        Gizmos.color = Color.cyan;
        Gizmos.DrawWireCube(arenaCenter, arenaSize);

        // Draw success radius around each active victim
        if (activeVictims != null)
        {
            Gizmos.color = Color.green;
            foreach (var v in activeVictims)
            {
                if (v != null && v.activeSelf)
                    Gizmos.DrawWireSphere(v.transform.position, successRadius);
            }
        }
    }
}
