using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;
using System.IO;
using System;

public class DroneController : Agent
{
    // ── Physics ────────────────────────────────────────────────────────
    private Rigidbody rb;
    public float thrust        = 15f;
    public float maxTiltAngle  = 30f;
    public float tiltSpeed     = 8f;
    public float maxYawSpeed   = 120f;
    public float hoverForce    = 9.81f;
    public float drag          = 2f;
    public float angularDrag   = 5f;
    public float maxVelocity   = 15f;
    public bool  test          = false;
    public bool  recordEvaluationMetrics = false;
    public Camera droneCamera;
    public Animator[] fanAnimators;

    float yaw, pitch, roll, throttle;
    public float actionSmoothing = 10f;
    private float smoothThrottle, smoothRoll, smoothPitch, smoothYaw;
    public bool hoverMode = true;

    // ── Scene refs ─────────────────────────────────────────────────────
    private Vector3    startPosition;
    private Quaternion startRotation;
    private DroneEnvironmentConfig envConfig;
    private GameObject[] cachedObstacles;
    private string logFilePath;
    private int groundMask;

    // ── Episode stats ──────────────────────────────────────────────────
    private int   episodeCount      = 0;
    private float episodeStartTime;
    private int   stepCount         = 0;
    private float episodeReward     = 0f;
    private float minDistanceAchieved = float.MaxValue;
    private List<float> distanceToTargetHistory = new List<float>();

    private int   successCount      = 0;
    private int   crashCount        = 0;
    private int   timeoutCount      = 0;
    private int   outOfBoundsCount  = 0;
    private int   goalsReached      = 0;
    private int   groundCollisionCount = 0;
    private float distanceTraveled  = 0f;
    private float shortestPath      = 0f;
    private string endReason        = "start";

    [Header("Test Mode")]
    public int testEpisodeLimit  = 100;
    private int testEpisodesRun  = 0;

    // ── HP ─────────────────────────────────────────────────────────────
    public float droneHP         = 500f;
    public const float MAX_HP    = 500f;
    private float softLandingThreshold = 2f;
    private float hardLandingThreshold = 5f;
    private float lastStayDamageTime   = 0f;

    // ── Navigation state ───────────────────────────────────────────────
    private float previousDistance    = Mathf.Infinity;
    private float closestDistanceEver = Mathf.Infinity;
    private int   backtrackCounter    = 0;
    private HashSet<Vector3Int> visitedCells = new HashSet<Vector3Int>();
    private int   hoverCounter        = 0;
    private Vector3 lastHoverCheck;
    private const float GRID_CELL_SIZE = 3f;
    private string _endReason         = "start";
    private float groundDistance      = 20f;
    private bool  isNearGround        = false;
    private Vector3 lastPos;

    // ── Visibility state ───────────────────────────────────────────────
    private bool isVictimVisible         = false;
    private bool wasVictimVisibleLastStep = false;

    // ── Target Lock ────────────────────────────────────────────────────
    private bool  isLockedOn       = false;
    private float lockedDirX       = 0.5f;
    private float lockedDirY       = 0.5f;
    private float lockedDistance   = 0f;
    private int   lockLostCounter  = 0;
    private const int   LOCK_LOST_THRESHOLD = 10;
    private const float DISTANCE_SCALE      = 30f;
    private Vector3 lockedEstimatedPos      = Vector3.zero;

    [Header("YOLO Calibration")]
    public float yoloHorizMultiplier = 1.09f;
    public float yoloVertMultiplier  = 1.09f;
    public float yoloForwardBias     = 1.0f;

    // ── YOLO Channel ───────────────────────────────────────────────────
    private YoloUDPReceiver yoloChannel;

    // ══════════════════════════════════════════════════════════════════════
    // DEBUG RAY VISUALIZATION
    // All visualization is controlled from the Inspector.
    // Nothing here uses GPS/ground truth during training.
    // ══════════════════════════════════════════════════════════════════════

    [Header("──── Debug Rays ────")]
    [Tooltip("Master switch — turn off for final training runs")]
    public bool showDebugRays = true;

    [Header("  Center Ray")]
    public bool  showCenterRay   = true;
    public Color centerRayColor  = new Color(0f, 1f, 0f, 1f);     // green

    [Header("  Spread Rays")]
    public bool  showSpreadRays  = true;
    public Color spreadRayColor  = new Color(0f, 0.8f, 1f, 0.8f); // cyan
    public Color spreadHitColor  = new Color(1f, 0.4f, 0f, 1f);   // orange on hit

    [Header("  Ground Ray")]
    public bool  showGroundRay   = true;
    public Color groundRayColor  = new Color(0.2f, 1f, 0.2f, 1f); // bright green
    public Color groundMissColor = new Color(1f, 0.2f, 0.2f, 1f); // red on miss

    [Header("  Estimated Position Sphere")]
    public bool  showEstimatedSphere    = true;
    public Color estimatedSphereColor   = new Color(0f, 1f, 1f, 1f);  // cyan
    public float estimatedSphereRadius  = 0.6f;

    [Header("  Lock Lost Ray (dashed-style)")]
    public bool  showLockLostIndicator  = true;
    public Color lockLostColor          = new Color(1f, 0f, 0f, 0.5f); // red semi-transparent

    [Header("  Obstacle Threat Rays")]
    public bool  showObstacleThreatRays = true;
    public Color obstacleThreatColor    = new Color(1f, 0f, 0.5f, 1f); // magenta

    [Header("  Velocity Vector")]
    public bool  showVelocityVector  = true;
    public Color velocityVectorColor = new Color(1f, 1f, 0f, 1f);     // yellow
    public float velocityScale       = 0.3f;

    [Header("  Drone Up Axis")]
    public bool  showUpAxis   = true;
    public Color upAxisColor  = new Color(0.5f, 0.5f, 1f, 1f);        // light blue
    public float upAxisLength = 2f;

    // ── Spread offsets in VIEWPORT space (0-1 coords, relative to bbox center)
    // These are ADDED to (lockedDirX, lockedDirY) when building spread rays.
    // Keep small — we're sampling around the detected bbox center, not far from it.
    private static readonly Vector2[] SPREAD_OFFSETS = {
        new Vector2( 0f,      0f),       // [0] center — primary ray
        new Vector2( 0.03f,   0f),       // [1] right
        new Vector2(-0.03f,   0f),       // [2] left
        new Vector2( 0f,      0.03f),    // [3] up   (viewport +Y = image up)
        new Vector2( 0f,     -0.03f),    // [4] down
        new Vector2( 0.02f,   0.02f),    // [5] TR diagonal
        new Vector2(-0.02f,  -0.02f),    // [6] BL diagonal
        new Vector2( 0.02f,  -0.02f),    // [7] BR diagonal
        new Vector2(-0.02f,   0.02f),    // [8] TL diagonal
    };

    // Tracks which spread hits came from direct Victim-layer contact (most trusted)
    private bool[] _spreadHitVictimDirect = new bool[9];

    // ══════════════════════════════════════════════════════════════════════
    // INITIALIZE
    // ══════════════════════════════════════════════════════════════════════

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        rb.linearDamping  = drag;
        rb.angularDamping = angularDrag;
        groundMask  = LayerMask.GetMask("Ground");
        envConfig   = FindFirstObjectByType<DroneEnvironmentConfig>();
        startPosition = transform.position;
        startRotation = transform.rotation;

        if (droneCamera == null)
            droneCamera = GetComponentInChildren<Camera>();

        if (droneCamera != null)
            Debug.Log($"[CAM AXES] forward={droneCamera.transform.forward} " +
                      $"up={droneCamera.transform.up} right={droneCamera.transform.right}");

        yoloChannel = GetComponent<YoloUDPReceiver>();

        if (recordEvaluationMetrics)
        {
            logFilePath = Path.Combine(Application.dataPath, "..", "Drone_Test_Results.csv");
            if (!File.Exists(logFilePath))
            {
                try { File.WriteAllText(logFilePath,
                    "Episode,VictimsRescued,TotalVictims,StepsTaken,PathEfficiency,DistanceTraveled,EndReason,DroneHP\n"); }
                catch (Exception e) { Debug.LogError("CSV Init Error: " + e.Message); }
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // EPISODE BEGIN
    // ══════════════════════════════════════════════════════════════════════

    public override void OnEpisodeBegin()
    {
        LogTestRun();
        episodeCount++;

        if (recordEvaluationMetrics)
        {
            testEpisodesRun++;
            if (testEpisodesRun > testEpisodeLimit)
            {
                Debug.Log($"[Test Complete] {testEpisodesRun - 1} episodes logged to: {logFilePath}");
#if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false;
#else
                Application.Quit();
#endif
                return;
            }
        }

        if (envConfig != null)
        {
            envConfig.UpdateEnvironment();
            cachedObstacles = GameObject.FindGameObjectsWithTag("Obstacle");
            transform.rotation = Quaternion.identity;
        }

        episodeStartTime  = Time.time;
        stepCount         = 0;
        episodeReward     = 0f;
        droneHP           = MAX_HP;
        minDistanceAchieved = float.MaxValue;
        distanceToTargetHistory.Clear();
        smoothThrottle = smoothRoll = smoothPitch = smoothYaw = 0f;
        previousDistance    = Mathf.Infinity;
        closestDistanceEver = Mathf.Infinity;
        backtrackCounter    = 0;
        visitedCells.Clear();
        hoverCounter        = 0;
        lastHoverCheck      = Vector3.zero;
        distanceTraveled    = 0f;
        endReason           = "start";
        goalsReached        = 0;
        groundCollisionCount = 0;
        shortestPath        = 0f;
        isVictimVisible     = false;
        wasVictimVisibleLastStep = false;
        lastStayDamageTime  = 0f;
        lastPos             = transform.localPosition;
        lockedEstimatedPos  = Vector3.zero;
        shortestPath        = CalculateShortestPath();
        lastHoverCheck      = transform.localPosition;
        ResetLock();
    }

    // ══════════════════════════════════════════════════════════════════════
    // LOCK MANAGEMENT
    // ══════════════════════════════════════════════════════════════════════

    private void ResetLock()
    {
        isLockedOn         = false;
        lockedDirX         = 0.5f;
        lockedDirY         = 0.5f;
        lockedDistance     = 0f;
        lockLostCounter    = 0;
        lockedEstimatedPos = Vector3.zero;
    }

    // ══════════════════════════════════════════════════════════════════════
    // TARGET LOCK + RAY CASTING  (the core of the observation pipeline)
    // ══════════════════════════════════════════════════════════════════════

    private void UpdateTargetLock()
    {
        // ══════════════════════════════════════════════════════════════
        // STEP 1 — Update lock state from smoothed YOLO detections
        // Raw YOLO is noisy frame-to-frame. We only update the locked
        // values when confidence is high; once locked we lerp slowly
        // so individual bad frames can't corrupt the estimate.
        // ══════════════════════════════════════════════════════════════
        if (yoloChannel.VictimDetected > 0.5f && yoloChannel.VictimConfidence > 0.45f)
        {
            if (!isLockedOn)
            {
                isLockedOn      = true;
                lockedDirX      = yoloChannel.VictimDirX;
                lockedDirY      = yoloChannel.VictimDirY;
                lockedDistance  = yoloChannel.VictimDistance;
                lockLostCounter = 0;
                Debug.Log($"[LOCK] Acquired  DirX={lockedDirX:F3} DirY={lockedDirY:F3} Dist={lockedDistance:F3}");
            }
            else
            {
                // Low lerp factor (0.2) = heavy smoothing — absorbs YOLO jitter
                lockedDirX      = Mathf.Lerp(lockedDirX,      yoloChannel.VictimDirX,      0.2f);
                lockedDirY      = Mathf.Lerp(lockedDirY,      yoloChannel.VictimDirY,      0.2f);
                lockedDistance  = Mathf.Lerp(lockedDistance,  yoloChannel.VictimDistance,   0.2f);
                lockLostCounter = 0;
            }
        }
        else
        {
            if (isLockedOn)
            {
                lockLostCounter++;
                if (lockLostCounter >= LOCK_LOST_THRESHOLD)
                {
                    isLockedOn      = false;
                    lockLostCounter = 0;
                    Debug.Log("[LOCK] Lost — victim not detected for threshold frames");
                }
            }
        }

        // Always draw the downward ground ray and velocity vector regardless of lock
        if (showDebugRays)
        {
            if (showGroundRay)
            {
                bool gHit = Physics.Raycast(transform.position, Vector3.down,
                    out RaycastHit gResult, 20f, groundMask);
                Debug.DrawRay(transform.position,
                    Vector3.down * (gHit ? gResult.distance : 20f),
                    gHit ? groundRayColor : groundMissColor);
            }
            if (showVelocityVector && rb != null)
                Debug.DrawRay(transform.position, rb.linearVelocity * velocityScale, velocityVectorColor);
            if (showUpAxis)
                Debug.DrawRay(transform.position, transform.up * upAxisLength, upAxisColor);

            if (!isLockedOn && showLockLostIndicator)
                Debug.DrawRay(transform.position, transform.forward * 3f, lockLostColor);
        }

        if (!isLockedOn || droneCamera == null) return;

        // ══════════════════════════════════════════════════════════════
        // STEP 2 — Convert smoothed bbox centre (u,v) → world ray dirs
        //
        // lockedDirX/Y are YOLO viewport coords in [0,1]:
        //   u=0 left edge, u=1 right edge, v=0 top, v=1 bottom
        //
        // camera.ViewportPointToRay(u, 1-v, 0) gives the exact ray
        // Unity's Camera API handles all FOV/aspect math correctly —
        // this replaces the manual forward+right*camX construction
        // which was wrong because it ignored FOV scaling.
        //
        // Ray ORIGIN = droneCamera.transform.position, NOT transform.position
        // These differ by the camera offset mount on the drone body.
        // ══════════════════════════════════════════════════════════════
        Vector3 camOrigin = droneCamera.transform.position;

        // Viewport point: Unity viewport has Y=0 at bottom, but YOLO has Y=0 at top
        // So flip Y: viewportY = 1 - lockedDirY
        Vector3 centerViewport = new Vector3(lockedDirX, 1f - lockedDirY, 0f);
        Ray     centerRay      = droneCamera.ViewportPointToRay(centerViewport);
        Vector3 centerRayDir   = centerRay.direction.normalized;

        // ══════════════════════════════════════════════════════════════
        // STEP 3 — Shoot 9-ray spread around the bbox centre
        //
        // Each spread ray samples a viewport position near (u,v).
        // SPREAD_OFFSETS are in viewport units — small (0.02-0.03)
        // so we stay close to the detected bbox.
        //
        // For each ray, priority:
        //   P1: direct Victim-layer hit → use collider.bounds.center
        //       (avoids capsule top-dome surface bias)
        //   P2: ground hit → lift by victim half-height
        //       (victim standing on ground, ray missed capsule)
        //   P3: altitude projection
        //       (ray going upward or no geometry — last resort)
        // ══════════════════════════════════════════════════════════════
        int victimMask = LayerMask.GetMask("Victim");

        // Per-ray results (parallel arrays, indexed by SPREAD_OFFSETS index)
        var spreadDirs     = new Vector3[SPREAD_OFFSETS.Length];
        var spreadHitPos   = new Vector3[SPREAD_OFFSETS.Length];
        var spreadHitValid = new bool[SPREAD_OFFSETS.Length];
        var spreadHitDirect= new bool[SPREAD_OFFSETS.Length]; // true = P1 Victim hit

        for (int i = 0; i < SPREAD_OFFSETS.Length; i++)
        {
            Vector2 off = SPREAD_OFFSETS[i];
            // Build viewport point for this spread sample
            // Note: offset.y is in viewport-Y (up positive), already consistent
            var vp  = new Vector3(
                Mathf.Clamp01(lockedDirX  + off.x),
                Mathf.Clamp01(1f - lockedDirY + off.y),   // flip Y for Unity viewport
                0f);
            Ray ray = droneCamera.ViewportPointToRay(vp);
            Vector3 dir = ray.direction.normalized;
            spreadDirs[i] = dir;

            // P1: Victim layer direct hit
            if (Physics.Raycast(camOrigin, dir, out RaycastHit hV, 60f, victimMask))
            {
                spreadHitPos[i]    = hV.collider.bounds.center; // body center, not surface
                spreadHitValid[i]  = true;
                spreadHitDirect[i] = true;
            }
            // P2: Ground hit → infer victim is standing here
            else if (Physics.Raycast(camOrigin, dir, out RaycastHit hG, 100f, groundMask))
            {
                Vector3 gp    = hG.point;
                gp.y         += 1.0f;    // victim capsule half-height ≈ 0.9 * scale
                spreadHitPos[i]    = gp;
                spreadHitValid[i]  = true;
                spreadHitDirect[i] = false;
            }
            // P3: Altitude projection fallback
            else
            {
                float t = groundDistance / Mathf.Max(-dir.y, 0.01f);
                t = Mathf.Clamp(t, 1f, 50f);
                spreadHitPos[i]    = camOrigin + dir * t;
                spreadHitValid[i]  = true;   // position exists but least trusted
                spreadHitDirect[i] = false;
            }
        }

        // ══════════════════════════════════════════════════════════════
        // STEP 4 — Trimmed mean position estimate
        //
        // Strategy:
        //   a) If any P1 (direct Victim) hits exist → use ONLY those.
        //      Direct hits are ground truth, don't average with fallbacks.
        //   b) Otherwise use all valid hits.
        //   c) Compute mean, then drop any point more than 2m from mean
        //      (outlier rejection), recompute final mean.
        // ══════════════════════════════════════════════════════════════
        var candidates = new List<Vector3>();

        // Prefer direct Victim hits exclusively when available
        bool anyDirect = false;
        for (int i = 0; i < SPREAD_OFFSETS.Length; i++)
            if (spreadHitValid[i] && spreadHitDirect[i]) { anyDirect = true; break; }

        for (int i = 0; i < SPREAD_OFFSETS.Length; i++)
        {
            if (!spreadHitValid[i]) continue;
            if (anyDirect && !spreadHitDirect[i]) continue; // skip fallbacks when direct hits available
            candidates.Add(spreadHitPos[i]);
        }

        Vector3 estimatedPos;
        if (candidates.Count == 0)
        {
            // Pure fallback: project center ray to YOLO estimated range
            float estRange = lockedDistance > 0.01f
                ? lockedDistance * DISTANCE_SCALE
                : groundDistance;
            estimatedPos = camOrigin + centerRayDir * Mathf.Clamp(estRange, 1f, 50f);
        }
        else
        {
            // First-pass mean
            Vector3 mean = Vector3.zero;
            foreach (var p in candidates) mean += p;
            mean /= candidates.Count;

            // Outlier rejection: drop points > 2m from first-pass mean
            var inliers = new List<Vector3>();
            foreach (var p in candidates)
                if (Vector3.Distance(p, mean) < 2f) inliers.Add(p);

            // If all were outliers (shouldn't happen), fall back to full mean
            var final = inliers.Count > 0 ? inliers : candidates;
            estimatedPos = Vector3.zero;
            foreach (var p in final) estimatedPos += p;
            estimatedPos /= final.Count;
        }

        lockedEstimatedPos = estimatedPos;

        // ══════════════════════════════════════════════════════════════
        // STEP 5 — Debug visualization
        // All Debug.DrawLine calls persist for exactly one frame (duration=0)
        // so they're visible in Scene view with Gizmos on and in Game view
        // with the Gizmos toolbar button enabled.
        // Toggle master switch with F1 at runtime, or via Inspector.
        // ══════════════════════════════════════════════════════════════
        if (showDebugRays)
        {
            // Center ray — green line from camera to estimated position
            if (showCenterRay)
                Debug.DrawLine(camOrigin, lockedEstimatedPos, centerRayColor);

            // Spread rays — individually coloured by hit type
            if (showSpreadRays)
            {
                for (int i = 0; i < SPREAD_OFFSETS.Length; i++)
                {
                    if (i == 0) continue; // center already drawn above
                    Color c = spreadHitDirect[i] ? spreadHitColor   // orange = direct victim
                            : spreadHitValid[i]  ? spreadRayColor   // cyan   = ground/fallback
                            : lockLostColor;                         // red    = no hit
                    Debug.DrawLine(camOrigin, spreadHitPos[i], c);
                    // Small cross at each hit point for clarity
                    DrawCross(spreadHitPos[i], 0.15f, c);
                }
            }

            // Estimated position sphere — cyan wireframe
            if (showEstimatedSphere)
                DrawWireSphere(lockedEstimatedPos, estimatedSphereRadius, estimatedSphereColor);

            // Obstacle threat lines — magenta lines to nearby obstacles in drone's path
            if (showObstacleThreatRays && cachedObstacles != null)
            {
                foreach (var obs in cachedObstacles)
                {
                    if (obs == null || !obs.activeSelf) continue;
                    float d = Vector3.Distance(transform.position, obs.transform.position);
                    if (d > 8f) continue;
                    float dot = Vector3.Dot(
                        (obs.transform.position - transform.position).normalized,
                        centerRayDir);
                    if (dot > 0.5f)
                        Debug.DrawLine(transform.position, obs.transform.position, obstacleThreatColor);
                }
            }
        }

        // ══════════════════════════════════════════════════════════════
        // STEP 6 — Calibration logging (editor only, never during training)
        // Compares our estimated position against actual GPS victim pos.
        // Use this to tune yoloHorizMultiplier / yoloVertMultiplier.
        // ══════════════════════════════════════════════════════════════
#if UNITY_EDITOR
        if (Time.frameCount % 30 == 0)
        {
            Vector3 actualPos = FindClosestVictimPosition();
            if (actualPos != Vector3.zero)
            {
                float posError   = Vector3.Distance(lockedEstimatedPos, actualPos);
                float estDist    = Vector3.Distance(camOrigin, lockedEstimatedPos);
                float actualDist = Vector3.Distance(camOrigin, actualPos);
                string hitType   = anyDirect ? "DIRECT" : "FALLBACK";
                Debug.Log($"[CALIB | {hitType}] PosErr={posError:F2}m  " +
                          $"EstDist={estDist:F2}  ActualDist={actualDist:F2}  " +
                          $"Candidates={candidates.Count}  " +
                          $"u={lockedDirX:F3} v={lockedDirY:F3}");

                // Draw line from estimated to actual so you can see the error vector
                Debug.DrawLine(lockedEstimatedPos, actualPos, Color.magenta);
            }

            Debug.Log($"[LOCK] Det={yoloChannel.VictimDetected:F2} " +
                      $"Conf={yoloChannel.VictimConfidence:F2} " +
                      $"u={yoloChannel.VictimDirX:F3} v={yoloChannel.VictimDirY:F3} " +
                      $"NormDist={yoloChannel.VictimDistance:F3} Locked={isLockedOn}");
        }
#endif
    }

    // ══════════════════════════════════════════════════════════════════════
    // OBSERVATIONS  (21 values — unchanged count, improved quality)
    // ══════════════════════════════════════════════════════════════════════

    public override void CollectObservations(VectorSensor sensor)
    {
        if (envConfig != null && envConfig.isInitializing)
        {
            sensor.AddObservation(Vector3.zero);        // 3
            sensor.AddObservation(Vector3.zero);        // 3
            sensor.AddObservation(Vector3.zero);        // 3
            sensor.AddObservation(Quaternion.identity); // 4
            sensor.AddObservation(0f);                  // detected
            sensor.AddObservation(0f);                  // dir_x
            sensor.AddObservation(0f);                  // dir_y
            sensor.AddObservation(0f);                  // distance
            sensor.AddObservation(0f);                  // confidence
            sensor.AddObservation(0f);                  // progress
            sensor.AddObservation(0f);                  // remaining
            sensor.AddObservation(0f);                  // ground
            return;                                     // total: 21
        }

        // ── Drone state (13) ──────────────────────────────────────────
        Vector3 normalizedPos = (transform.localPosition - envConfig.arenaCenter);
        normalizedPos.x /= envConfig.arenaSize.x;
        normalizedPos.y /= envConfig.arenaSize.y;
        normalizedPos.z /= envConfig.arenaSize.z;
        sensor.AddObservation(normalizedPos);                                                        // 3

        sensor.AddObservation(transform.InverseTransformDirection(rb.linearVelocity) / maxVelocity); // 3

        float maxYawRad = maxYawSpeed * Mathf.Deg2Rad;
        sensor.AddObservation(rb.angularVelocity / maxYawRad);                                      // 3

        sensor.AddObservation(transform.localRotation);                                             // 4

        // ── Victim detection — lock-stabilised (5) ────────────────────
        // UpdateTargetLock runs first; ALL 5 obs come from the lock.
        // Raw YOLO is never fed to the policy directly.
        UpdateTargetLock();

        if (isLockedOn && lockedEstimatedPos != Vector3.zero)
        {
            // Feed bearing/elevation in drone-local space — rotation invariant
            Vector3 localOffset = transform.InverseTransformPoint(lockedEstimatedPos);
            float bearing   = Mathf.Atan2(localOffset.x, localOffset.z) / Mathf.PI;        // -1..1
            float elevation = Mathf.Atan2(localOffset.y,
                                new Vector2(localOffset.x, localOffset.z).magnitude)
                                / (Mathf.PI * 0.5f);                                        // -1..1

            sensor.AddObservation(1f);                                // detected flag   (1)
            sensor.AddObservation(bearing);                           // bearing         (1)
            sensor.AddObservation(elevation);                         // elevation       (1)
            sensor.AddObservation(localOffset.magnitude
                / Mathf.Max(envConfig.arenaSize.x, envConfig.arenaSize.z)); // dist norm (1)
            sensor.AddObservation(yoloChannel.VictimConfidence);      // confidence      (1)
        }
        else
        {
            sensor.AddObservation(0f); sensor.AddObservation(0f);
            sensor.AddObservation(0f); sensor.AddObservation(0f);
            sensor.AddObservation(0f);                                                              // 5
        }

        isVictimVisible = isLockedOn;

        // ── Mission progress (2) ──────────────────────────────────────
        int totalVictims = envConfig != null ? envConfig.victimCount : 1;
        sensor.AddObservation((float)goalsReached / Mathf.Max(totalVictims, 1));                    // 1
        int remaining = envConfig != null ? envConfig.RemainingVictims() : 0;
        sensor.AddObservation(remaining / (float)Mathf.Max(totalVictims, 1));                       // 1

        // ── Ground proximity (1) ──────────────────────────────────────
        if (Physics.Raycast(transform.position, Vector3.down,
            out RaycastHit groundHit, 20f, groundMask))
        {
            groundDistance = groundHit.distance;
        }
        else
        {
            groundDistance = 20f;
        }
        isNearGround = groundDistance < 3f;
        sensor.AddObservation(groundDistance / 20f);                                                // 1

        // total: 21
    }

    // ══════════════════════════════════════════════════════════════════════
    // PHYSICS UPDATE
    // ══════════════════════════════════════════════════════════════════════

    void FixedUpdate()
    {
        if (envConfig != null && envConfig.isInitializing) return;

        Vector3 currentEuler   = transform.rotation.eulerAngles;
        float newYaw           = currentEuler.y + (yaw * maxYawSpeed * Time.fixedDeltaTime);
        Quaternion targetRot   = Quaternion.Euler(pitch * maxTiltAngle, newYaw, -roll * maxTiltAngle);
        rb.MoveRotation(Quaternion.Slerp(transform.rotation, targetRot, Time.fixedDeltaTime * tiltSpeed));

        float tiltAngle        = Vector3.Angle(Vector3.up, transform.up);
        float tiltComp         = 1f / Mathf.Max(Mathf.Cos(tiltAngle * Mathf.Deg2Rad), 0.2f);
        float gravityCounter   = Physics.gravity.magnitude * tiltComp;
        float totalThrust      = gravityCounter + (throttle * thrust);
        rb.AddForce(transform.up * totalThrust, ForceMode.Acceleration);

        if (rb.linearVelocity.magnitude > maxVelocity)
            rb.linearVelocity = rb.linearVelocity.normalized * maxVelocity;
    }

    // ══════════════════════════════════════════════════════════════════════
    // ACTIONS
    // ══════════════════════════════════════════════════════════════════════

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (envConfig != null && envConfig.isInitializing) return;

        float rawThrottle = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float rawRoll     = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        float rawPitch    = Mathf.Clamp(actions.ContinuousActions[2], -1f, 1f);
        float rawYaw      = Mathf.Clamp(actions.ContinuousActions[3], -1f, 1f);

        float actionDelta = Mathf.Abs(rawThrottle - smoothThrottle)
                          + Mathf.Abs(rawRoll     - smoothRoll)
                          + Mathf.Abs(rawPitch    - smoothPitch)
                          + Mathf.Abs(rawYaw      - smoothYaw);

        if (actionDelta > 0.5f)
        {
            float penalty = actionDelta * 0.002f;
            AddReward(-penalty);
            episodeReward -= penalty;
        }

        smoothThrottle = Mathf.Lerp(smoothThrottle, rawThrottle, Time.fixedDeltaTime * actionSmoothing);
        smoothRoll     = Mathf.Lerp(smoothRoll,     rawRoll,     Time.fixedDeltaTime * actionSmoothing);
        smoothPitch    = Mathf.Lerp(smoothPitch,    rawPitch,    Time.fixedDeltaTime * actionSmoothing);
        smoothYaw      = Mathf.Lerp(smoothYaw,      rawYaw,      Time.fixedDeltaTime * actionSmoothing);

        throttle = smoothThrottle;
        roll     = smoothRoll;
        pitch    = smoothPitch;
        yaw      = smoothYaw;

        stepCount++;
        CalculateRewards();
        GiveScanningRewards();
        CheckTerminationConditions();
    }

    // ══════════════════════════════════════════════════════════════════════
    // REWARDS
    // ══════════════════════════════════════════════════════════════════════

    private void GiveScanningRewards()
    {
        bool victimsRemain = envConfig != null && envConfig.RemainingVictims() > 0;
        if (victimsRemain)
        {
            float hSpeed = new Vector3(rb.linearVelocity.x, 0, rb.linearVelocity.z).magnitude;
            if (hSpeed > 1f)            AddReward( 0.001f);
            else if (rb.linearVelocity.magnitude < 0.1f) AddReward(-0.001f);

            if (isVictimVisible && !wasVictimVisibleLastStep)
                AddReward(0.05f);
        }
        wasVictimVisibleLastStep = isVictimVisible;
    }

    void CalculateRewards()
    {
        float reward = 0f;

        // Hover reward (when no victims remain)
        bool hasVictims = envConfig != null && envConfig.RemainingVictims() > 0;
        if (!hasVictims)
        {
            float altErr = Mathf.Abs(transform.localPosition.y - envConfig.spawnHeight);
            reward += altErr < 0.5f ? 0.05f : altErr < 1.0f ? 0.02f : altErr < 2.0f ? 0.005f : -0.02f;
            if (rb.linearVelocity.magnitude < 1f) reward += 0.002f;
        }

        // Altitude penalty
        if (transform.localPosition.y <= 1f) reward -= 0.08f;

        // Stability
        Vector3 euler     = transform.localRotation.eulerAngles;
        float   tiltX     = NormalizeAngle(euler.x);
        float   tiltZ     = NormalizeAngle(euler.z);
        float   combined  = Mathf.Sqrt(tiltX * tiltX + tiltZ * tiltZ);
        if (combined > maxTiltAngle + 15f) reward -= 0.4f;
        if (rb.linearVelocity.y < -2.5f)  reward -= 0.01f;

        // Ground proximity
        if (isNearGround && rb.linearVelocity.y < -1.5f)
        {
            reward -= 0.05f;
            if (rb.linearVelocity.y < -3f) reward -= 0.2f;
        }
        if (isNearGround && rb.linearVelocity.y > -1.0f && rb.linearVelocity.y < 0.5f)
            reward += 0.005f;
        if (transform.localPosition.y < 0.5f) reward -= 0.01f;

        // Spinning in place
        float hSpeed  = new Vector3(rb.linearVelocity.x, 0, rb.linearVelocity.z).magnitude;
        float yawRate = Mathf.Abs(rb.angularVelocity.y);
        if (yawRate > 0.3f && hSpeed < 0.3f && hasVictims)
            reward -= 0.01f;

        // Obstacle awareness
        bool   obstacleInPath  = false;
        float  closestObsDist  = Mathf.Infinity;
        Vector3 agentPos       = transform.position;

        if (cachedObstacles != null)
        {
            Vector3 estimatedVictimDir = Vector3.zero;
            if (isVictimVisible && droneCamera != null)
            {
                float cx = (lockedDirX - 0.5f) * 2f;
                float cy = (lockedDirY - 0.5f) * 2f;
                estimatedVictimDir = (droneCamera.transform.forward
                                    + droneCamera.transform.right * cx * yoloHorizMultiplier
                                    + droneCamera.transform.up    * (-cy) * yoloVertMultiplier
                                    ).normalized;
            }

            foreach (var obs in cachedObstacles)
            {
                if (obs == null || !obs.activeSelf) continue;
                float dist = Vector3.Distance(agentPos, obs.transform.position);
                if (dist < closestObsDist) closestObsDist = dist;

                if (isVictimVisible && estimatedVictimDir != Vector3.zero)
                {
                    float victimDist = lockedEstimatedPos != Vector3.zero
                        ? Vector3.Distance(agentPos, lockedEstimatedPos) : 100f;
                    if (dist < victimDist)
                    {
                        Vector3 obsDir = (obs.transform.position - agentPos).normalized;
                        if (Vector3.Dot(estimatedVictimDir, obsDir) > 0.7f)
                            obstacleInPath = true;
                    }
                }
            }
        }

        // Centering reward (gated on forward movement)
        if (isVictimVisible)
        {
            float centeredness = 1f - Mathf.Abs(lockedDirX - 0.5f) * 2f;
            if (hSpeed > 0.5f && centeredness > 0.7f) reward += 0.002f;
            else if (centeredness < 0.2f)              reward -= 0.002f;
        }

        if (closestObsDist < 4f)
        {
            reward -= (4f - closestObsDist) / 4f * 0.05f;
            float spd = rb.linearVelocity.magnitude;
            if (spd > 3f) reward -= (spd - 3f) / 10f * 0.03f;
        }

        // ── Progress reward (lock-based, bearing+elevation delta) ─────
        if (isVictimVisible && lockedEstimatedPos != Vector3.zero)
        {
            float currentDistance = Vector3.Distance(transform.position, lockedEstimatedPos);
            distanceToTargetHistory.Add(currentDistance);
            if (currentDistance < minDistanceAchieved) minDistanceAchieved = currentDistance;

            bool  validPrev          = previousDistance < 1e6f;
            float progressMultiplier = obstacleInPath ? 0.3f : 1.0f;

            if (validPrev)
            {
                float delta = Mathf.Clamp(previousDistance - currentDistance, -2f, 2f);
                if (delta > 0.001f)
                {
                    int   victimsLeft       = envConfig != null ? envConfig.RemainingVictims() : 1;
                    float urgencyMultiplier = 1f + (victimsLeft * 0.1f);
                    reward += Mathf.Clamp(delta * 0.1f, 0f, 0.3f) * progressMultiplier * urgencyMultiplier;
                    backtrackCounter = 0;
                    if (currentDistance < closestDistanceEver) closestDistanceEver = currentDistance;
                }
                else if (delta < -0.001f && !obstacleInPath)
                {
                    reward -= Mathf.Clamp(-delta * 0.05f, 0f, 0.1f);
                    backtrackCounter++;
                    if (backtrackCounter > 20) reward -= 0.02f;
                }
                else backtrackCounter = Mathf.Max(0, backtrackCounter - 1);
            }
            previousDistance = currentDistance;
        }
        else
        {
            float blindPenalty = Mathf.Lerp(0.005f, 0.02f, Mathf.Clamp01(backtrackCounter / 30f));
            reward -= blindPenalty;
            backtrackCounter = Mathf.Max(0, backtrackCounter - 1);
            if (previousDistance < 1e6f) previousDistance += 0.1f;
        }

        // ── Victim rescue ─────────────────────────────────────────────
        if (envConfig != null && envConfig.IsVictimReached(transform.localPosition, out GameObject rescuedVictim))
        {
            envConfig.RescueVictim(rescuedVictim);
            ResetLock();
            previousDistance    = Mathf.Infinity;
            closestDistanceEver = Mathf.Infinity;
            backtrackCounter    = 0;
            goalsReached++;

            AddReward(reward); episodeReward += reward; reward = 0f;

            int   remainingAfterRescue = envConfig.RemainingVictims();
            float baseReward    = 5.0f;
            float timeBonus     = Mathf.Max(0f, (envConfig.episodeLength - stepCount) / (float)envConfig.episodeLength);
            float diffBonus     = envConfig.obstacleCount > 0 ? 2.0f : 0f;
            float yoloBonus     = (isVictimVisible && yoloChannel.VictimConfidence > 0.6f) ? 1.0f : 0f;
            float rescueReward  = baseReward + timeBonus + diffBonus + yoloBonus;

            AddReward(rescueReward); episodeReward += rescueReward;
            Debug.Log($"VICTIM RESCUED! {goalsReached}/{envConfig.victimCount} | Conf={yoloChannel.VictimConfidence:F2} Steps={stepCount} R={rescueReward:F2}");

            if (remainingAfterRescue == 0)
            {
                successCount++;
                float efficiency       = shortestPath / Mathf.Max(distanceTraveled, 0.001f);
                float completionBonus  = 10f;
                float efficiencyBonus  = efficiency * 5f;
                AddReward(completionBonus + efficiencyBonus);
                episodeReward += completionBonus + efficiencyBonus;

                var rec = Academy.Instance.StatsRecorder;
                rec.Add("Efficiency",      efficiency);
                rec.Add("CompletionTime",  stepCount);

                endReason = "all_victims_rescued";
                EndEpisode();
            }
            return;
        }

        // ── Exploration (only when not locked) ───────────────────────
        if (!isVictimVisible)
        {
            Vector3Int cell = new Vector3Int(
                Mathf.RoundToInt(transform.localPosition.x / GRID_CELL_SIZE),
                Mathf.RoundToInt(transform.localPosition.y / GRID_CELL_SIZE),
                Mathf.RoundToInt(transform.localPosition.z / GRID_CELL_SIZE));
            if (!visitedCells.Contains(cell))
            {
                visitedCells.Add(cell);
                reward += 0.05f / (1f + visitedCells.Count * 0.05f);
            }
        }

        // ── Anti-hover ────────────────────────────────────────────────
        if (hasVictims && stepCount % 10 == 0)
        {
            float movement = Vector3.Distance(transform.localPosition, lastHoverCheck);
            if (movement < 0.2f)
            {
                hoverCounter++;
                if (hoverCounter > 15) reward -= 0.1f;
                if (hoverCounter > 30)
                {
                    reward -= 1f; endReason = "idle_hovering";
                    AddReward(reward); episodeReward += reward;
                    if (!test) EndEpisode();
                    return;
                }
            }
            else hoverCounter = 0;
            lastHoverCheck = transform.localPosition;
        }

        // ── Movement tracking ─────────────────────────────────────────
        float newDist = Vector3.Distance(transform.localPosition, lastPos);
        if (newDist < 0.001f) reward -= 0.01f;
        distanceTraveled += newDist;
        lastPos = transform.localPosition;

        reward -= 0.005f;
        float speed = rb.linearVelocity.magnitude;
        if (speed > 12f) reward -= 0.01f * (speed - 12f);

        AddReward(reward); episodeReward += reward;

        // ── Stats ─────────────────────────────────────────────────────
        float uprightness = Vector3.Dot(transform.up, Vector3.up);
        var stats = Academy.Instance.StatsRecorder;
        stats.Add("AngleStability",   uprightness);
        stats.Add("GroundDistance",   groundDistance);
        stats.Add("VerticalVelocity", rb.linearVelocity.y);
        stats.Add("DroneHP",          droneHP);
        stats.Add("YoloDetected",     yoloChannel.VictimDetected,  Unity.MLAgents.StatAggregationMethod.MostRecent);
        stats.Add("YoloConfidence",   yoloChannel.VictimConfidence,Unity.MLAgents.StatAggregationMethod.MostRecent);
        stats.Add("LockOn",           isLockedOn ? 1f : 0f,        Unity.MLAgents.StatAggregationMethod.MostRecent);
        stats.Add("GoalsReached",     goalsReached,                 Unity.MLAgents.StatAggregationMethod.MostRecent);
        stats.Add("GroundCollisions", groundCollisionCount,         Unity.MLAgents.StatAggregationMethod.MostRecent);
        stats.Add("ExploredCells",    visitedCells.Count,           Unity.MLAgents.StatAggregationMethod.MostRecent);
        stats.Add("BacktrackCount",   backtrackCounter,             Unity.MLAgents.StatAggregationMethod.MostRecent);
        stats.Add("PathEfficiency",   shortestPath / Mathf.Max(distanceTraveled, 0.001f));
        stats.Add("MissionProgress",  (float)goalsReached / Mathf.Max(envConfig != null ? envConfig.victimCount : 1, 1));
    }

    // ══════════════════════════════════════════════════════════════════════
    // COLLISIONS
    // ══════════════════════════════════════════════════════════════════════

    void OnCollisionEnter(Collision other)
    {
        if (other.gameObject.CompareTag("Ground"))
        {
            groundCollisionCount++;
            float impact = other.relativeVelocity.magnitude;

            if (hoverMode) { AddReward(-3f); episodeReward -= 3f; crashCount++; endReason = "ground_contact_hover"; }

            if (impact >= hardLandingThreshold)
            {
                float dmg     = impact * impact * 1.0f;
                float penalty = dmg / MAX_HP * 2f;
                droneHP -= dmg; AddReward(-penalty); episodeReward -= penalty;
                if (droneHP <= 0f) { AddReward(-2f); episodeReward -= 2f; crashCount++; endReason = "hard_crash"; EndEpisode(); return; }
            }
            else
            {
                float dmg = impact * 2f + 2f;
                droneHP -= dmg; AddReward(-0.5f); episodeReward -= 0.5f;
                if (droneHP <= 0f) { AddReward(-1f); episodeReward -= 1f; crashCount++; endReason = "rough_landing"; EndEpisode(); return; }
            }
        }

        if (other.gameObject.CompareTag("Obstacle"))
        {
            float impact  = other.relativeVelocity.magnitude;
            float dmg     = (impact * impact * 1.5f) + 10f;
            float penalty = (dmg / MAX_HP * 3f) + 3f;
            droneHP -= dmg; AddReward(-penalty); episodeReward -= penalty;
            if (droneHP <= 0f) { AddReward(-10f); episodeReward -= 10f; crashCount++; endReason = "destroyed_obstacle"; EndEpisode(); }
        }
    }

    void OnCollisionStay(Collision other)
    {
        if (Time.time - lastStayDamageTime < 0.5f) return;
        lastStayDamageTime = Time.time;

        if (other.gameObject.CompareTag("Obstacle"))
        {
            droneHP -= 5f; AddReward(-0.05f); episodeReward -= 0.05f;
            if (droneHP <= 0f) { AddReward(-10f); episodeReward -= 10f; crashCount++; endReason = "grinding"; EndEpisode(); }
        }
        if (other.gameObject.CompareTag("Ground") && !hoverMode)
        { AddReward(-0.01f); episodeReward -= 0.01f; }
    }

    // ══════════════════════════════════════════════════════════════════════
    // TERMINATION
    // ══════════════════════════════════════════════════════════════════════

    void CheckTerminationConditions()
    {
        if (test || envConfig == null) return;

        if (envConfig.IsOutOfBounds(transform.localPosition))
        { AddReward(-10f); episodeReward -= 10f; outOfBoundsCount++; endReason = "out_of_bounds"; EndEpisode(); return; }

        if (transform.localPosition.y < 0.2f)
        { AddReward(-5f); episodeReward -= 5f; crashCount++; endReason = "below_ground"; EndEpisode(); return; }

        if (stepCount >= envConfig.episodeLength)
        {
            int remaining = envConfig.RemainingVictims();
            float penalty = -2f - (remaining * 1f);
            AddReward(penalty); episodeReward += penalty;
            timeoutCount++; endReason = "timeout";
            EndEpisode();
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // HEURISTIC (manual control)
    // ══════════════════════════════════════════════════════════════════════

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ca = actionsOut.ContinuousActions;
        ca[0] = Input.GetKey(KeyCode.W) ? 1f : Input.GetKey(KeyCode.S) ? -1f : 0f;
        ca[1] = Input.GetAxis("Horizontal");
        ca[2] = Input.GetKey(KeyCode.UpArrow) ? 1f : Input.GetKey(KeyCode.DownArrow) ? -1f : 0f;
        ca[3] = Input.GetKey(KeyCode.Q) ? -1f : Input.GetKey(KeyCode.E) ? 1f : 0f;

        float spd = Mathf.Abs(ca[0]) > 0 ? 2f : 1f;
        foreach (var anim in fanAnimators) anim.SetFloat("fanSpeed", spd);
    }

    // ══════════════════════════════════════════════════════════════════════
    // DEBUG VISUALIZATION HELPERS
    // ══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Draws a wireframe sphere using Debug.DrawLine — visible in both
    /// Scene view and Game view (when Gizmos is enabled).
    /// </summary>
    private void DrawWireSphere(Vector3 center, float radius, Color color, int segments = 16)
    {
        float step = 360f / segments;

        // XZ plane (horizontal ring)
        for (int i = 0; i < segments; i++)
        {
            float a1 = i * step * Mathf.Deg2Rad;
            float a2 = (i + 1) * step * Mathf.Deg2Rad;
            Debug.DrawLine(
                center + new Vector3(Mathf.Cos(a1) * radius, 0, Mathf.Sin(a1) * radius),
                center + new Vector3(Mathf.Cos(a2) * radius, 0, Mathf.Sin(a2) * radius),
                color);
        }
        // XY plane (vertical ring — frontal)
        for (int i = 0; i < segments; i++)
        {
            float a1 = i * step * Mathf.Deg2Rad;
            float a2 = (i + 1) * step * Mathf.Deg2Rad;
            Debug.DrawLine(
                center + new Vector3(Mathf.Cos(a1) * radius, Mathf.Sin(a1) * radius, 0),
                center + new Vector3(Mathf.Cos(a2) * radius, Mathf.Sin(a2) * radius, 0),
                color);
        }
        // YZ plane (vertical ring — side)
        for (int i = 0; i < segments; i++)
        {
            float a1 = i * step * Mathf.Deg2Rad;
            float a2 = (i + 1) * step * Mathf.Deg2Rad;
            Debug.DrawLine(
                center + new Vector3(0, Mathf.Sin(a1) * radius, Mathf.Cos(a1) * radius),
                center + new Vector3(0, Mathf.Sin(a2) * radius, Mathf.Cos(a2) * radius),
                color);
        }
    }

    /// <summary>
    /// Draws a small world-space cross (+) at a point — marks individual ray hit locations.
    /// </summary>
    private void DrawCross(Vector3 center, float size, Color color)
    {
        Debug.DrawLine(center - Vector3.right   * size, center + Vector3.right   * size, color);
        Debug.DrawLine(center - Vector3.up      * size, center + Vector3.up      * size, color);
        Debug.DrawLine(center - Vector3.forward * size, center + Vector3.forward * size, color);
    }

    // ══════════════════════════════════════════════════════════════════════
    // CALIBRATION HELPERS  (editor-only, never used in training)
    // ══════════════════════════════════════════════════════════════════════

    private Vector3 FindClosestVictimPosition()
    {
        if (envConfig == null) return Vector3.zero;
        var victims = envConfig.GetActiveVictims();
        if (victims == null || victims.Length == 0) return Vector3.zero;

        Vector3 closest = Vector3.zero;
        float   minDist = Mathf.Infinity;
        foreach (var v in victims)
        {
            if (v == null || !v.activeSelf) continue;
            float d = Vector3.Distance(transform.position, v.transform.position);
            if (d < minDist) { minDist = d; closest = v.transform.position; }
        }
        return minDist < Mathf.Infinity ? closest : Vector3.zero;
    }

    // ══════════════════════════════════════════════════════════════════════
    // MISC
    // ══════════════════════════════════════════════════════════════════════

    public void ResetDrone()
    {
        throttle = roll = pitch = yaw = 0;
        rb.linearVelocity  = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        rb.Sleep();
        rb.position = startPosition;
        rb.rotation = startRotation;
        rb.WakeUp();
    }

    float NormalizeAngle(float angle)
    {
        while (angle > 180f)  angle -= 360f;
        while (angle < -180f) angle += 360f;
        return angle;
    }

    float CalculateShortestPath()
    {
        if (envConfig == null) return 0f;
        var victims = envConfig.GetActiveVictims();
        if (victims == null || victims.Length == 0) return 0f;

        float total   = 0f;
        Vector3 cur   = transform.localPosition;
        var remaining = new List<GameObject>(victims);

        while (remaining.Count > 0)
        {
            float nearest = Mathf.Infinity;
            GameObject nearestV = null;
            foreach (var v in remaining)
            {
                if (v == null) continue;
                float d = Vector3.Distance(cur, v.transform.localPosition);
                if (d < nearest) { nearest = d; nearestV = v; }
            }
            if (nearestV == null) break;
            total += nearest;
            cur = nearestV.transform.localPosition;
            remaining.Remove(nearestV);
        }
        return total;
    }

    private void LogTestRun()
    {
        if (!recordEvaluationMetrics || episodeCount == 0 || string.IsNullOrEmpty(logFilePath)) return;
        int   totalVictims = envConfig != null ? envConfig.victimCount : 1;
        float efficiency   = shortestPath > 0 ? shortestPath / Mathf.Max(distanceTraveled, 0.001f) : 0f;
        try { File.AppendAllText(logFilePath,
            $"{episodeCount},{goalsReached},{totalVictims},{stepCount},{efficiency:F3},{distanceTraveled:F2},{endReason},{droneHP:F1}\n"); }
        catch (Exception e) { Debug.LogError("CSV Write Error: " + e.Message); }
    }

    // ══════════════════════════════════════════════════════════════════════
    // ON GUI
    // ══════════════════════════════════════════════════════════════════════

    void OnGUI()
    {
        var style        = new GUIStyle { fontSize = 14 };
        style.normal.textColor = Color.white;

        int victimsLeft = envConfig != null ? envConfig.RemainingVictims() : 0;
        float bearing   = 0f, elevation = 0f;
        if (isLockedOn && lockedEstimatedPos != Vector3.zero)
        {
            Vector3 lo = transform.InverseTransformPoint(lockedEstimatedPos);
            bearing    = Mathf.Atan2(lo.x, lo.z) * Mathf.Rad2Deg;
            elevation  = Mathf.Atan2(lo.y, new Vector2(lo.x, lo.z).magnitude) * Mathf.Rad2Deg;
        }

        float y = 30f;
        GUI.Label(new Rect(40, y,      350, 24), $"Reward: {GetCumulativeReward():F2}",              style); y += 20;
        GUI.Label(new Rect(40, y,      400, 24), $"Victims Left: {victimsLeft} | Goals: {goalsReached}", style); y += 20;
        GUI.Label(new Rect(40, y,      400, 24), $"Explored: {visitedCells.Count} | HP: {droneHP:F1}",  style); y += 20;
        GUI.Label(new Rect(40, y,      400, 24), $"EndReason: {endReason}",                          style); y += 20;
        GUI.Label(new Rect(40, y,      400, 24), $"LOCK: {(isLockedOn ? "ON" : $"OFF (lost {lockLostCounter}/{LOCK_LOST_THRESHOLD})")}", style); y += 20;
        GUI.Label(new Rect(40, y,      400, 24), $"YOLO Conf={yoloChannel.VictimConfidence:F2}  Det={yoloChannel.VictimDetected:F2}", style); y += 20;
        GUI.Label(new Rect(40, y,      400, 24), $"DirX={lockedDirX:F2}  DirY={lockedDirY:F2}  NormDist={lockedDistance:F3}", style); y += 20;
        GUI.Label(new Rect(40, y,      400, 24), $"Bearing={bearing:F1}°  Elev={elevation:F1}°",     style); y += 20;
        GUI.Label(new Rect(40, y,      400, 24), $"GroundDist={groundDistance:F1}m  NearGround={isNearGround}", style); y += 20;
        GUI.Label(new Rect(40, y,      400, 24), $"EstPos={lockedEstimatedPos:F1}",                  style); y += 20;

        // Debug toggle hint
        GUI.Label(new Rect(40, y,      400, 24),
            $"[Rays: {(showDebugRays ? "ON" : "OFF")}] — toggle 'showDebugRays' in Inspector", style);
    }

    void Update()
    {
        if (envConfig != null && envConfig.isInitializing) return;
        if (Input.GetKeyDown(KeyCode.H)) { hoverMode = !hoverMode; Debug.Log("Hover: " + hoverMode); }
        if (Input.GetKeyDown(KeyCode.R)) EndEpisode();
        // Quick runtime toggle for rays
        if (Input.GetKeyDown(KeyCode.F1)) showDebugRays = !showDebugRays;
    }
}