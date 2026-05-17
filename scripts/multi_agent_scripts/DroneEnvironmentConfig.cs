using UnityEngine;
using Unity.MLAgents;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

public class DroneEnvironmentConfig : MonoBehaviour
{
    [Header("Curriculum Parameters (Auto-Updated)")]
    public float targetDistance = 0f;
    public int obstacleCount = 0;
    public float episodeLength = 5000f;
    public float spawnHeight = 10f;
    public float successRadius = 1f;

    [Header("References")]
    public GameObject victimPrefab;
    public GameObject obstaclePrefab;

    [Header("Agents")]
    public GameObject[] agentObjects; // Assign in Inspector
    private DroneController[] agents;
    private Rigidbody[] droneRbs;

    [Header("Victim Settings")]
    public int victimCount = 10;

    [Header("Environment Bounds")]
    public Vector3 arenaSize = new Vector3(50f, 20f, 50f);
    public Vector3 arenaCenter = Vector3.zero;

    public GameObject[] activeObstacles;
    public GameObject[] activeVictims;
    private EnvironmentParameters envParams;

    public bool isInitializing = true;

    // Shared exploration grid
    public HashSet<Vector3Int> teamExploredCells = new HashSet<Vector3Int>();

    // Victim claiming
    private Dictionary<int, int> victimClaims = new Dictionary<int, int>(); // victimIndex -> agentIndex
    private int resetCount = 0;

    public struct AgentInfo
    {
        public int goals;
        public float hp;
    }

    public AgentInfo GetAgentInfo(int index)
    {
        if (agents == null || index < 0 || index >= agents.Length || agents[index] == null)
            return new AgentInfo { goals = 0, hp = 0f };
        return new AgentInfo
        {
            goals = agents[index].goalsReached,
            hp = agents[index].droneHP
        };
    }

    void Awake()
    {
        envParams = Academy.Instance.EnvironmentParameters;

        if (agentObjects != null && agentObjects.Length > 0)
        {
            agents = new DroneController[agentObjects.Length];
            droneRbs = new Rigidbody[agentObjects.Length];

            for (int i = 0; i < agentObjects.Length; i++)
            {
                agents[i] = agentObjects[i].GetComponent<DroneController>();
                droneRbs[i] = agentObjects[i].GetComponent<Rigidbody>();
                if (agents[i] != null) agents[i].agentIndex = i;
            }
        }
    }

    public void RequestReset()
    {
        resetCount++;
        if (resetCount == 1)
        {
            StartCoroutine(CoordinatedReset());
        }
    }

    IEnumerator CoordinatedReset()
    {
        isInitializing = true;

        targetDistance = envParams.GetWithDefault("target_distance", 2f);
        obstacleCount = Mathf.RoundToInt(envParams.GetWithDefault("obstacle_count", 8f));

        DeactivateAllAgents();
        SpawnVictims();
        SpawnObstacles();

        teamExploredCells.Clear();
        victimClaims.Clear();

        if (targetDistance > 0f)
        {
            yield return StartCoroutine(WaitForVictimsToSettle());
        }
        else
        {
            yield return new WaitForFixedUpdate();
        }

        SpawnAllAgents();
        resetCount = 0;
    }

    void DeactivateAllAgents()
    {
        if (agentObjects == null) return;
        for (int i = 0; i < agentObjects.Length; i++)
        {
            agentObjects[i].transform.position = new Vector3(0, -100 - i * 10, 0);
            if (droneRbs[i] != null) droneRbs[i].Sleep();
        }
    }

    void SpawnAllAgents()
    {
        if (agentObjects == null || agentObjects.Length == 0) return;

        float spacing = 24f / (agentObjects.Length + 1);

        for (int i = 0; i < agentObjects.Length; i++)
        {
            if (droneRbs[i] != null) droneRbs[i].WakeUp();

            float xPos = arenaCenter.x - 12f + spacing * (i + 1) + Random.Range(-1f, 1f);
            xPos = Mathf.Clamp(xPos, arenaCenter.x - 12f, arenaCenter.x + 12f);

            agentObjects[i].transform.position = new Vector3(
                xPos,
                spawnHeight,
                arenaCenter.z + Random.Range(-2f, 2f)
            );

            agentObjects[i].transform.rotation = Quaternion.identity;

            if (droneRbs[i] != null)
            {
                droneRbs[i].linearVelocity = Vector3.zero;
                droneRbs[i].angularVelocity = Vector3.zero;
            }
        }

        isInitializing = false;

        for (int i = 0; i < agents.Length; i++)
        {
            if (agents[i] != null) agents[i].PostEnvironmentReset();
        }
    }

    // ========== VICTIM CLAIMING ==========

    /// <summary>
    /// Claim the nearest unclaimed (or already claimed by this agent) active victim.
    /// Uses world-space positions throughout.
    /// </summary>
    public int ClaimNearestAvailableVictim(int agentIndex, Vector3 agentWorldPos)
    {
        float bestDist = Mathf.Infinity;
        int bestIndex = -1;

        if (activeVictims == null) return -1;

        for (int i = 0; i < activeVictims.Length; i++)
        {
            if (activeVictims[i] == null || !activeVictims[i].activeSelf) continue;

            // Skip victims claimed by a different agent
            if (victimClaims.ContainsKey(i) && victimClaims[i] != agentIndex) continue;

            // FIX: use world position (transform.position, not localPosition)
            float dist = Vector3.Distance(agentWorldPos, activeVictims[i].transform.position);
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIndex = i;
            }
        }

        if (bestIndex >= 0)
        {
            // Release any previous claim by this agent
            var oldClaims = victimClaims.Where(kv => kv.Value == agentIndex).Select(kv => kv.Key).ToList();
            foreach (var key in oldClaims) victimClaims.Remove(key);

            victimClaims[bestIndex] = agentIndex;
        }

        return bestIndex;
    }

    public void ReleaseClaim(int agentIndex)
    {
        var claims = victimClaims.Where(kv => kv.Value == agentIndex).Select(kv => kv.Key).ToList();
        foreach (var key in claims) victimClaims.Remove(key);
    }

    public bool IsVictimClaimed(int victimIndex, int byAgentIndex)
    {
        return victimClaims.ContainsKey(victimIndex) && victimClaims[victimIndex] == byAgentIndex;
    }

    public GameObject GetVictim(int index)
    {
        if (activeVictims != null && index >= 0 && index < activeVictims.Length)
            return activeVictims[index];
        return null;
    }

    /// <summary>
    /// Centralized victim rescue. Deactivates the victim and returns true if the
    /// agent is close enough to ANY active victim (world-space distance <= successRadius).
    /// </summary>
    public bool TryRescueAnyVictim(Vector3 agentWorldPos, out int rescuedIndex)
    {
        rescuedIndex = -1;
        if (activeVictims == null) return false;

        for (int i = 0; i < activeVictims.Length; i++)
        {
            GameObject victim = activeVictims[i];
            if (victim == null || !victim.activeSelf) continue;

            float dist = Vector3.Distance(agentWorldPos, victim.transform.position);

            if (dist <= successRadius)
            {
                victim.SetActive(false);
                victimClaims.Remove(i);
                rescuedIndex = i;
                return true;
            }
        }

        return false;
    }

    // ========== TEAM EXPLORATION ==========
    public bool RegisterExploredCell(Vector3Int cell)
    {
        if (!teamExploredCells.Contains(cell))
        {
            teamExploredCells.Add(cell);
            return true;
        }
        return false;
    }

    // ========== TEAMMATE INFO ==========
    public struct TeammateInfo
    {
        public Vector3 position;   // world position
        public Vector3 velocity;
        public float hp;
        public bool isActive;
    }

    public TeammateInfo GetTeammateInfo(int requestingAgent, int teammateSlot)
    {
        int actualIndex = -1;
        int slotCount = 0;

        for (int i = 0; i < agents.Length; i++)
        {
            if (i == requestingAgent) continue;

            if (slotCount == teammateSlot)
            {
                actualIndex = i;
                break;
            }
            slotCount++;
        }

        if (actualIndex < 0 || agents == null || actualIndex >= agents.Length || agents[actualIndex] == null)
        {
            return new TeammateInfo { isActive = false };
        }

        return new TeammateInfo
        {
            // FIX: was transform.localPosition — must be world position so
            // DroneController can do: info.position - transform.position (world - world)
            position = agentObjects[actualIndex].transform.position,
            velocity = droneRbs[actualIndex].linearVelocity,
            hp = agents[actualIndex].droneHP,
            isActive = true
        };
    }

    public int AgentCount => agentObjects != null ? agentObjects.Length : 0;

    // ========== TEAM METRICS ==========
    public int TotalVictimsFound()
    {
        int total = 0;
        if (agents == null) return 0;
        foreach (var agent in agents)
            if (agent != null) total += agent.goalsReached;
        return total;
    }

    public bool AllVictimsFound()
    {
        return RemainingVictims() == 0 && activeVictims != null && activeVictims.Length > 0;
    }

    public int RemainingVictims()
    {
        int count = 0;
        if (activeVictims == null) return 0;
        foreach (var v in activeVictims)
            if (v != null && v.activeSelf) count++;
        return count;
    }

    // ========== SPAWNING & BOUNDS ==========
    void SpawnVictims()
    {
        if (activeVictims != null)
        {
            foreach (GameObject v in activeVictims)
                if (v != null) Destroy(v);
        }

        if (victimPrefab == null || targetDistance == 0f)
        {
            activeVictims = new GameObject[0];
            return;
        }

        activeVictims = new GameObject[victimCount];
        for (int i = 0; i < victimCount; i++)
        {
            Vector3 spawnPos = GetValidVictimPosition();
            activeVictims[i] = Instantiate(victimPrefab, spawnPos, Quaternion.identity);
            // FIX: Do NOT parent victims to this transform — parenting makes
            // localPosition diverge from world position, which was the original bug.
            // If you need them grouped in the hierarchy, parent them but always
            // use transform.position (world) for all distance calculations.
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
            Vector3 randomDir = Random.onUnitSphere;
            randomDir.y = Mathf.Abs(randomDir.y) + 0.1f;
            randomDir.Normalize();

            float dist = Random.Range(targetDistance * 0.4f, targetDistance);
            Vector3 candidate = arenaCenter + randomDir * dist;

            candidate.x = Mathf.Clamp(candidate.x, arenaCenter.x - arenaSize.x / 2f + 1f, arenaCenter.x + arenaSize.x / 2f - 1f);
            candidate.y = Mathf.Clamp(candidate.y, 1.5f, arenaSize.y - 1f);
            candidate.z = Mathf.Clamp(candidate.z, arenaCenter.z - arenaSize.z / 2f + 1f, arenaCenter.z + arenaSize.z / 2f - 1f);

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

        return new Vector3(
            Random.Range(arenaCenter.x - arenaSize.x / 2f + 2f, arenaCenter.x + arenaSize.x / 2f - 2f),
            Random.Range(2f, arenaSize.y - 2f),
            Random.Range(arenaCenter.z - arenaSize.z / 2f + 2f, arenaCenter.z + arenaSize.z / 2f - 2f)
        );
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
                Debug.LogWarning($"[SpawnObstacles] Fallback placing {i}");

            Quaternion randomYRotation = Quaternion.Euler(0, Random.Range(0f, 360f), 0);
            activeObstacles[i] = Instantiate(obstaclePrefab, obstaclePos, randomYRotation);
            activeObstacles[i].transform.SetParent(transform);
            activeObstacles[i].tag = "Obstacle";
        }
    }

    bool IsTooCloseToKeyObjects(Vector3 pos, float minDist)
    {
        if (agentObjects != null)
        {
            foreach (var droneObj in agentObjects)
            {
                if (droneObj != null && Vector3.Distance(pos, droneObj.transform.position) < minDist)
                    return true;
            }
        }

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

    IEnumerator WaitForVictimsToSettle()
    {
        float timeout = 5f;
        float elapsed = 0f;

        while (elapsed < timeout)
        {
            bool allSettled = true;

            foreach (GameObject victim in activeVictims)
            {
                if (victim == null) continue;
                Rigidbody victimRb = victim.GetComponent<Rigidbody>();
                if (victimRb != null &&
                    (!victimRb.IsSleeping() || victimRb.linearVelocity.magnitude > 0.05f))
                {
                    allSettled = false;
                    break;
                }
            }

            if (allSettled) break;

            yield return new WaitForFixedUpdate();
            elapsed += Time.fixedDeltaTime;
        }

        yield return new WaitForFixedUpdate();
    }

    public bool IsOutOfBounds(Vector3 worldPosition)
    {
        return Mathf.Abs(worldPosition.x - arenaCenter.x) > arenaSize.x / 2 ||
               worldPosition.y < 0 || worldPosition.y > arenaSize.y ||
               Mathf.Abs(worldPosition.z - arenaCenter.z) > arenaSize.z / 2;
    }

    void OnDrawGizmos()
    {
        Gizmos.color = Color.cyan;
        Gizmos.DrawWireCube(arenaCenter, arenaSize);

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