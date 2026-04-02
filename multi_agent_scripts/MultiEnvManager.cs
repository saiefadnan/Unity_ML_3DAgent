using UnityEngine;
using Unity.MLAgents;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

public class MultiEnvManager : MonoBehaviour
{
    [Header("Prefabs")]
    public GameObject victimPrefab;

    [Header("Agents")]
    public GameObject[] agentObjects; // Assign all drone GameObjects in Inspector
    private MultiAgent2D[] agents;
    private Rigidbody2D[] droneRbs;

    [Header("Environment")]
    public Transform[] obstacles;
    public int victimCount = 5;
    public int obstacleCount = 0;
    public float hoverHeight = 5.0f;
    public bool hoverMode = false;

    [Header("State")]
    public List<GameObject> activeVictims = new List<GameObject>();
    public bool isInitializing = true;

    // Shared exploration grid — rewards agents for covering NEW ground as a team
    public HashSet<Vector2Int> teamExploredCells = new HashSet<Vector2Int>();
    public const float GRID_CELL_SIZE = 2.5f;

    // Victim claiming — prevents two drones chasing the same victim
    private Dictionary<int, int> victimClaims = new Dictionary<int, int>(); // victimIndex -> agentIndex

    // Track which agent triggered the reset (only first one resets the env)
    private int resetCount = 0;
    private int expectedResets = 0;

    float targetDistance = 0.0f;

    void Awake()
    {
        agents = new MultiAgent2D[agentObjects.Length];
        droneRbs = new Rigidbody2D[agentObjects.Length];

        for (int i = 0; i < agentObjects.Length; i++)
        {
            agents[i] = agentObjects[i].GetComponent<MultiAgent2D>();
            droneRbs[i] = agentObjects[i].GetComponent<Rigidbody2D>();
            agents[i].agentIndex = i;
        }
    }

    // ========== EPISODE MANAGEMENT ==========
    // In multi-agent, we need coordinated resets.
    // First agent to call ResetEnvironment actually resets; others just wait.

    public void RequestReset()
    {
        resetCount++;

        // Only actually reset on the FIRST request per episode
        if (resetCount == 1)
        {
            StartCoroutine(CoordinatedReset());
        }
    }

    IEnumerator CoordinatedReset()
    {
        isInitializing = true;

        InitCurriculumEnv();

        // Deactivate all agents
        for (int i = 0; i < agentObjects.Length; i++)
        {
            agentObjects[i].transform.position = new Vector3(0, -100 - i * 10, 0);
            droneRbs[i].Sleep();
        }

        // Spawn victims and obstacles
        if (!hoverMode) SpawnVictims(victimCount);
        SpawnObstacles(obstacleCount);

        // Clear shared state
        teamExploredCells.Clear();
        victimClaims.Clear();

        // Wait for physics to settle
        yield return StartCoroutine(WaitForVictimsToSettle());

        // Spawn all agents at spread-out positions
        SpawnAllAgents();

        resetCount = 0;
    }

    void InitCurriculumEnv()
    {
        var envParams = Academy.Instance.EnvironmentParameters;
        hoverHeight = envParams.GetWithDefault("hover_height", 5.0f);
        targetDistance = envParams.GetWithDefault("target_distance", 20f);
        obstacleCount = (int)envParams.GetWithDefault("obstacle_count", 6f);
        hoverMode = targetDistance == 0f && obstacleCount == 0f;
    }

    // ========== VICTIM CLAIMING ==========
    // Prevents two drones from chasing the same victim

    public int ClaimNearestAvailableVictim(int agentIndex, Vector2 agentPos)
    {
        float bestDist = Mathf.Infinity;
        int bestIndex = -1;

        for (int i = 0; i < activeVictims.Count; i++)
        {
            if (activeVictims[i] == null || !activeVictims[i].activeSelf) continue;

            // Skip if claimed by another agent
            if (victimClaims.ContainsKey(i) && victimClaims[i] != agentIndex) continue;

            float dist = Vector2.Distance(agentPos, activeVictims[i].transform.position);
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

            // Claim new target
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

    // ========== TEAM EXPLORATION ==========

    public bool RegisterExploredCell(Vector2Int cell)
    {
        if (!teamExploredCells.Contains(cell))
        {
            teamExploredCells.Add(cell);
            return true; // New cell — reward the discoverer
        }
        return false; // Already explored by a teammate
    }

    // ========== TEAMMATE INFO ==========
    // Each agent can observe its teammates

    public struct TeammateInfo
    {
        public Vector2 position;
        public Vector2 velocity;
        public float hp;
        public bool isActive;
    }

    public TeammateInfo GetTeammateInfo(int requestingAgent, int teammateSlot)
    {
        // teammateSlot is 0-based index into "other agents"
        int actualIndex = 0;
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

        if (actualIndex >= agents.Length || agents[actualIndex] == null)
        {
            return new TeammateInfo { isActive = false };
        }

        return new TeammateInfo
        {
            position = agentObjects[actualIndex].transform.localPosition,
            velocity = droneRbs[actualIndex].linearVelocity,
            hp = agents[actualIndex].droneHP,
            isActive = true
        };
    }

    public int AgentCount => agentObjects.Length;

    // ========== TEAM COMPLETION CHECK ==========

    public int TotalVictimsFound()
    {
        int total = 0;
        foreach (var agent in agents)
        {
            if (agent != null) total += agent.goalsReached;
        }
        return total;
    }

    public bool AllVictimsFound()
    {
        int remaining = 0;
        foreach (var v in activeVictims)
        {
            if (v != null && v.activeSelf) remaining++;
        }
        return remaining == 0;
    }

    public bool AllAgentsDead()
    {
        foreach (var agent in agents)
        {
            if (agent != null && agent.droneHP > 0) return false;
        }
        return true;
    }

    // ========== SPAWNING ==========

    void SpawnAllAgents()
    {
        float spacing = 24f / (agentObjects.Length + 1); // Spread across arena

        for (int i = 0; i < agentObjects.Length; i++)
        {
            droneRbs[i].WakeUp();

            // Spread agents horizontally
            float xPos = -12f + spacing * (i + 1) + Random.Range(-1f, 1f);
            xPos = Mathf.Clamp(xPos, -12f, 12f);

            agentObjects[i].transform.localPosition = new Vector2(xPos, 1.5f);
            droneRbs[i].SetRotation(0f);
            droneRbs[i].linearVelocity = Vector2.zero;
            droneRbs[i].angularVelocity = 0f;
        }

        isInitializing = false;

        // Notify all agents
        for (int i = 0; i < agents.Length; i++)
        {
            agents[i].PostEnvironmentReset();
        }
    }

    Vector2 GetRandomTargetPosition()
    {
        Vector2 arenaCenter = Vector2.zero;
        float angle = Random.Range(0f, Mathf.PI * 2f);
        float minSpread = 3f;
        float maxSpread = Mathf.Min(targetDistance * 0.8f, 16f);
        float distance = Random.Range(minSpread, maxSpread);

        float offsetX = Mathf.Cos(angle) * distance;
        float offsetY = Mathf.Sin(angle) * distance * 0.3f;

        float targetX = Mathf.Clamp(arenaCenter.x + offsetX, -14f, 14f);
        float targetY;

        if (obstacleCount > 0 && Random.value > 0.5f)
        {
            targetY = Mathf.Clamp(arenaCenter.y + offsetY, 2f, 6f);
        }
        else
        {
            targetY = Mathf.Clamp(arenaCenter.y + offsetY, -1f, 4f);
        }

        return new Vector2(targetX, targetY);
    }

    void SpawnVictims(int count)
    {
        foreach (GameObject victim in activeVictims)
        {
            if (victim != null) Destroy(victim);
        }
        activeVictims.Clear();

        for (int i = 0; i < count; i++)
        {
            Vector2 pos = GetRandomTargetPosition();
            GameObject victim = Instantiate(victimPrefab, pos, Quaternion.identity);
            Rigidbody2D rb = victim.GetComponent<Rigidbody2D>();
            if (rb != null)
            {
                rb.bodyType = obstacleCount == 0 ? RigidbodyType2D.Kinematic : RigidbodyType2D.Dynamic;
            }
            activeVictims.Add(victim);
        }
    }

    void SpawnObstacles(int count)
    {
        int maxObs = Mathf.Min(count, obstacles.Length);
        for (int i = 0; i < maxObs; i++)
        {
            int randomIndex = Random.Range(i, obstacles.Length);
            Vector2 pos1 = obstacles[i].localPosition;
            Vector2 pos2 = obstacles[randomIndex].localPosition;
            obstacles[i].localPosition = new Vector2(pos2.x, pos1.y);
            obstacles[randomIndex].localPosition = new Vector2(pos1.x, pos2.y);
            obstacles[i].gameObject.SetActive(true);
        }

        for (int i = maxObs; i < obstacles.Length; i++)
        {
            obstacles[i].gameObject.SetActive(false);
        }
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
                Rigidbody2D victimRb = victim.GetComponent<Rigidbody2D>();
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
}
