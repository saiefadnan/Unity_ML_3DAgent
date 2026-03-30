# Thesis Roadmap: Multi-Agent Drone Search and Rescue Using Deep Reinforcement Learning

## 2D to 3D Progression — Full Roadmap from Simulation to Submission

---

## OVERVIEW

This thesis compares single-agent and multi-agent PPO performance across 2D and 3D search and rescue environments. The core narrative is a progressive complexity study: start simple (2D, single), scale up (2D, multi), then prove generalization (3D, single), then prove scalability (3D, multi).

### Experiment Matrix

| Experiment | Environment | Agents | Status |
|---|---|---|---|
| 1 | 2D | Single (PPO) | DONE |
| 2 | 2D | Multi (PPO) | DONE |
| 3 | 3D | Single (PPO) | IN PROGRESS |
| 4 | 3D | Multi (PPO) | NEXT |

### Research Questions
1. Does a PPO agent trained in a 2D environment generalize to 3D?
2. Does multi-agent coordination improve SAR performance in both environments?
3. How does environmental complexity (2D vs 3D) affect learning efficiency and convergence speed?
4. Does multi-agent coordination scale better in 3D than in 2D?

---

## PHASE 1: 2D Environments (COMPLETED)

### 1.1 2D Single Agent PPO
- Environment: 2D flat arena with victims and obstacles
- Agent: single drone/agent with vector observations
- Algorithm: PPO
- Status: DONE — collect final evaluation metrics if not already done
- Key data needed:
  - VictimsRescued per episode (mean over last 100 episodes)
  - MissionSuccessRate (episodes where all victims rescued)
  - StepsTaken to completion
  - PathEfficiency
  - Training steps to convergence

### 1.2 2D Multi Agent PPO
- Environment: same 2D arena, 2 agents sharing policy
- Algorithm: PPO shared policy
- Status: DONE — collect final evaluation metrics if not already done
- Key data needed: same metrics as 1.1
- Additional: did agents learn to split up or follow each other?

### 1.3 2D Evaluation Checklist
- [ ] Run 200 evaluation episodes for single agent — record CSV
- [ ] Run 200 evaluation episodes for multi agent — record CSV
- [ ] Screenshot TensorBoard learning curves for both
- [ ] Note curriculum stages reached and at what step
- [ ] Screenshot or record video of agent behavior in final policy

---

## PHASE 2: 3D Single Agent PPO (IN PROGRESS)

### 2.1 Environment Design
- 3D arena: 50x50x20 units
- Drone with Angle Mode physics controller (thrust, tilt compensation)
- Victim prefabs spawned at curriculum-defined distances
- Obstacle prefabs with tag-based collision detection
- DroneEnvironmentConfig managing spawn, curriculum, and victim rescue logic
- Camera sensor: 84x84 RGB, nature_cnn encoder, LSTM memory

### 2.2 Agent Observations (20 total)
- Position normalized by arena size (3)
- Local velocity normalized by maxVelocity (3)
- Angular velocity normalized by maxYawSpeed in radians (3)
- Local rotation as quaternion (4)
- Victim local direction — full precision GPS, always on (3)
- Victim normalized distance — exact Clamp01(dist/50f) (1)
- Mission progress — rescued / total (1)
- Remaining victims ratio (1)
- Ground distance normalized (1)

### 2.3 Reward System
- Hover reward: altitude error + low velocity bonus (hover training stage only)
- Progress reward: delta distance to nearest victim, obstacle-aware multiplier, urgency multiplier
- Movement reward: +0.001f for horizontal movement, -0.001f for being stationary
- Rescue reward: base 5.0 + time bonus + difficulty bonus
- Completion bonus: 10.0 + efficiency bonus when all victims rescued
- Penalties: spin-in-place, ground proximity, obstacle collision, idle hovering, out of bounds, timeout scaled by remaining victims

### 2.4 Curriculum (9 stages)
| Stage | Distance | Obstacles | Threshold |
|---|---|---|---|
| hover_training | 0 | 0 | 600.0 |
| close_target | 5 | 0 | 15.0 |
| medium_target | 15 | 0 | 25.0 |
| static_obstacles | 15 | 3 | 30.0 |
| complex_obstacles | 20 | 6 | 30.0 |
| medium_obstacles | 25 | 6 | 35.0 |
| mid_distance | 35 | 8 | 40.0 |
| long_distance | 50 | 10 | 45.0 |
| expert_flight | 60 | 12 | final |

### 2.5 PPO Configuration
```yaml
trainer_type: ppo
batch_size: 512
buffer_size: 20480
learning_rate: 0.0003
beta: 0.01
epsilon: 0.2
lambd: 0.95
num_epoch: 3
vis_encode_type: nature_cnn
memory_size: 256
sequence_length: 128
curiosity strength: 0.005
min_lesson_length: 100 (hover), 200 (navigation)
```

### 2.6 Training Steps
1. Run: `mlagents-learn config3D/single_occ.yaml --run-id=drone3d_ppo_v1 --train`
2. Monitor TensorBoard: `tensorboard --logdir results`
3. Watch for curriculum stage advancement in logs
4. Target: 5-10 million steps or all curriculum stages completed
5. Save best checkpoint at each curriculum stage

### 2.7 Data to Collect
- VictimsRescued per episode
- TotalVictims per episode
- StepsTaken per episode
- PathEfficiency (shortestPath / distanceTraveled)
- DistanceTraveled per episode
- EndReason (success, crash, timeout, out of bounds)
- DroneHP at end of episode
- TensorBoard: mean reward, GoalsReached, PathEfficiency, ExploredCells, AngleStability

---

## PHASE 2.5: Baseline Benchmarks (run after 3D single agent completes)

Run these on the 3D environment only. Each uses 200 evaluation episodes at mid-difficulty (distance=20, obstacles=6).

### Baseline 1 — Random Policy
```csharp
public override void Heuristic(in ActionBuffers actionsOut)
{
    var ca = actionsOut.ContinuousActions;
    ca[0] = Random.Range(-1f, 1f);
    ca[1] = Random.Range(-1f, 1f);
    ca[2] = Random.Range(-1f, 1f);
    ca[3] = Random.Range(-1f, 1f);
}
```
Purpose: sets absolute performance floor.

### Baseline 2 — Greedy Policy
```csharp
public override void Heuristic(in ActionBuffers actionsOut)
{
    var ca = actionsOut.ContinuousActions;
    Transform nearest = envConfig.GetNearestVictim(transform.localPosition);
    if (nearest == null) { ca[0]=0; ca[1]=0; ca[2]=0; ca[3]=0; return; }
    Vector3 toVictim = nearest.position - transform.position;
    Vector3 localDir = transform.InverseTransformDirection(toVictim.normalized);
    ca[0] = Mathf.Clamp(toVictim.y * 0.5f, -1f, 1f);
    ca[1] = Mathf.Clamp(localDir.x, -1f, 1f);
    ca[2] = Mathf.Clamp(localDir.z, -1f, 1f);
    ca[3] = 0f;
}
```
Purpose: proves RL beats naive navigation, especially with obstacles.

### Baseline 3 — PPO Without Curriculum
Remove curriculum from yaml, set fixed values:
```yaml
environment_parameters:
  target_distance:
    value: 20.0
  obstacle_count:
    value: 6
```
Run ID: drone3d_ppo_nocurriculum
Purpose: proves curriculum is a genuine contribution.

### Baseline Comparison Table

| Metric | Random | Greedy | PPO No Curriculum | PPO 3D Full |
|---|---|---|---|---|
| Avg victims rescued | | | | |
| Mission success rate | | | | |
| Avg steps to completion | | | | |
| Path efficiency | | | | |
| Crash rate | | | | |
| Timeout rate | | | | |

---

## PHASE 3: 3D Multi Agent PPO (NEXT)

### 3.1 Environment Changes Required
- Add second DroneController to scene
- Both drones share behavior name DroneAgent
- ML-Agents handles shared policy automatically
- Add victim reservation system to DroneEnvironmentConfig:

```csharp
private Dictionary<GameObject, string> victimReservedBy = new Dictionary<GameObject, string>();

public bool TryReserveVictim(GameObject victim, string droneId)
{
    if (victimReservedBy.ContainsKey(victim)) return false;
    victimReservedBy[victim] = droneId;
    return true;
}

public void ClearReservations()
{
    victimReservedBy.Clear();
}
```

Call ClearReservations() in UpdateEnvironment().

### 3.2 Observation Changes
- Add relative position of other drone (3 values)
- Total observations: 23
- Update BehaviorParameters Vector Observation Size to 23

### 3.3 Configuration
Same yaml as 3D single agent, same curriculum, same reward function.
Run ID: drone3d_ppo_multi_v1

### 3.4 Metrics to Compare vs Single Agent
- Total victims rescued per episode
- Time to clear all victims
- Did drones learn to split territory or follow each other?
- Drone-drone collision frequency
- Path efficiency per drone

---

## PHASE 4: Ablation Studies (run on 3D single agent only)

Run 200 evaluation episodes each at mid-difficulty (distance=20, obstacles=6).

### Ablation 1 — No GPS Hints (hardest)
Remove victim observations entirely:
```csharp
sensor.AddObservation(Vector3.zero);
sensor.AddObservation(0f);
```
What it proves: victim direction hints are necessary for navigation.

### Ablation 2 — No Curriculum
Already covered in Baseline 3. Reference that result.
What it proves: progressive difficulty is necessary for convergence.

### Ablation 3 — No Movement Reward
Remove GiveScanningRewards() call from OnActionReceived.
What it proves: movement incentive reduces idle behavior.

### Ablation 4 — No LSTM (plain MLP)
Remove memory block from yaml network settings.
What it proves: LSTM aids multi-victim sequential search.

### Ablation 5 — No Curiosity
Set curiosity strength to 0.0 in yaml.
What it proves: curiosity aids exploration in large 3D arena.

### Ablation Results Table

| Configuration | Victims Rescued | Path Efficiency | Success Rate | Steps to First Rescue |
|---|---|---|---|---|
| Full 3D system | | | | |
| No GPS hints | | | | |
| No curriculum | | | | |
| No movement reward | | | | |
| No LSTM | | | | |
| No curiosity | | | | |

Priority if time is limited: Ablation 1 and 2 are most important. 3, 4, 5 are optional.

---

## PHASE 5: Results and Analysis

### 5.1 Cross-Environment Comparison Table

| Metric | 2D Single | 2D Multi | 3D Single | 3D Multi |
|---|---|---|---|---|
| Avg victims rescued | | | | |
| Mission success rate | | | | |
| Avg steps to completion | | | | |
| Path efficiency | | | | |
| Training steps to converge | | | | |
| Reward std (stability) | | | | |
| Crash rate | | | | |
| Timeout rate | | | | |

### 5.2 Baseline Comparison Table

| Metric | Random | Greedy | No Curriculum | PPO 3D Full |
|---|---|---|---|---|
| Avg victims rescued | | | | |
| Mission success rate | | | | |
| Path efficiency | | | | |
| Crash rate | | | | |

### 5.3 Ablation Table
(see Phase 4)

### 5.4 Statistical Analysis
- Run each experiment 3 times with different random seeds
- Report mean and std across seeds
- Use t-test or Mann-Whitney U between 2D and 3D results
- Report p-values in results tables

### 5.5 Graphs to Generate
- Learning curves: all 4 experiments on one plot (mean reward vs steps)
- Curriculum advancement timeline: 3D single and multi side by side
- Bar chart: mission success rate across all experiments
- Box plots: path efficiency distribution per experiment
- Bar chart: ablation study victims rescued vs full system
- Line chart: victims rescued per episode over training (2D vs 3D)

---

## PHASE 6: Thesis Report Structure

### Title
Progressive Complexity in Drone-Based Search and Rescue: A Comparative Study of Single and Multi-Agent PPO from 2D to 3D Environments

---

### Abstract (300-500 words)
- Problem: autonomous drone SAR in unknown environments
- Approach: PPO with curriculum learning, 2D to 3D progression
- Experiments: single vs multi agent in both environments
- Key findings: how complexity affects learning, whether coordination scales
- Contribution: first systematic 2D-to-3D comparison of single and multi-agent drone SAR

---

### Chapter 1: Introduction
- Motivation: why autonomous drone SAR matters
- Problem statement: navigating unknown environments to find victims
- Research questions (4 questions listed above)
- Contributions of the thesis
- Thesis structure overview

---

### Chapter 2: Literature Review
- Reinforcement learning fundamentals: MDP, policy gradient, actor-critic
- PPO: Schulman et al. 2017
- Multi-agent RL: cooperative settings, shared vs independent policy
- Curriculum learning: Bengio et al. 2009
- Drone control with RL: existing 2D and 3D work
- Search and rescue with autonomous systems
- Gap: no systematic 2D-to-3D comparison of single and multi-agent drone SAR with curriculum

---

### Chapter 3: Methodology

#### 3.1 2D Environment
- Arena design, observation space, action space
- Reward function for 2D
- Curriculum design for 2D
- Multi-agent setup for 2D

#### 3.2 3D Environment
- Unity 3D with ML-Agents toolkit
- Arena dimensions and rationale
- Drone physics: Angle Mode, thrust, tilt compensation
- Observation space: all 20 observations with rationale
- Action space: 4 continuous actions
- Camera sensor: 84x84, nature_cnn, LSTM
- Reward function: full description with table
- Curriculum: 9 stages, driver/follower design

#### 3.3 Multi-Agent Design (3D)
- Shared policy rationale
- Victim reservation system
- Additional observations

#### 3.4 PPO Configuration and Rationale
- Hyperparameters table with rationale for each value
- Network architecture choices

#### 3.5 Baseline and Ablation Design
- Why each baseline was chosen
- Why each ablation was chosen

---

### Chapter 4: Results

#### 4.1 2D Results
- Single agent learning curve and final metrics
- Multi agent learning curve and final metrics
- 2D single vs multi comparison
- Did coordination emerge in 2D?

#### 4.2 Baseline Results (3D)
- Random, greedy, no-curriculum comparison
- Establishes performance floor and proves curriculum value

#### 4.3 3D Single Agent Results
- Learning curve with curriculum stage markers
- Curriculum progression analysis
- Final performance metrics
- Comparison vs 2D single agent

#### 4.4 3D Multi Agent Results
- Learning curve
- Coordination behavior analysis
- Final performance metrics
- Comparison vs 3D single agent
- Comparison vs 2D multi agent

#### 4.5 Ablation Results
- Ablation table populated
- Which component contributed most
- Visualization

#### 4.6 Cross-Environment Summary
- Full comparison table populated
- Key findings

---

### Chapter 5: Discussion
- 2D to 3D generalization: what transferred, what had to be relearned
- Did multi-agent coordination improve in 3D vs 2D?
- Curriculum design: was the same structure effective in both environments?
- Why the drone hovered/spun after rescuing nearby victims and how it was fixed
- Limitations:
  - Simulation only
  - Fixed arena size
  - Victims do not move
  - Shared policy may limit coordination depth
- Threats to validity

---

### Chapter 6: Conclusion
- Answer each research question with evidence
- Which configuration is recommended for drone SAR
- Practical implications
- Future work:
  - Real hardware transfer (sim-to-real)
  - Moving victims
  - Larger arenas with more drones
  - Communication between drones (QMIX, MADDPG)
  - SAC comparison as future work
  - True vision-based navigation

---

### References
- Schulman et al. 2017 — PPO
- Mnih et al. 2015 — DQN (nature_cnn origin)
- Bengio et al. 2009 — Curriculum learning
- Juliani et al. 2020 — ML-Agents toolkit
- OpenAI Five 2019 — multi-agent cooperation
- Any 2D/3D drone RL papers from your literature search

---

### Appendices
- A: Full 2D reward function code
- B: Full 3D reward function code
- C: Full curriculum yaml
- D: TensorBoard screenshots for all 4 experiments
- E: Raw CSV data summary tables
- F: Unity environment screenshots (2D and 3D)
- G: Key decisions log

---

## TIMELINE ESTIMATE

| Phase | Estimated Time |
|---|---|
| 2D data collection (eval runs) | 2-3 days |
| 3D single agent training | 2-3 weeks (in progress) |
| Baseline benchmarks | 3-5 days |
| 3D multi-agent setup and training | 2-3 weeks |
| Ablation studies (5 runs) | 1 week |
| Data analysis and graphs | 1 week |
| Thesis writing | 4-6 weeks |
| Revision and submission | 2 weeks |
| **Total** | **~16-20 weeks** |

---

## CHECKLIST BEFORE EACH 3D TRAINING RUN

- [ ] visibilityMask set in Inspector (Obstacle + Wall only, NOT Ground)
- [ ] cameraFOV = 100 in DroneController Inspector
- [ ] Camera sensor stack size = 1
- [ ] Vector observation stacked = 1
- [ ] Dead code removed from OnActionReceived
- [ ] recordEvaluationMetrics = true for final test runs
- [ ] yaml saved as UTF-8 without BOM
- [ ] run-id is unique
- [ ] TensorBoard open and monitoring
- [ ] CSV log path verified

---

## KEY DECISIONS LOG (for Methods chapter)

| Decision | Rationale |
|---|---|
| 2D before 3D | Progressive complexity — simpler environment validates approach |
| Angle Mode instead of direct force | Realistic drone behavior |
| Full GPS hints instead of vision-gated | Stability and trainability — camera still present for supplementary context |
| nature_cnn over resnet | Faster training, sufficient for 84x84 |
| LSTM over frame stacking | Better temporal memory for multi-victim search |
| Curiosity strength 0.005 | Prevent random spinning, aid exploration |
| Camera FOV 100 degrees | Matches actual Unity camera setting |
| Victim reservation system | Prevents double-rescue in multi-agent |
| Urgency multiplier in progress reward | Encourages pursuing remaining victims rather than hovering |
| Scaled timeout penalty | Leaving victims unrescued costs proportionally more |
| Shared policy in multi-agent | Simpler, well-validated approach — independent policy is future work |