# Thesis Roadmap: Multi-Agent Drone Search and Rescue Using Deep Reinforcement Learning

## Plan A — Full Roadmap from Simulation to Submission

---

## PHASE 1: Environment Setup and Baseline (COMPLETED)

### 1.1 Unity Environment
- 3D arena: 50x50x20 units
- Drone with Angle Mode physics controller
- Victim prefabs spawned at curriculum-defined distances
- Obstacle prefabs with tag-based collision detection
- DroneEnvironmentConfig managing spawn, curriculum, and victim rescue logic

### 1.2 Agent Design
- DroneController extending ML-Agents Agent class
- 20 vector observations:
  - Position normalized by arena size (3)
  - Local velocity normalized by maxVelocity (3)
  - Angular velocity normalized by maxYawSpeed in radians (3)
  - Local rotation as quaternion (4)
  - Victim direction hint — vision gated, coarse sign only (3)
  - Victim distance — coarse bucket near/mid/far (1)
  - Mission progress — rescued / total (1)
  - Remaining victims ratio (1)
  - Ground distance normalized (1)
- 4 continuous actions: throttle, roll, pitch, yaw
- Camera sensor: 84x84 RGB, stack size 1 (LSTM handles temporal memory)
- CanSeeVictim: FOV + range + raycast LOS gate for vision-based hint unlock

### 1.3 Reward System
- Hover reward: altitude error + low velocity bonus
- Progress reward: delta distance to nearest victim, obstacle-aware multiplier
- Scanning reward: yaw rotation when victim not visible
- Aha reward: one-time bonus when victim first enters FOV (per victim per episode)
- Rescue reward: base 5.0 + time bonus + difficulty bonus
- Completion bonus: 10.0 + efficiency bonus when all victims rescued
- Penalties: ground proximity, obstacle collision, idle hovering, out of bounds, timeout

### 1.4 Curriculum Design
- 9 stages: hover_training to expert_flight
- Driver: target_distance
- Follower: obstacle_count
- min_lesson_length: 100 for hover, 200 for all navigation stages
- signal_smoothing: true on all stages

---

## PHASE 2: Experiment 1 — PPO Single Agent (IN PROGRESS)

### 2.1 Configuration
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
```

### 2.2 Training Steps
1. Run: `mlagents-learn config3D/single_occ.yaml --run-id=drone3d_ppo_v1 --train`
2. Monitor TensorBoard: `tensorboard --logdir results`
3. Watch for curriculum stage advancement in logs
4. Target: 5-10 million steps or all curriculum stages completed
5. Save best checkpoint at each curriculum stage

### 2.3 Data to Collect (CSV logging already implemented)
- VictimsRescued per episode
- TotalVictims per episode
- StepsTaken per episode
- PathEfficiency (shortestPath / distanceTraveled)
- DistanceTraveled per episode
- EndReason (success, crash, timeout, out of bounds)
- DroneHP at end of episode

### 2.4 TensorBoard Metrics to Screenshot
- Mean reward curve (full training run)
- GoalsReached over time
- PathEfficiency over time
- GroundDistance over time
- ExploredCells over time
- AngleStability over time
- Curriculum stage advancement markers

---

## PHASE 2.5: Baseline Benchmarks

Run these BEFORE SAC and multi-agent experiments so you have baselines to compare everything against. Each baseline uses the same environment and same evaluation metrics as your main experiments.

### 2.5.1 Baseline 1 — Random Policy
The simplest possible baseline. Drone takes random continuous actions every step.

**How to run:**
- In Unity set Behavior Type to Heuristic
- Override Heuristic() to return random values:

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

- Run 200 episodes at mid-difficulty stage (distance=20, obstacles=6)
- Record all CSV metrics
- Expected result: near zero victims rescued, mostly timeouts and crashes

**Purpose:** Proves any learned policy is better than chance. Sets the absolute floor for your comparison table.

---

### 2.5.2 Baseline 2 — Greedy Policy
Drone always flies directly toward the nearest victim with no obstacle avoidance or scanning.

**How to run:**
- Override Heuristic() with greedy logic:

```csharp
public override void Heuristic(in ActionBuffers actionsOut)
{
    var ca = actionsOut.ContinuousActions;
    Transform nearest = envConfig.GetNearestVictim(transform.localPosition);
    if (nearest == null) { ca[0] = 0; ca[1] = 0; ca[2] = 0; ca[3] = 0; return; }

    Vector3 toVictim = nearest.position - transform.position;
    Vector3 localDir = transform.InverseTransformDirection(toVictim.normalized);

    ca[0] = Mathf.Clamp(toVictim.y * 0.5f, -1f, 1f);
    ca[1] = Mathf.Clamp(localDir.x, -1f, 1f);
    ca[2] = Mathf.Clamp(localDir.z, -1f, 1f);
    ca[3] = 0f;
}
```

- Run 200 episodes at mid-difficulty stage
- Record all CSV metrics
- Expected result: rescues victims in open stages, fails when obstacles present

**Purpose:** Proves RL is necessary for obstacle avoidance and that your system adds value beyond naive navigation.

---

### 2.5.3 Baseline 3 — PPO Without Curriculum
Train PPO from scratch starting directly at mid-difficulty with no curriculum progression.

**How to run:**
- Copy your yaml, remove all curriculum stages
- Set fixed values:

```yaml
environment_parameters:
  target_distance:
    value: 20.0
  obstacle_count:
    value: 6
```

- Train for same number of steps as main PPO run
- Run ID: drone3d_ppo_nocurriculum

**Purpose:** Proves your curriculum design is a genuine contribution. If no-curriculum PPO performs worse, curriculum is justified.

---

### 2.5.4 Baseline Comparison Table

| Metric | Random | Greedy | PPO No Curriculum | PPO Full | SAC Full |
|---|---|---|---|---|---|
| Avg victims rescued | | | | | |
| Mission success rate | | | | | |
| Avg steps to completion | | | | | |
| Path efficiency | | | | | |
| Crash rate | | | | | |
| Timeout rate | | | | | |

---

## PHASE 3: Experiment 2 — SAC Single Agent

### 3.1 Why SAC
- SAC (Soft Actor-Critic) is designed for continuous action spaces
- Off-policy: more sample efficient than PPO
- Entropy maximization: better exploration without manual beta tuning
- Expected to converge faster and with less reward variance

### 3.2 Configuration Changes
```yaml
trainer_type: sac
hyperparameters:
  learning_rate: 0.0003
  learning_rate_schedule: constant
  batch_size: 256
  buffer_size: 50000
  tau: 0.005
  steps_per_update: 1
  save_replay_buffer: false
  init_entcoef: 0.5
  reward_signal_steps_per_update: 1
```
- Keep all network settings identical to PPO run
- Keep same curriculum and thresholds
- Keep same reward function — no changes to Unity code
- Run ID: `drone3d_sac_v1`

### 3.3 Key Differences to Observe
- How many steps to reach each curriculum stage vs PPO
- Reward variance (std) — SAC should be lower
- Whether SAC gets stuck at same stages as PPO
- Final path efficiency comparison

---

## PHASE 4: Experiment 3 — PPO Multi Agent

### 4.1 Environment Changes Required
- Add second DroneController to scene
- Both drones share the same behavior name (DroneAgent)
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

- Call ClearReservations() in UpdateEnvironment()
- Add drone ID field to DroneController
- Check reservation before rescue attempt

### 4.2 Observation Changes
- Add relative position of other drone as observation (3 values)
- Total observations: 23
- Update BehaviorParameters Vector Observation Size to 23

### 4.3 Configuration
```yaml
trainer_type: ppo
# same hyperparameters as Experiment 1
# same curriculum
```

### 4.4 Metrics to Compare vs Single Agent
- Total victims rescued per episode (should be higher with 2 drones)
- Time to clear all victims (steps taken)
- Do drones learn to split up or follow each other?
- Collision between drones (add penalty if needed)

---

## PHASE 5: Experiment 4 — SAC Multi Agent

### 5.1 Configuration
- Same environment as Experiment 3
- Change trainer_type to sac
- Same SAC hyperparameters as Experiment 2
- Run ID: `drone3d_sac_multi_v1`

### 5.2 This is Your Final Experiment
- Collect all same metrics as previous runs
- This closes your comparison matrix

---

---

## PHASE 5.5: Ablation Studies

Run ablations on single-agent PPO only. Each ablation removes or disables one component to prove it contributes to performance. Use the same evaluation stage (distance=20, obstacles=6) and run 200 episodes each.

### 5.5.1 Ablation 1 - No Vision Gating (GPS Mode)

Remove CanSeeVictim and always provide exact victim direction.

What to change in CollectObservations - replace vision-gated block with always-on GPS direction:

```csharp
Transform nearestVictim = envConfig.GetNearestVictim(transform.localPosition);
if (nearestVictim != null)
{
    Vector3 toVictim = nearestVictim.localPosition - transform.localPosition;
    Vector3 localDir = transform.InverseTransformDirection(toVictim).normalized;
    sensor.AddObservation(localDir.x);
    sensor.AddObservation(localDir.y);
    sensor.AddObservation(localDir.z);
    float dist = toVictim.magnitude;
    float coarseDist = dist < 10f ? 1f : dist < 30f ? 0.5f : 0f;
    sensor.AddObservation(coarseDist);
    isVictimVisible = true;
}
```

Expected result: Faster victim finding but drone ignores camera, flies backward blindly.
What it proves: Vision gating is necessary for realistic camera-based navigation.

---

### 5.5.2 Ablation 2 - No Scanning Reward

Remove GiveScanningRewards() call from OnActionReceived.

```csharp
// Comment out this line in OnActionReceived:
// GiveScanningRewards();
```

Expected result: Drone does not learn to rotate and search, gets stuck when victim not in FOV.
What it proves: Scanning reward is necessary to bootstrap search behavior.

---

### 5.5.3 Ablation 3 - No Curriculum

Already covered in Baseline 3 (Phase 2.5.3). Reference that result here.
What it proves: Progressive difficulty is necessary for convergence in complex stages.

---

### 5.5.4 Ablation 4 - No Curiosity

Set curiosity strength to 0 in yaml:

```yaml
curiosity:
  strength: 0.0
```

Expected result: Slower exploration, agent stays near spawn point longer.
What it proves: Curiosity aids exploration in large 3D environments.

---

### 5.5.5 Ablation 5 - No LSTM (Plain MLP)

Remove memory block from network settings in yaml:

```yaml
network_settings:
  normalize: true
  hidden_units: 256
  num_layers: 2
  vis_encode_type: nature_cnn
```

Expected result: Drone cannot remember where it searched, revisits same areas repeatedly.
What it proves: LSTM is necessary for efficient multi-victim search across an episode.

---

### 5.5.6 Ablation Results Table

| Configuration | Victims Rescued | Path Efficiency | Success Rate | Steps to First Rescue |
|---|---|---|---|---|
| Full system | | | | |
| No vision gating | | | | |
| No scanning reward | | | | |
| No curriculum | | | | |
| No curiosity | | | | |
| No LSTM | | | | |

Priority order if time is limited:
1. No vision gating - your core design contribution
2. No curriculum - your most impactful engineering decision
3. No scanning reward - supports reward design chapter
4. No LSTM - supports architecture choice
5. No curiosity - least critical, skip if time runs out

---

## PHASE 6: Results and Analysis

### 6.1 Primary Comparison Table
Populate this table from your CSV logs and TensorBoard:

| Metric | PPO Single | SAC Single | PPO Multi | SAC Multi |
|---|---|---|---|---|
| Avg victims rescued | | | | |
| Mission success rate | | | | |
| Avg steps to completion | | | | |
| Path efficiency | | | | |
| Training steps to converge | | | | |
| Reward std (stability) | | | | |
| Crash rate | | | | |
| Timeout rate | | | | |

### 6.2 Statistical Analysis
- Run each experiment 3 times with different seeds
- Report mean and standard deviation across seeds
- Use t-test or Mann-Whitney U to test significance between PPO and SAC
- Report p-values in results table

### 6.3 Graphs to Generate
- Learning curves: mean reward vs steps for all 4 experiments (one plot, 4 lines)
- Curriculum advancement timeline: which stage reached at which step
- Box plots: path efficiency distribution per experiment
- Bar chart: mission success rate per experiment
- Line chart: victims rescued per episode over training

---

## PHASE 7: Thesis Report Structure

### Title
Comparative Analysis of PPO and SAC Algorithms for Single and Multi-Agent Drone-Based Search and Rescue in 3D Environments

---

### Abstract (300-500 words)
- Problem: search and rescue in unknown 3D environments
- Approach: deep reinforcement learning with curriculum learning
- Experiments: PPO vs SAC, single vs multi agent
- Key findings: which algorithm performed best and why
- Contribution: first comparison of these four configurations in a 3D drone SAR context

---

### Chapter 1: Introduction
- Motivation: why autonomous drone SAR matters
- Problem statement: navigating unknown 3D environments to find victims
- Research questions:
  1. Does SAC outperform PPO for continuous drone control?
  2. Does multi-agent coordination improve SAR performance?
  3. Which combination is most sample efficient?
- Contributions of the thesis
- Thesis structure overview

---

### Chapter 2: Literature Review
- Reinforcement learning fundamentals: MDP, policy gradient, actor-critic
- PPO: Schulman et al. 2017 — proximal policy optimization
- SAC: Haarnoja et al. 2018 — soft actor-critic
- Multi-agent RL: cooperative vs competitive, shared vs independent policy
- Curriculum learning in RL: Bengio et al. 2009
- Drone control with RL: existing work in 2D and 3D
- Search and rescue with autonomous systems: existing work
- Gap in literature: no direct PPO vs SAC comparison in 3D drone SAR

---

### Chapter 3: Methodology

#### 3.1 Simulation Environment
- Unity 3D with ML-Agents toolkit
- Arena dimensions and design rationale
- Drone physics: Angle Mode, thrust, tilt compensation
- Victim spawning and rescue mechanics
- Obstacle placement and collision system

#### 3.2 Agent Design
- Observation space: describe all 20 observations and why each was chosen
- Action space: 4 continuous actions, Angle Mode rationale
- Vision system: CanSeeVictim, FOV=100, range=25, LOS raycast
- Why vision-gated hints instead of GPS: forces camera usage

#### 3.3 Reward Function
- Table of all reward components with values and rationale
- Scanning reward design rationale
- Aha reward and anti-farming mechanism (HashSet)
- Rescue reward structure
- Penalty design rationale

#### 3.4 Curriculum Learning
- Stage progression table
- Driver/follower design
- Threshold and min_lesson_length rationale
- How premature advancement was identified and fixed

#### 3.5 Algorithm Configurations
- PPO hyperparameters and rationale
- SAC hyperparameters and rationale
- Network architecture: nature_cnn + LSTM
- Why nature_cnn over resnet or simple
- Why LSTM for temporal memory

#### 3.6 Multi-Agent Design
- Shared policy rationale
- Victim reservation system
- Additional observations for multi-agent

---

### Chapter 4: Results

#### 4.1 Baseline and Benchmark Results
- Random policy results: victims rescued, crash rate, timeout rate
- Greedy policy results: performance in open vs obstacle stages
- PPO no-curriculum results: convergence comparison vs full curriculum
- Baseline comparison table (fully populated)
- Key finding: RL outperforms baselines by X percent, curriculum improves convergence by Y steps

#### 4.2 Ablation Study Results
- Ablation table (fully populated)
- No vision gating: how much faster does agent find victims vs full system
- No scanning reward: how many episodes before first victim rescue
- No LSTM: comparison of revisit rate and exploration coverage
- No curiosity: comparison of exploration speed
- Key finding: which component contributes most to performance
- Visualize with bar chart: each ablation vs full system on victims rescued

#### 4.3 Experiment 1 - PPO Single Agent
- Learning curve with curriculum stage markers
- Final performance metrics
- Curriculum stages reached
- Analysis of collapse at mid_distance (first run) and fix applied
- Comparison vs baselines and ablations

#### 4.4 Experiment 2 - SAC Single Agent
- Learning curve comparison vs PPO
- Convergence speed comparison
- Final performance metrics
- Stability comparison (reward std)
- Steps to reach each curriculum stage: PPO vs SAC side by side

#### 4.5 Experiment 3 - PPO Multi Agent
- Learning curve
- Emergent coordination behavior (did drones split up?)
- Performance vs single agent PPO
- Victim reservation system effectiveness
- Drone collision frequency if tracked

#### 4.6 Experiment 4 - SAC Multi Agent
- Learning curve
- Performance vs all previous experiments
- Final comparison table (fully populated)

---

### Chapter 5: Discussion

- Why PPO collapsed at mid_distance jump (catastrophic forgetting)
- Why curriculum design matters more than algorithm choice at early stages
- SAC vs PPO tradeoffs in this specific domain
- Multi-agent coordination: did it emerge or did drones interfere?
- Limitations:
  - Simulation only, not real hardware
  - Fixed arena size
  - No wind simulation in final runs
  - Victims do not move
- Threats to validity

---

### Chapter 6: Conclusion

- Answer each research question with evidence from results
- Which algorithm and configuration is recommended for drone SAR
- Practical implications for real-world deployment
- Future work:
  - Real hardware transfer (sim-to-real)
  - Moving victims
  - Larger arenas with more drones
  - Communication between drones (QMIX, MADDPG)
  - Wind and weather disturbances

---

### References
Key papers to cite:
- Schulman et al. 2017 — PPO
- Haarnoja et al. 2018 — SAC
- Mnih et al. 2015 — DQN (nature_cnn origin)
- Bengio et al. 2009 — Curriculum learning
- Juliani et al. 2020 — ML-Agents toolkit
- OpenAI Five 2019 — multi-agent cooperation
- Any drone RL papers you find in your literature search

---

### Appendices
- A: Full reward function code
- B: Full observation space code
- C: Full curriculum yaml
- D: TensorBoard screenshots for all 4 runs
- E: Raw CSV data summary tables
- F: Unity environment screenshots

---

## TIMELINE ESTIMATE

| Phase | Estimated Time |
|---|---|
| PPO single agent training | 2-3 weeks (running now) |
| Baseline benchmarks (random, greedy, no-curriculum) | 3-5 days |
| SAC single agent training | 1-2 weeks |
| Ablation studies (5 runs x 200 episodes each) | 1 week |
| Multi-agent environment setup | 1 week |
| PPO multi-agent training | 2 weeks |
| SAC multi-agent training | 1-2 weeks |
| Data analysis and graphs | 1-2 weeks |
| Thesis writing | 4-6 weeks |
| Revision and submission | 2 weeks |
| **Total** | **~18-23 weeks** |

---

## CHECKLIST BEFORE EACH TRAINING RUN

- [ ] visibilityMask set correctly in Inspector (Obstacle + Wall only)
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
| Angle Mode instead of direct force | More realistic, matches real drone behavior |
| Vision-gated hints instead of GPS | Forces camera usage, more realistic SAR |
| nature_cnn over resnet | Faster training, sufficient for 84x84 |
| LSTM over frame stacking | Better temporal memory, less input size |
| Curiosity strength 0.005 | Prevent random spinning, aid exploration |
| Camera FOV 100 degrees | Matches actual Unity camera setting |
| successRadius from config | Allows curriculum tuning without code change |
| Victim reservation system | Prevents double-rescue in multi-agent |
| HashSet for aha reward | Prevents farming by FOV oscillation |
| Coarse sign hints only | Prevent GPS cheating while aiding early learning |
