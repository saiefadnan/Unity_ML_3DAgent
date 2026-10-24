# RL Algorithm
*Proximal Policy Optimization (PPO) and curriculum learning implementation*

## PPO Theory
The project relies on PPO to optimize the policy network, keeping training stable with a clipped objective function:

$$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right] $$

Where $r_t(\theta)$ is the probability ratio and $\hat{A}_t$ is the Generalized Advantage Estimate (GAE). An entropy bonus enforces exploration.

## Hyperparameter Comparison

| Parameter | S3 (single) | S4 (multi) |
|-----------|-------------|------------|
| batch_size | 512 | 2048 |
| buffer_size | 10240 | 81920 |
| learning_rate | 0.0003 | 0.0003 |
| beta | 0.005 | 0.005 |
| epsilon | 0.2 | 0.2 |
| lambd | 0.95 | 0.95 |
| num_epoch | 3 | 3 |

## Curiosity Module
A curiosity module (ICM) generates an intrinsic reward to encourage exploration of the environment. The `strength` is kept very low (**0.01**) with a discount gamma of 0.99. Highly dense GPS signals heavily guide the system; high curiosity could override the victim-seeking behavior.

## No LSTM in 3D
Recurrent models (LSTM) were explicitly disabled (`memory_block` removed) in 3D. The combination of dense observation features (GPS array) and raycast distance inputs already perfectly encodes spatial positioning over time, preventing LSTM's massive performance hit.

## Curriculum Learning
Training follows an 8-stage ML-Agents curriculum. The lesson difficulty scales upwards only when the agent can consistently achieve reward thresholds (using EMA `signal_smoothing`). Advancements dictate greater distances or denser fields.

| Stage Name | target_distance | obstacle_count | Threshold (S3 vs S4) | min_lesson_length (S3 vs S4) |
|------------|-----------------|----------------|----------------------|------------------------------|
| hover_training | 0.0 | 0 | 8.0 vs 12.0 | 20 vs 200 |
| close_target | 5.0 | 0 | 15.0 vs 20.0 | 20 vs 200 |
| medium_target | 15.0 | 0 | 22.0 vs 28.0 | 20 vs 300 |
| static_obstacles| 15.0 | 3 | 25.0 vs 28.0 | 20 vs 300 |
| complex_obstacles| 20.0 | 6 | 25.0 vs 30.0 | 20 vs 400 |
| mid_distance | 35.0 | 8 | 30.0 vs 35.0 | 20 vs 400 |
| long_distance | 50.0 | 10 | 35.0 vs 40.0 | 20 vs 500 |
| expert_flight | 60.0 | 12 | N/A | N/A |

## Related Docs
- [Network Architecture](./network_architecture.md)
- [Environment and Task](./environment_and_task.md)
- [Overview](./overview.md)
