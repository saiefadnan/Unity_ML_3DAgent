# Reward Design
*Detailed shaping of the reinforcement learning reward function*

## Reward Components Table

| Component | Formula / Expression | Purpose | Magnitude |
|-----------|----------------------|---------|-----------|
| **Step Penalty** | `-0.005f` | Encourages resolving the episode fast. | Small |
| **Approach Reward** | `Mathf.Clamp(delta * 2f, 0f, 0.5f) * (1f + (victimsLeft * 0.1f))` | Dense reward for halving distance to victim. Scaled by urgency multiplier. | Moderate |
| **Victim Found** | `5.0f + ((epLength - stepCount) / epLength) * 1.0f + (obsCount > 0 ? 2.0f : 0f)` | Primary success signal. Contains time and difficulty bonuses. | Large (+5 to +8) |
| **Full Completion** | `10.0f + (shortestPath / distanceTraveled) * 5.0f` | Rewarded when all victims are found. Includes path efficiency bonus. | Extra Large |
| **Collision (Ground)**| `-0.5f` to `-2.0f` | Penalizes crashing/rough land based on HP damage. -3f in hover mode. | Large |
| **Collision (Obstacle)**| `-(damage / MAX_HP * 3f + 3f)` | Heavy penalty to discourage drone destruction. Conditionally ends episode. | Large |
| **Blind Penalty** | `-Mathf.Clamp(backtrackDelta * 1.5f, 0f, 0.4f)` | Penalizes moving away from the target, scales with `backtrackCounter`. | Moderate |
| **Exploration** | `+0.05f / (1f + visitedCells.Count * 0.05f)` | Discovery reward for covering new volume grid coordinates. | Small |
| **Hover Stability** | Altitude based: `<0.5f = +0.15f`, `>2.0 = -0.02f` | Trains stabilization during empty environments. | Moderate |

## Curriculum Stage Design Rationale
- **hover_training**: With 0 objects, forces the drone to counteract gravity, eliminating drone control ambiguity before target tracking.
- **close_target / medium_target**: Target increments to distances of 5.0 and 15.0 to familiarize the pitch/thrust combinations necessary for xy navigation.
- **static_obstacles / complex_obstacles**: The target distance effectively halts whilst obstacles are dropped directly in flight bounds. Teaches collision deterrence.
- **expert_flight**: Tests ultimate generalization with 12 obstacles and targets 60 range units away.

## Known Failure Modes Prevented
- **Circling/Spinning in place**: Monitored strictly; a yaw rate > 0.3 without proper xz propulsion incurs a continuous `-0.01f` penalty.
- **Hover stalling**: To limit stagnation during searches, hovering identically for 30 cycles rapidly bleeds the reward (by `-1f`) and forcefully aborts the trial.
- **Blind Backtracking**: Prevents erratic loops by penalizing any delta expanding the `closestDistanceEver` boundary unless specifically dodging obstacles.

## Related Docs
- [RL Algorithm](./rl_algorithm.md)
- [Results](./results.md)
- [Overview](./overview.md)
