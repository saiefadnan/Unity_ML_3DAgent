# Multi-Agent Cooperative Design
*Differences and enhancements from S3 to S4 configuration*

## Transition Differences (S3 vs S4)
The Multi-Agent architecture heavily upgrades the baseline network and environment scope:
- **Observation Additions:** Adds 14 dimensions (two teammates, each supplying relative position `(3)`, velocity `(3)`, and normalized HP `(1)`). Also tracks team-wide explored cells and team victims found.
- **Network Scaled:** Fully expanded from `256×2` to `512×3` to parse the 34-dimension vector effectively.
- **Buffer Scaled:** Buffer dramatically expanded from 10,240 to 81,920 supporting the multi-data payload. Batch size quadrupled from 512 to 2048.
- **Thresholds Raised:** Lesson benchmarks drastically boosted. The `min_lesson_length` requires 200–500 episodes (vs 20 in single-agent) for steady swarm confidence. Initial reward goals advanced from 8->12 and 15->20.

## Config Diff Table
| Parameter | S3 (Single) | S4 (Multi) |
|-----------|-------------|------------|
| `batch_size` | 512 | 2048 |
| `buffer_size` | 10240 | 81920 |
| `hidden_units` | 256 | 512 |
| `num_layers` | 2 | 3 |
| `min_lesson_length`| 20 | 200 - 500 |
| Stage Thresholds | 8.0, 15.0... | 12.0, 20.0... |

## GPS Communication
Teammate interactions operate over simulated systemic GPS queries. Agents compute raw comparative world positions directly injected into models (`relPos.x / arenaSize.x`) processed securely by `envConfig.GetTeammateInfo()`.

## Cooperative Target Claiming and Reward Structure
- **Centralized Target Claiming:** `ClaimNearestAvailableVictim` binds indices intelligently blocking repetitive drone lock-ons.
- **Reward Balance:**
  - Standard collision, flying, and proximity rewards uniquely govern individual drones.
  - Interacting with a victim awards a singular `+5.0f` but instantly shares progress.
  - **Shared Policy Boom:** Earning the final `+10f` reward triggers independently for everyone upon achieving swarm completion, fusing decentralized maneuvers into true cooperative strategy.

## Related Docs
- [Environment and Task](./environment_and_task.md)
- [Results](./results.md)
- [Overview](./overview.md)
