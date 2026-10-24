# Environment and Task

_Definition of the 3D Unity SAR environment and state spaces_

## Task Definition

The Search-and-Rescue (SAR) task requires the drone(s) to navigate a 3D arena, avoid static and complex obstacles, and locate a specified number of victims.

- **Success:** Locating all victims before the episode lengths timeout (5000 steps).
- **Failure:** Falling below ground level, flying out of bounds, or losing all drone HP (0/500) due to severe collisions.

## 3D Unity Environment

- **Arena Bounds:** 160(x) × 40(y) × 160(z) units with the center shifted to (0, 20, 0).
- **Agent Spawn:** Drones spawn around the x/z center (±5) maintaining a fixed `spawnHeight` variable (defaulted to 10 units).
- **Victim Placement:** Victims are placed at a variable `targetDistance`, randomized across the available volume while ensuring a minimum spacing of 3 units.
- **Obstacle Placement:** Obstacles are placed randomly across the floor, avoiding key objects, and rotated randomly along the Y-axis.

## Observation Space Breakdown

### S3 (Single-Agent) - 20 Dimensions

| Group       | Dimensions | Label / Description                          |
| ----------- | ---------- | -------------------------------------------- |
| Drone State | 3          | Normalized Local Position                    |
| Drone State | 3          | Normalized Linear Velocity                   |
| Drone State | 3          | Normalized Angular Velocity (yaw)            |
| Drone State | 4          | Local Rotation (Quaternion)                  |
| GPS Target  | 4          | Local direction to victim (3) + Distance (1) |
| Mission     | 1          | Progression (Goals Reached fraction)         |
| Mission     | 1          | Progression (Remaining fraction)             |
| Proximity   | 1          | Ground Distance Raycast (Normalized)         |

### S4 (Multi-Agent) - 34 Dimensions

| Group       | Dimensions | Label / Description                              |
| ----------- | ---------- | ------------------------------------------------ |
| Drone State | 13         | Pos (3), Vel (3), AngVel (3), Rot (4)            |
| GPS Target  | 4          | Target Data local direction (3) + Distance (1)   |
| Mission     | 2          | Team Found Fraction, Remaining Fraction          |
| Proximity   | 1          | Ground Distance                                  |
| Teammate 1  | 7          | Relative Pos (3), Rel Vel (3), Normalized HP (1) |
| Teammate 2  | 7          | Relative Pos (3), Rel Vel (3), Normalized HP (1) |

> **Note:** The progression metrics in S4 are team-wide, tracking collective goals.

## Action Space

The drone uses **4 Continuous Actions** [-1, 1], representing quadcopter flight controls:

1. `throttle`: Controls target altitude/vertical acceleration.
2. `roll`: Sideways tilt (left/right).
3. `pitch`: Forward/backward tilt.
4. `yaw`: Rotational spin around the Y-axis.

## Raycast Obstacle Detection

The drone features a LiDAR-inspired proximity sensor leveraging `RayPerceptionSensorComponent3D`, which casts a **51-ray horizontal ring** (25 rays per direction + 1 forward ray, spanning a full 360 degrees). These rays detect colliders on the `Ground`, `Wall`, `Obstacle`, and `Victim` layers. The sensor utilizes a sphere cast radius of 0.5 and a length of 20 units, feeding dense distance measurements natively into ML-Agents for robust obstacle avoidance.

## Related Docs

- [RL Algorithm](./rl_algorithm.md)
- [Reward Design](./reward_design.md)
- [Overview](./overview.md)
