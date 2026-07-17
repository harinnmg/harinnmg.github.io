# BCR Bot RL Navigation — Training, Deployment & Iteration Guide

> Covers the full cycle: train a PPO policy in Isaac Lab → export → deploy in
> Ignition Gazebo → read logs → diagnose → retrain.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Training in Isaac Lab](#2-training-in-isaac-lab)
3. [Exporting the Policy](#3-exporting-the-policy)
4. [Deploying in Ignition Gazebo](#4-deploying-in-ignition-gazebo)
5. [Reading the Logs](#5-reading-the-logs)
6. [Diagnosing Problems](#6-diagnosing-problems)
7. [Reward Tuning & Retraining](#7-reward-tuning--retraining)
8. [Known Issues & Fix History](#8-known-issues--fix-history)
9. [Quick-Reference Constants](#9-quick-reference-constants)

---

## 1. System Overview

```
Isaac Lab (PhysX)          Ignition Gazebo (deployed)
─────────────────          ──────────────────────────
  BCR-Navigation-v0   ──►  rl_test.launch.py
  2048 parallel envs        │
  RSL-RL PPO                ├─ ign.launch.py  (Gazebo + bridge)
  ↓ export policy.pt        └─ rl_policy_node.py
                                 ├─ sub: /bcr_bot/scan   (LaserScan)
                                 ├─ sub: /bcr_bot/odom   (Odometry)
                                 ├─ sub: /goal_pose      (PoseStamped)
                                 └─ pub: /bcr_bot/cmd_vel (Twist)
```

### Observation space (23-dim)

| Idx | Description | Range |
|-----|-------------|-------|
| 0–16 | 17 LiDAR rays, −90° to +90° (0=right, 8=front, 16=left), normalised | [0, 1] |
| 17 | cos(heading_err) — goal direction in robot frame | [−1, 1] |
| 18 | sin(heading_err) | [−1, 1] |
| 19 | norm_dist — goal distance / GOAL_DIST_MAX | [0, 1] |
| 20 | VAPF Fx (robot frame, tanh-capped to 5) | [−5, 5] |
| 21 | VAPF Fy (robot frame, tanh-capped to 5) | [−5, 5] |
| 22 | VAPF magnitude (tanh-capped to 5) | [0, 5] |

### Action space (2-dim)

Raw network output: `[u_left, u_right]` wheel velocity commands.

```
linear_vel  = R × (u_l + u_r) / 2       R = 0.1 m (wheel radius)
angular_vel = R × (u_r − u_l) / L       L = 0.6 m (wheel track)
```

Both `middle_left_wheel_joint` and `middle_right_wheel_joint` have axis = +Y.

---

## 2. Training in Isaac Lab

### 2.1 Directory layout

```
bcr_botrl1/bcr_botrl/bcr_botrl/
├── source/bcr_botrl/bcr_botrl/tasks/manager_based/bcr_botrl/
│   ├── bcr_navigation_env_cfg.py   ← env config (rewards, obs, LiDAR, episode)
│   ├── agents/rsl_rl_ppo_cfg.py    ← PPO hyperparameters
│   └── mdp/                        ← reward functions
├── scripts/
│   ├── rsl_rl/train.py             ← training entry point
│   └── rsl_rl/export.py            ← export TorchScript
└── logs/rsl_rl/bcr_navigation/
    └── YYYY-MM-DD_HH-MM-SS/        ← one folder per run
        ├── model_N.pt              ← checkpoints every 25 iters
        ├── exported/policy.pt      ← exported TorchScript (after export step)
        └── events.out.tfevents.*   ← TensorBoard log
```

### 2.2 Launch training

```bash
cd ~/bcr_botrl1/bcr_botrl
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task BCR-Navigation-v0 \
  --num_envs 2048
```

Add `--headless` to skip rendering and train faster on a GPU server.

### 2.3 PPO hyperparameters (current)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_steps_per_env` | 128 | Covers 17% of each 30 s episode per update |
| `max_iterations` | 3000 | ~770 M total env steps at 2048 envs |
| `save_interval` | 25 | Frequent enough to capture the peak checkpoint |
| `gamma` | 0.999 | Sparse `goal_reached` reward survives ~50% to episode start |
| `learning_rate` | 1e-4 | Lower LR prevents regression after a peak |
| `clip_param` | 0.15 | Tighter trust region than default 0.2 |
| `entropy_coef` | 0.005 | Maintains exploration through training |
| `lam` | 0.95 | GAE lambda |
| `actor_hidden_dims` | [256, 128] | Plain MLP |
| `critic_hidden_dims` | [256, 128] | Shared architecture |
| `init_noise_std` | 0.7 | Wide initial exploration |

### 2.4 Monitor training with TensorBoard

```bash
tensorboard --logdir ~/bcr_botrl1/bcr_botrl/bcr_botrl/logs/rsl_rl/bcr_navigation
```

Key metrics to watch:

| Metric | Healthy sign | Warning sign |
|--------|-------------|--------------|
| `Train/mean_reward` | Climbing past iter 500 | Flat or oscillating |
| `Train/goal_success` | >30% by iter 1000 | Stays <5% → reward bug |
| `Train/policy_loss` | Small and stable | Spiky or diverging |
| `Train/mean_noise_std` | Decays from 0.7 | Explodes (>10) → LR too high |
| `Train/value_loss` | Near 0 or low | 4–6 → rewards too large |

### 2.5 Reward configuration

File: `bcr_navigation_env_cfg.py`, class `BcrRewardsCfg`

| Term | Weight | Notes |
|------|--------|-------|
| `goal_approach` | +5.0 (gain=3.0) | Dense: Δdistance to goal each step |
| `goal_reached` | +10.0 | Sparse: bonus when within 0.5 m of goal |
| `heading_align` | **+0.3** | `cos(heading_err)` — drives robot to face goal |
| `stationary` | −0.05 | Penalises standing still |
| `action_rate` | −0.0001 | Smoothness penalty |
| `yaw_rate_penalty` | **−0.001** | Small penalty for spinning |

**Critical**: `heading_align` and `yaw_rate_penalty` must not cancel out.
Previous bug: both were `±0.01` → net zero → policy learned to ignore heading.

---

## 3. Exporting the Policy

After training, pick the best checkpoint (highest `goal_success` on TensorBoard)
and export it to a self-contained TorchScript file:

```bash
cd ~/bcr_botrl1/bcr_botrl
./isaaclab.sh -p scripts/rsl_rl/export.py \
  --task BCR-Navigation-v0 \
  --checkpoint logs/rsl_rl/bcr_navigation/YYYY-MM-DD_HH-MM-SS/model_N.pt
```

Output: `logs/rsl_rl/bcr_navigation/YYYY-MM-DD_HH-MM-SS/exported/policy.pt`

The exported file is a `torch.jit.ScriptModule`. It takes a `(1, 23)` float32
tensor and returns a `(1, 2)` tensor `[u_left, u_right]`.

### Update the launch file

Edit [rl_test.launch.py](nav2_ws/src/bcr_bot/launch/rl_test.launch.py):

```python
_DEFAULT_MODEL = expanduser(
    '~/bcr_botrl1/bcr_botrl/bcr_botrl/logs/rsl_rl/bcr_navigation'
    '/YYYY-MM-DD_HH-MM-SS/exported/policy.pt'   # ← update timestamp
)
```

Also update `goal_dist_max` default if `_GOAL_DIST_MAX` was changed in training:

```python
goal_dist_max_arg = DeclareLaunchArgument(
    'goal_dist_max',
    default_value='12.0',   # ← must match _GOAL_DIST_MAX in env_cfg.py
    ...
)
```

---

## 4. Deploying in Ignition Gazebo

### 4.1 Launch

```bash
# Terminal 1 — Gazebo + RL node
ros2 launch bcr_bot rl_test.launch.py

# Optional: pass a different model at launch time
ros2 launch bcr_bot rl_test.launch.py \
  model_path:=/path/to/exported/policy.pt \
  goal_dist_max:=12.0

# Terminal 2 — send a navigation goal
python3 ~/bcr_botrl1/bcr_botrl/bcr_botrl/scripts/goal_sender.py
```

### 4.2 Startup sequence

```
t=0 s   Ignition Gazebo starts, robot spawns
t=5 s   rl_policy_node.py starts (LiDAR bridge ready, no model load delay)
        Wait for /goal_pose from goal_sender.py before control loop runs
```

### 4.3 Key ROS topics

| Topic | Type | Direction |
|-------|------|-----------|
| `/bcr_bot/scan` | `LaserScan` | → node (obstacle sensing) |
| `/bcr_bot/odom` | `Odometry` | → node (position / heading) |
| `/goal_pose` | `PoseStamped` | → node (navigation target) |
| `/bcr_bot/cmd_vel` | `Twist` | ← node (wheel commands) |

### 4.4 LiDAR ray sampling

The BCR bot LiDAR (`/bcr_bot/scan`) covers 360°. The node samples 17 rays at:

```
−90°, −78.75°, −67.5°, ..., 0° (front), ..., +78.75°, +90°
step = 11.25°    index 0 = right,  index 8 = front,  index 16 = left
```

ROS `LaserScan` convention: `angle_min` is the most-clockwise ray; increasing
index → CCW (positive = left). The node does:

```python
idx = round((target_rad - msg.angle_min) / msg.angle_increment)
sectors[i] = min(r / LIDAR_MAX_DIST, 1.0)   # LIDAR_MAX_DIST = 5.0 m
```

### 4.5 VAPF — Virtual Attractive-Potential Field

The node reconstructs the same VAPF the policy was trained with:

```
Attractive:  F_att = k_att × [cos(θ), sin(θ)] × tanh(dist)
             k_att = 1.2
Repulsive:   F_rep = k_rep × (1/d − 1/d₀) / d²  (for each ray d < d₀)
             k_rep = 1.5,  d₀ = 2.0 m
Vortex:      F_vort = k_vortex × (1/d − 1/d₀) × [+sin(φ), −cos(φ)]  (CW around left obstacles)
             k_vortex = 1.5
All capped:  tanh(·/5) × 5  before entering obs vector
```

---

## 5. Reading the Logs

### 5.1 Per-step diagnostic (every 500 steps)

The node logs a block like:

```
[step 1500]  robot=(+2.31, −1.05)  yaw=+0.87 rad  goal=(+8.00, +0.00)
  dist=5.71 m  heading_err=−0.41 rad (−23.5°)  norm_dist=0.476
  lidar  : [5.00 5.00 5.00 5.00 4.82 3.10 1.13 5.00 5.00 5.00 5.00 5.00 5.00 5.00 5.00 5.00 5.00]
  min=1.13m@33.8°  front=5.00m  mean=4.65m
  vapf   : Fx=+0.872  Fy=+0.341  mag=0.940
  policy : u_l=+0.4512  u_r=+0.3881  raw→ lin=+0.042  ang=+0.006
  cmd_vel: linear=+0.400  angular=+0.108
```

| Field | What it tells you |
|-------|------------------|
| `heading_err` | Negative = goal is to the right; positive = goal is to the left |
| `lidar[8]` | Front clearance (index 8 = 0°) |
| `min=Xm@Y°` | Closest obstacle and its bearing |
| `vapf Fx/Fy` | VAPF vector in robot frame: Fx>0 → forward, Fy>0 → left |
| `u_l / u_r` | Raw network output (pre-clamp) |
| `cmd_vel` | What actually goes to the robot (clamped to max_linear/max_angular) |

### 5.2 Stuck detection log

```
[WARN] [STUCK] Only moved 0.03m in 5.0s — rotating CW for 1.5 s to escape saddle point.
[INFO] [STUCK] Recovery complete — resuming policy.
```

Stuck detection triggers when the robot moves <0.15 m in any 5 s window.
Recovery: publish `angular.z = −max_angular` for 1.5 s (75 steps at 50 Hz).

### 5.3 TensorBoard (training)

```bash
tensorboard --logdir ~/bcr_botrl1/bcr_botrl/bcr_botrl/logs/rsl_rl/bcr_navigation
```

Open `http://localhost:6006`. Useful scalars: `Train/mean_reward`,
`Train/goal_success`, `Train/mean_episode_length`, `Train/mean_noise_std`.

---

## 6. Diagnosing Problems

### 6.1 Robot does not move at all

**Symptom**: `u_l=+0.00, u_r=+0.00` at every step.

**Checklist**:
1. Is `goal_dist_max` in the launch file matching `_GOAL_DIST_MAX` from training?
   - Check: `norm_dist` in logs should be <1.0 for a reachable goal.
   - Fix: set `goal_dist_max` to the value used during training (current: 12.0).
2. Is the model loaded? Look for `model loaded` in startup logs.
3. Is scan data arriving? Look for `[SCAN] First scan received`.

### 6.2 Robot moves but goes wrong direction / misses goal

**Symptom**: `heading_err` stays large while robot drives forward.

**Root cause A — training defect (heading weight too weak)**:
- At `heading_err=0°`, policy outputs hard right turn (VAPF pointing forward = no
  lateral component; policy has no heading gradient to correct with).
- Fix: increase `heading_align` weight, reduce `yaw_rate_penalty` (see §7).

**Root cause B — wrong `heading_err` sign convention**:
- `heading_err = yaw_error_in_robot_frame` should be positive when goal is to
  the robot's left (positive Fy in robot frame).
- Verify in logs: if `heading_err < 0` and goal is visually to the left, there
  is a sign flip in `goal_obs()` in `bcr_navigation_env_cfg.py`.

**Root cause C — LiDAR angle convention mismatch**:
- The node assumes `angle_min` is the most-clockwise ray and uses
  `idx = round((target_rad − angle_min) / angle_increment)`.
- Verify: `lidar[8]` (index 8, 0°) should report front clearance matching what
  you see in Gazebo.

### 6.3 Robot gets stuck in open space (VAPF saddle point)

**Symptom**: `vapf mag ≈ 1.42`, `u_l ≈ 0, u_r ≈ 0`, robot stationary.

**Explanation**: The policy was trained on VAPF and can produce near-zero output
at VAPF zero-crossings when heading_align is too weak to break the tie.

**Fix in deployment**: Stuck detection (built into `rl_policy_node.py`) rotates
the robot CW for 1.5 s to escape the saddle, then resumes the policy.

**Fix in training**: Increase `heading_align` weight so the policy develops a
genuine heading-correction response that doesn't depend solely on VAPF.

### 6.4 Robot spins in place

**Symptom**: `angular` always near max, `linear` near zero.

**Cause**: `heading_align` weight is too high relative to `goal_approach`, making
the policy prefer rotating to face the goal rather than driving.

**Fix**: Reduce `heading_align` or increase `goal_approach` weight.

---

## 7. Reward Tuning & Retraining

### 7.1 Workflow

```
Edit bcr_navigation_env_cfg.py (rewards)
    ↓
Run training (§2.2)
    ↓
Monitor TensorBoard — pick best checkpoint
    ↓
Export policy (§3)
    ↓
Update _DEFAULT_MODEL in rl_test.launch.py
    ↓
Deploy & test in Gazebo (§4)
    ↓
Read logs (§5) — diagnose (§6) — iterate
```

### 7.2 Current reward weights (post-fix)

```python
# bcr_navigation_env_cfg.py — BcrRewardsCfg
goal_approach    weight=5.0    params={"gain": 3.0}     # dense, dominates
goal_reached     weight=10.0                             # sparse terminal
heading_align    weight=0.3                              # ← was 0.01 (too weak)
stationary       weight=-0.05
action_rate      weight=-0.0001
yaw_rate_penalty weight=-0.001                           # ← was -0.01 (cancelled heading)
```

**Rule**: `abs(yaw_rate_penalty) < heading_align / 10`  
i.e. the penalty for turning must be much smaller than the reward for facing the goal.

### 7.3 Common tuning levers

| Symptom | Adjustment |
|---------|-----------|
| Robot ignores heading | Increase `heading_align` weight |
| Robot spins to align before moving | Decrease `heading_align` or increase `goal_approach` |
| Robot stops short of goal | Increase `goal_reached` weight or reduce `goal_reached_radius` |
| Policy value loss explodes (>4) | Reduce all reward weights by 5–10× |
| `mean_noise_std` doesn't decay | Increase `entropy_coef` or reduce `clip_param` |
| Policy peaks then regresses | Reduce `learning_rate`; lower `clip_param` to 0.1 |
| Only 1–2% goal success after 1000 iters | Check `gamma` (use 0.999 for 30 s episodes) |

### 7.4 Checking obs/reward consistency between Isaac Lab and Gazebo

Constants that **must match** between `bcr_navigation_env_cfg.py` and
`rl_policy_node.py`:

| Constant | Training value | Deployment param |
|----------|----------------|-----------------|
| `_GOAL_DIST_MAX` | 12.0 | `goal_dist_max` ROS param |
| `LIDAR_MAX_DIST` | 5.0 m | `LIDAR_MAX_DIST` in node |
| `_N_LIDAR_RAYS` | 16 (→ 17 rays) | `NUM_LIDAR_RAYS = 17` in node |
| `_K_ATT / _K_REP / _D_0 / _K_VORTEX` | 1.2 / 1.5 / 2.0 / 1.5 | same in `_compute_vapf()` |
| `_VAPF_CAP` | 5.0 | `_VAPF_CAP = 5.0` in node |
| Ray angles | −90° to +90°, step 11.25° | `_TARGET_ANGLES_DEG` in node |

Any mismatch silently corrupts the observation and causes erratic behaviour.

---

## 8. Known Issues & Fix History

### Training instability fixes (PPO)

| Iteration | Problem | Fix |
|-----------|---------|-----|
| Run 1 | Value loss spiked to 4–6, `mean_noise_std` → 39 | Reduced reward weights 100× |
| Run 2 | Policy collapsed, `mean_std → 0.06`, 1% goal success | Widened gap: `goal_approach` weight=5 gain=3 |
| Run 3 | 2.6% success after 3000 iters | `gamma` 0.99→0.999; `num_steps_per_env` 24→128 |
| Run 4 | Peak 60% at iter 1000, decayed to 23% by iter 3000 | LR 3e-4→1e-4; `clip_param` 0.2→0.15; `entropy_coef` 0.002→0.005 |
| Run 5 | Orientation wrong in Gazebo, robot misses goal | Found: `heading_align=0.01` cancelled by `yaw_rate_penalty=−0.01` → fix applied |

### Deployment fixes (Gazebo)

| Issue | Root cause | Fix |
|-------|-----------|-----|
| Robot stuck at VAPF saddle | VAPF ≈ 1.42, policy zero-crosses, no heading gradient | Stuck detection: CW recovery rotation for 1.5 s |
| Near-zero outputs on new model | `goal_dist_max=5.0` in launch but training used 12.0 | Updated launch default to `12.0` |
| Wrong obstacle sensing FOV | Camera 60° vs policy trained on 180° LiDAR | Switched to `/bcr_bot/scan` (LiDAR) — dropped `depth_anything_node` |
| Floor detected as obstacle | Camera angle, DA depth picking up floor at 1.70 m | (Obsolete — depth_anything_node no longer used) |

---

## 9. Quick-Reference Constants

```
Wheel radius R         = 0.1 m
Wheel track L          = 0.6 m
Max linear speed       = 0.4 m/s
Max angular speed      = 1.2 rad/s
Control rate           = 50 Hz
Episode length         = 30 s  (750 steps)
Goal reached radius    = 0.30 m (training) / 0.30 m (Gazebo)
Goal dist max          = 12.0 m
LiDAR max dist         = 5.0 m
LiDAR rays             = 17  (−90° to +90°, 11.25° step)
Sim dt                 = 0.01 s  (decimation=2 → policy dt=0.02 s)
Num envs (training)    = 2048
Stuck window           = 5.0 s
Stuck min travel       = 0.15 m
Recovery duration      = 1.5 s (CW rotation)
```

### File map

| Purpose | File |
|---------|------|
| Env / reward config | `bcr_botrl/source/bcr_botrl/bcr_botrl/tasks/manager_based/bcr_botrl/bcr_navigation_env_cfg.py` |
| PPO hyperparameters | `bcr_botrl/source/bcr_botrl/bcr_botrl/tasks/manager_based/bcr_botrl/agents/rsl_rl_ppo_cfg.py` |
| Training script | `bcr_botrl/scripts/rsl_rl/train.py` |
| Export script | `bcr_botrl/scripts/rsl_rl/export.py` |
| Deployed RL node | `bcr_botrl/scripts/rl_policy_node.py` |
| Gazebo launch | `~/nav2_ws/src/bcr_bot/launch/rl_test.launch.py` |
| Goal sender | `bcr_botrl/scripts/goal_sender.py` |
| Training logs | `bcr_botrl/logs/rsl_rl/bcr_navigation/YYYY-MM-DD_HH-MM-SS/` |
| Exported policy | `…/YYYY-MM-DD_HH-MM-SS/exported/policy.pt` |

