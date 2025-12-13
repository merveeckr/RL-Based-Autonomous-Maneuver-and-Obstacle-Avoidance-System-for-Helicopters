# Hatice – reward & environment implementation

import numpy as np

class RewardConfig:
    # weights
    w_safety = 1.0
    w_eff    = 0.01
    w_dist   = 1.0
    w_jerk   = 0.02

    # NEW (bugün eklenenler)
    w_att    = 0.05     # roll/pitch smoothness penalty
    w_time   = 1.0      # time penalty weight
    time_penalty = 0.001  # her step için küçük ceza

    collision_penalty = 80.0
    danger_penalty    = 1.0
    goal_bonus        = 30.0

    d_safe = 3.0       # meters
    goal_eps = 2.0     # meters
    prog_clip = 2.0    # max progress reward per step

class DomainRandConfig:
    wind_speed = (0.0, 15.0)         # m/s
    wind_dir   = (0.0, 360.0)        # deg
    turbulence = (0.0, 1.0)          # 0..1

    imu_acc_sigma  = (0.01, 0.10)
    imu_gyro_sigma = (0.001, 0.02)

    pressure_offset = (-300.0, 300.0)  # "altitude equivalent" meters

def sample_domain_randomization(rng: np.random.Generator, cfg: DomainRandConfig):
    return {
        "wind_speed": float(rng.uniform(*cfg.wind_speed)),
        "wind_dir": float(rng.uniform(*cfg.wind_dir)),
        "turbulence": float(rng.uniform(*cfg.turbulence)),
        "imu_acc_sigma": float(rng.uniform(*cfg.imu_acc_sigma)),
        "imu_gyro_sigma": float(rng.uniform(*cfg.imu_gyro_sigma)),
        "pressure_offset": float(rng.uniform(*cfg.pressure_offset)),
    }

def compute_reward(obs, action, info, prev, rcfg: RewardConfig):
    collision = bool(info.get("collision", False))
    dist = float(info.get("dist_to_target", 0.0))
    d_min = float(info.get("min_obstacle_dist", 1e6))

    # 1) Distance progress
    dist_prev = float(prev.get("dist_to_target_prev", dist))
    progress = dist_prev - dist
    r_dist = float(np.clip(progress, -rcfg.prog_clip, rcfg.prog_clip))

    # 2) Safety shaping (örnek: d_min < d_safe ise penalty)
    p_danger = 0.0
    if d_min < rcfg.d_safe:
        p_danger = float(np.exp(-(d_min / max(rcfg.d_safe, 1e-6))))
    r_safety = 1.0 - rcfg.danger_penalty * p_danger

    # 3) Efficiency: action magnitude
    a = np.asarray(action, dtype=np.float32)
    r_eff = -float(np.dot(a, a))

    # 4) Smoothness: action jerk
    a_prev = np.asarray(prev.get("action_prev", np.zeros_like(a)), dtype=np.float32)
    jerk = a - a_prev
    p_jerk = float(np.dot(jerk, jerk))

    # 5) Smoothness: attitude rate (opsiyonel ama öneririm)
    # info roll/pitch veriyorsa:
    roll = info.get("roll", None)
    pitch = info.get("pitch", None)
    roll_prev = prev.get("roll_prev", roll)
    pitch_prev = prev.get("pitch_prev", pitch)
    p_att = 0.0
    if roll is not None and pitch is not None and roll_prev is not None and pitch_prev is not None:
        droll = float(roll) - float(roll_prev)
        dpitch = float(pitch) - float(pitch_prev)
        p_att = droll*droll + dpitch*dpitch

    # 6) Time penalty
    p_time = rcfg.time_penalty  # örn 0.001 gibi sabit

    # Collision hard terminal
    if collision:
        terms = {"collision": 1, "total": -rcfg.collision_penalty}
        return -rcfg.collision_penalty, True, terms

    goal = dist < rcfg.goal_eps
    bonus = rcfg.goal_bonus if goal else 0.0

    total = (
        rcfg.w_safety * r_safety +
        rcfg.w_dist   * r_dist +
        rcfg.w_eff    * r_eff
        - rcfg.w_jerk * p_jerk
        - rcfg.w_att  * p_att
        - rcfg.w_time * p_time
        + bonus
    )

    terms = {
        "collision": 0,
        "r_safety": r_safety,
        "p_danger": p_danger,
        "r_dist": r_dist,
        "r_eff": r_eff,
        "p_jerk": p_jerk,
        "p_att": p_att,
        "p_time": p_time,
        "bonus": bonus,
        "total": float(total)
    }
    return float(total), bool(goal), terms

# Hatice - push check
