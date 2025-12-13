# Hatice – reward & environment implementation

import numpy as np

class RewardConfig:
    # weights
    w_safety = 1.0
    w_eff    = 0.01
    w_dist   = 1.0
    w_jerk   = 0.02

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
    d_min = float(info.get("min_obstacle_dist", 1e6))
    dist = float(info.get("dist_to_target", 0.0))

    # Distance progress
    dist_prev = float(prev.get("dist_to_target_prev", dist))
    progress = dist_prev - dist
    r_dist = float(np.clip(progress, -rcfg.prog_clip, rcfg.prog_clip))

    # Safety baseline (always positive)
    r_safety = 1.0

    # Danger penalty (separate penalty term)
    p_danger = 0.0
    if d_min < rcfg.d_safe:
        danger = np.exp(-(d_min / max(rcfg.d_safe, 1e-6)))  # 0..1-ish
        p_danger = float(rcfg.danger_penalty * danger)

    # Efficiency: small actions + jerk
    a = np.asarray(action, dtype=np.float32)
    r_eff = - float(np.dot(a, a))

    a_prev = np.asarray(prev.get("action_prev", np.zeros_like(a)), dtype=np.float32)
    jerk = a - a_prev
    p_jerk = float(np.dot(jerk, jerk))

    # Collision (early exit)
    if collision:
        info["reward_terms"] = {
            "collision": 1,
            "r_safety": r_safety,
            "p_danger": p_danger,
            "r_dist": r_dist,
            "r_eff": r_eff,
            "p_jerk": p_jerk,
            "bonus": 0.0,
            "total": -float(rcfg.collision_penalty),
        }
        return -float(rcfg.collision_penalty), True

    # Goal bonus
    goal = dist < rcfg.goal_eps
    bonus = float(rcfg.goal_bonus) if goal else 0.0

    total = (
        rcfg.w_safety * r_safety +
        rcfg.w_dist   * r_dist +
        rcfg.w_eff    * r_eff
        - rcfg.w_jerk  * p_jerk
        - p_danger
        + bonus
    )
    done = bool(goal)

    info["reward_terms"] = {
        "collision": 0,
        "r_safety": r_safety,
        "p_danger": p_danger,
        "r_dist": r_dist,
        "r_eff": r_eff,
        "p_jerk": p_jerk,
        "bonus": bonus,
        "total": float(total),
    }

    return float(total), done

# Hatice - push check
