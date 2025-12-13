import numpy as np

from env.reward import (
    RewardConfig, DomainRandConfig,
    sample_domain_randomization, compute_reward
)

class HelicopterEnv:
    def __init__(self, sim, target_pos, action_dim=4, max_steps=1000, seed=None):
        self.sim = sim
        self.target_pos = np.array(target_pos, dtype=np.float32)

        self.action_dim = int(action_dim)
        self.max_steps = int(max_steps)

        # configs
        self.rcfg = RewardConfig()
        self.drcfg = DomainRandConfig()

        # rng
        self.rng = np.random.default_rng(seed)

        # prev state for reward
        self.prev = {
            "dist_to_target_prev": 0.0,
            "action_prev": np.zeros(self.action_dim, dtype=np.float32),
        }

        self.domain_params = None
        self.step_count = 0

        # --- Obstacles (placeholder) ---
        # Senaryo: örnek 2 engel
        # (İleride Merve simülasyondan gerçek engelleri buraya verecek)
        self.obstacles = [
            np.array([5.0, 0.0, 0.0], dtype=np.float32),
            np.array([7.0, 2.0, 0.0], dtype=np.float32),
        ]




    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0

        # 1) Domain randomization sample
        self.domain_params = sample_domain_randomization(self.rng, self.drcfg)

        # 2) Simülasyona uygula
        if hasattr(self.sim, "apply_domain_randomization"):
            self.sim.apply_domain_randomization(self.domain_params)

        # 3) obs
        obs = self._get_obs()

        # 4) prev değerleri
        self.prev["dist_to_target_prev"] = self._compute_dist_to_target()
        self.prev["action_prev"] = np.zeros(self.action_dim, dtype=np.float32)

        info = {"domain_params": self.domain_params}
        return obs, info  # gymnasium ise
        # return obs       # gym ise


    def step(self, action):
        self.step_count += 1

        self.sim.apply_action(action)
        self.sim.step()

        obs = self._get_obs()

        dist = self._compute_dist_to_target()
        min_d = self._compute_min_obstacle_dist()
        collision = self.sim.get_collision()

        info = {
            "dist_to_target": dist,
            "min_obstacle_dist": min_d,   # <-- ÖNEMLİ: bu isim
            "collision": collision,
            "domain_params": self.domain_params
        }

        reward, done_reward = compute_reward(obs, action, info, self.prev, self.rcfg)

        done_timeout = (self.step_count >= self.max_steps)
        terminated = bool(done_reward)
        truncated = bool(done_timeout)

        # prev update
        self.prev["dist_to_target_prev"] = dist
        self.prev["action_prev"] = np.array(action, dtype=np.float32)

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        pos = np.array(self.sim.get_position(), dtype=np.float32)
        if hasattr(self.sim, "get_velocity"):
            vel = np.array(self.sim.get_velocity(), dtype=np.float32)
        else:
            vel = np.zeros(3, dtype=np.float32)
        return np.concatenate([pos, vel], axis=0)


    def _compute_dist_to_target(self):
        pos = self.sim.get_position()
        pos = np.array(pos, dtype=np.float32)
        return float(np.linalg.norm(pos - self.target_pos))

    def _compute_min_obstacle_dist(self):
        pos = np.array(self.sim.get_position(), dtype=np.float32)

        if not hasattr(self, "obstacles") or len(self.obstacles) == 0:
            return 1e6

        return float(min(np.linalg.norm(pos - o) for o in self.obstacles))




