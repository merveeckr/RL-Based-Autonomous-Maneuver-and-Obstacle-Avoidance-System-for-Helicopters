import numpy as np

from simulation.airsim_interface import AirSimInterface
from env.helicopter_env import HelicopterEnv
from reward_plot import plot_reward_terms


def main():
    sim = AirSimInterface()
    env = HelicopterEnv(sim, target_pos=[10, 0, 0], action_dim=4, max_steps=50, seed=42)

    obs, info = env.reset()
    terms_hist = []

    print("RESET")
    print("  obs:", obs)
    print("  info(domain):", info.get("domain_params"))

    for t in range(10):
        action = np.array([1, 0, 0, 0], dtype=np.float32)  # x yönüne it
        obs, reward, terminated, truncated, info = env.step(action)

        if "reward_terms" in info:
            terms_hist.append(info["reward_terms"])


        print(f"STEP {t}")
        print("  action:", action)
        print("  pos:", sim.get_position())
        print("  dist:", info['dist_to_target'])
        print("  min_d:", info['min_obstacle_dist'])
        print("  collision:", info['collision'])
        print("  reward:", reward, "| terminated:", terminated, "| truncated:", truncated)
        print("  reward terms:", info["reward_terms"])


        if terminated or truncated:
            print("Episode finished.")
            break

    if len(terms_hist) > 0:
        plot_reward_terms(terms_hist)
    else:
        print("No reward_terms found in info. Check env.step() and compute_reward().")

if __name__ == "__main__":
    main()
