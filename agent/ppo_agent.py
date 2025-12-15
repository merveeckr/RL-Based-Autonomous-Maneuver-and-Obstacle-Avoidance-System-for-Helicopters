## RL Agent
#The PPO-based agent uses an actor-critic architecture with continuous action space.
#The agent is designed independently from the simulation for modularity.

# action_dim = 3
# Actions: [pitch_delta, roll_delta, throttle_delta]
# Yaw control is excluded in the initial phase to simplify learning


import torch
import torch.nn as nn
import numpy as np

class ActorCritic(nn.Module):
    """
    PPO için Actor-Critic mimarisi
    Actor  -> aksiyon üretir
    Critic -> state value tahmini yapar
    """

    def __init__(self, obs_dim, action_dim):
        super().__init__()

        # Ortak feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Policy (Actor)
        self.policy_head = nn.Linear(128, action_dim)

        # Value (Critic)
        self.value_head = nn.Linear(128, 1)

    def forward(self, obs):
        features = self.shared_layers(obs)
        action_logits = self.policy_head(features)
        state_value = self.value_head(features)

        return action_logits, state_value
        
class PPOAgent:
    """
    PPO ajanı: ActorCritic modelini kullanarak
    observation -> action dönüşümünü yapar
    """

    def __init__(self, obs_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Gün 1'deki modeli burada kullanıyoruz.
        self.model = ActorCritic(obs_dim, action_dim).to(self.device)

    def select_action(self, obs):
        """
        Verilen observation'dan continuous action üretir
        """
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        action_logits, state_value = self.model(obs)

        # PPO için continuous action -> [-1, 1]
        action = torch.tanh(action_logits)

        return action.detach().cpu().numpy(), state_value.item()

