## RL Agent
#The PPO-based agent uses an actor-critic architecture with continuous action space.
#The agent is designed independently from the simulation for modularity.

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
