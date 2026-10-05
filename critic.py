import torch.nn as nn

class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(in_features=2, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, state):
        value = self.critic(state)
        return value # 1x1 tensor