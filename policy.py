import torch.nn as nn

class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(in_features=2, out_features=32, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=4, bias=True), # outputs 1 logit per direction
        )

    def forward(self, state):
        # state is a 1x2 tensor
        generated_action = self.policy(state)
        return generated_action # 1x4 tensor
