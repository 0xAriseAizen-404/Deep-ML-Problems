import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
    def forward(self, x):
        return self.mlp(x)

def count_params():
    model = MyModel()
    # return sum(p.nelement() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
   
