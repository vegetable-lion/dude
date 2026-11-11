import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 768)
        )


    def forward(self, x):
        x = self.mlp(x)
        return x

# Example usage
# model = CNNCodeBook()
# print(model)

# # Create a dummy input with shape (batch_size=1, channels=1000, height=16, width=16)
# dummy_input = torch.randn(1, 1280, 8, 8)
# output = model.get_cnnout(dummy_input)
# print(f"Output shape: {output.shape}")
