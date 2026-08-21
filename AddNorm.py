import torch
import torch.nn as nn
import math

class AddNormalization(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        # Apply residual connection followed by Layer Normalization
        return self.layer_norm(x + sublayer_output)
    