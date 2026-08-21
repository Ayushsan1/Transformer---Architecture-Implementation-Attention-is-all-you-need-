#Self Attention in Transformers
import torch
import torch.nn as nn
import math
from config import d_model as configured_d_model
from config import num_heads as configured_num_heads

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int = configured_d_model, num_heads: int = configured_num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        # Q, K, V shapes: (batch_size, num_heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # Scaled dot-product
        if mask is not None:
            scores = scores.masked-fill(mask == 0, float('-inf'))
        attention_weights = self.softmax(scores)  # Softmax to get attention weights
        output = torch.matmul(attention_weights, V)  # Weighted sum of values
        return output, attention_weights
    

