import torch
import torch.nn as nn
import math
import self_attention
from config import d_model as configured_d_model
from config import num_heads as configured_num_heads

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = configured_d_model, num_heads: int = configured_num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        # Define linear layers for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Define the final linear layer
        self.W_O = nn.Linear(d_model, d_model)

        # Scaled Dot-Product Attention
        self.attention = self_attention.ScaledDotProductAttention(d_model, num_heads)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        batch_size = Q.size(0)

        # Linear projections
        # Q_proj = self.W_Q(Q) #(32, 10, 512) (batch_size, seq_len, d_model) 
        # Q_proj = Q_proj.view(batch_size, -1, self.num_heads, self.d_k) # (batch_size, seq_len, num_heads=8, d_k)
        # Q_proj = Q_proj.transpose(1, 2) # (batch_size, num_heads, seq_len, d_k) 
        Q_proj = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) # -1 is Pytorch's way of inferring the sequence length dimension automatically based on the batch size and d_model.
        K_proj = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V_proj = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply Scaled Dot-Product Attention
        attention_output, attention_weights = self.attention(Q_proj, K_proj, V_proj) #here attention_weights is passed here because it is needed for visualization and analysis of the attention mechanism, allowing us to see which parts of the input sequence the model is focusing on when making predictions.

        # Concatenate heads and apply final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) #here, we first transpose the attention_output tensor to bring the num_heads dimension back to its original position. Then, we use contiguous() to ensure that the tensor is stored in a contiguous block of memory, which is necessary for the subsequent view operation. Finally, we reshape the tensor to have the shape (batch_size, seq_len, d_model) by combining the num_heads and d_k dimensions back into a single d_model dimension.
        output = self.W_O(attention_output)