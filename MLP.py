class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU() #activation function that is used in the feedforward network of the transformer architecture. It is a smooth, non-linear function that helps introduce non-linearity into the model, allowing it to learn complex patterns in the data. The GELU activation function is defined as GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))). It has been shown to perform better than other activation functions like ReLU in certain scenarios, particularly in transformer models. 
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x