import torch
from hrtx.ee_network import EarlyExitTransformer

# Input tensor
x = torch.randn(1, 10, 512)

# Create the model
model = EarlyExitTransformer(
    dim=512, depth=6, heads=8, dim_head=64, mlp_dim=2048, num_robots=3
)


# Forward pass
output = model(x)

# Print the output shape
print(output.shape)

# Output: torch.Size([1, 10, 512])
