import torch
from hrtx.sae_transformer import SAETransformer

# Input tensor
x = torch.randn(1, 10, 512)
x = [x, x, x]

# Create the model
model = SAETransformer(
    dim=512, depth=6, heads=8, dim_head=64, num_robots=3
)


# Forward pass
output = model(x)

# Print the output shape
print(output)

# Output: torch.Size([1, 10, 512])
