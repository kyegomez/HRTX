import torch
from hrtx.ee_network import EarlyExitTransformer

# Input tensor - tokens int
token = torch.randint(0, 1000, (1, 10))

# Create the model
model = EarlyExitTransformer(
    dim=512,
    depth=6,
    num_tokens=1000,
    seq_len=10,
    heads=8,
    dim_head=64,
    num_robots=3,
)


# Forward pass
output = model(token)

# Print the output shape
print(output.shape)

# Output: torch.Size([1, 10, 512])
