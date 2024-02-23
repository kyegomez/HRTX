import torch
from hrtx.mimmo import MIMMO


# Usage of the MIMMO module
x = [torch.randint(0, 1000, (1, 10)) for _ in range(3)]


# Create the model
model = MIMMO(
    dim=512,
    depth=6,
    num_tokens=1000,
    seq_len=10,
    heads=8,
    dim_head=64,
    num_robots=3,
)

# Forward pass
output = model(x)

# Print the output shape
print(output[0].shape)
