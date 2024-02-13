import torch
from hrtx.mimmo import MIMMO


# Load the model
model = MIMMO(512, 6, 8, 64, 3)

# Create a random input
x = torch.randn(2, 100, 512)
x = [x, x, x]

# Make a prediction
pred = model(x)

# Print the output
print(pred)
