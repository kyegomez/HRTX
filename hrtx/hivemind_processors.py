import torch
from torch import nn


class DynamicInputChannels(nn.Module):
    def __init__(self, num_robots, input_dim, output_dim):
        super(DynamicInputChannels, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for _ in range(num_robots)]
        )

    def forward(self, x):
        outputs = [layer(x) for layer in self.layers]
        return torch.cat(outputs, dim=1)


class OutputDecoders(nn.Module):
    def __init__(self, num_robots, input_dim, output_dim):
        super(OutputDecoders, self).__init__()
        self.decoders = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for _ in range(num_robots)]
        )

    def forward(self, x):
        return torch.stack([decoder(x) for decoder in self.decoders], dim=1)
