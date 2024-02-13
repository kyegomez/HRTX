from torch import nn, Tensor
from hrtx.transformer import Transformer
from hrtx.main import (
    OutputHead,
    MultiInputMultiModalConcatenation,
    SplitMultiOutput,
)
from typing import List


class SAETransformer(nn.Module):
    """
    EarlyExitTransformer is a PyTorch module that implements an early-exit transformer network.

    Args:
        dim (int): The input dimension.
        depth (int): The number of transformer layers.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mlp_dim (int): The dimension of the feed-forward network.
        num_robots (int): The number of robots.

    Attributes:
        dim (int): The input dimension.
        depth (int): The number of transformer layers.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mlp_dim (int): The dimension of the feed-forward network.
        num_robots (int): The number of robots.
        transformer (Transformer): The transformer network.
        output_head (OutputHead): The output head.

    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        num_robots: int,
        *args,
        **kwargs,
    ):
        super(SAETransformer, self).__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.num_robots = num_robots

        # Transformer network
        self.transformer = Transformer(
            dim,
            dim_head=dim,
            heads=heads,
            depth=depth,
            *args,
            **kwargs,
        )

        # Output head
        self.output_head = OutputHead(dim, -1)

        # Multi Input Multi Modal Concatenation
        self.multi_input = MultiInputMultiModalConcatenation(dim=1)

        # SplitMultiOutput
        self.split_output = SplitMultiOutput(
            dim=1,
            num_splits=num_robots,
            output_dims=3,
            *args,
            **kwargs,
        )

    def forward(self, x: List[Tensor]):
        """
        Forward pass of the EarlyExitTransformer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        x = self.multi_input(x)
        print(x.shape)

        for _ in range(self.num_robots):
            x = self.transformer(x)
            print(x)
            x = self.output_head(x)
            print(x)

        split = self.output_head(x)

        return split


# # Input tensor
# x = torch.randn(1, 10, 512)
# x = [x, x, x]

# # Create the model
# model = SAETransformer(dim=512, depth=6, heads=8, dim_head=64, num_robots=3)


# # Forward pass
# output = model(x)

# # Print the output shape
# print(output)

# # Output: torch.Size([1, 10, 512])
