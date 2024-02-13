from torch import nn, Tensor
from hrtx.transformer import Transformer
from hrtx.main import OutputHead


class EarlyExitTransformer(nn.Module):
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
        mlp_dim: int,
        num_robots: int,
    ):
        super(EarlyExitTransformer, self).__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.num_robots = num_robots

        self.transformer = Transformer(
            dim,
            dim_head=dim,
            heads=heads,
            depth=depth,
        )

        # Output head
        self.output_head = OutputHead(dim, -1)

    def forward(self, x: Tensor):
        """
        Forward pass of the EarlyExitTransformer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        for _ in range(self.depth):
            x = self.transformer(x)
            print(x)
            x = self.output_head(x)
            print(x)
            # Add to outputs list
            return x
        # return self.output_head(x)


# # Input tensor
# x = torch.randn(1, 10, 512)

# # Create the model
# model = EarlyExitTransformer(
#     dim=512, depth=6, heads=8, dim_head=64, mlp_dim=2048, num_robots=3
# )


# # Forward pass
# output = model(x)

# # Print the output shape
# print(output.shape)

# # Output: torch.Size([1, 10, 512])
