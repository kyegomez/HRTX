import torch
from torch import nn
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
        num_tokens: int,
        seq_len: int,
        dim_head: int,
        num_robots: int,
    ):
        super(EarlyExitTransformer, self).__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.max_tokens = num_tokens
        self.max_seq_len = seq_len
        self.dim_head = dim_head
        self.num_robots = num_robots

        self.transformer = Transformer(
            dim,
            dim_head=dim,
            heads=heads,
            depth=depth,
        )

        # Output head
        self.output_head = OutputHead(dim, -1)

        # Token embeddings
        self.embed = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        """
        Forward pass of the EarlyExitTransformer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        x = self.embed(x)

        for _ in range(self.depth):
            x = self.transformer(x)
            print(x)
            x = self.output_head(x)
            print(x)
            # Add to outputs list
            return x

        # return self.output_head(x)


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
