from torch import nn, Tensor
from hrtx.transformer import Transformer
from hrtx.main import (
    OutputHead,
    MultiInputMultiModalConcatenation,
    SplitMultiOutput,
)
from typing import List


class MIMMO(nn.Module):
    """
    MIMMO (Multi Input Multi Modal Output) module.

    Args:
        dim (int): Dimension of the input.
        depth (int): Number of transformer layers.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        mlp_dim (int): Dimension of the MLP layers.
        num_robots (int): Number of output splits for multi-robot scenarios.
        *args: Variable length arguments.
        **kwargs: Keyword arguments.

    Attributes:
        dim (int): Dimension of the input.
        depth (int): Number of transformer layers.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        mlp_dim (int): Dimension of the MLP layers.
        num_robots (int): Number of output splits for multi-robot scenarios.
        transformer (Transformer): Transformer model.
        output_head (OutputHead): Output head.
        multi_input (MultiInputMultiModalConcatenation): Multi Input Multi Modal Concatenation layer.
        c (nn.Linear): Reshaping connection layer.
        split_output (SplitMultiOutput): Splitting the output tensor layer.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        num_robots: int,
        num_tokens: int,
        seq_len: int,
        *args,
        **kwargs,
    ):
        super(MIMMO, self).__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.num_robots = num_robots
        self.num_tokens = num_tokens
        self.seq_len = seq_len

        # Transformer models
        self.transformer = Transformer(
            dim, dim_head, heads, depth, *args, **kwargs
        )

        # Output head
        self.output_head = OutputHead(dim, -1)

        # Multi Input Multi Modal Concatenation
        self.multi_input = MultiInputMultiModalConcatenation(dim=1)

        # Reshaping connection layers
        self.c = nn.Linear(dim, dim)

        # # Splitting the output tensor
        # self.split_output = SplitMultiOutput(
        #     dim=1,
        #     num_splits=self.num_robots,

        # )

        # Token embeddings
        self.embed = nn.Embedding(num_tokens, dim)

    def forward(self, x: List[Tensor]):
        """
        Forward pass of the MIMMO module.

        Args:
            x (List[Tensor]): List of input tensors.

        Returns:
            List[Tensor]: List of output tensors.
        """
        for i in range(len(x)):
            x[i] = self.embed(x[i])

        # b, s, d = x.shape
        x = self.multi_input(x)

        b, s, d = x.shape

        # Transformer forward pass
        for _ in range(self.depth):
            x = self.transformer(x)
            x = self.c(x)
            skip = x

        # Send the skip connection to the output head
        x = self.output_head(x) + skip

        # Split the output tensor
        # x = self.split_output(x)
        x = SplitMultiOutput(
            dim=1, num_splits=self.num_robots, output_dims=[s]
        )(x)

        return x


# # Usage of the MIMMO module
# x = [torch.randint(0, 1000, (1, 10)) for _ in range(3)]


# # Create the model
# model = MIMMO(
#     dim=512,
#     depth=6,
#     num_tokens=1000,
#     seq_len=10,
#     heads=8,
#     dim_head=64,
#     num_robots=3,
# )

# # Forward pass
# output = model(x)

# # Print the output shape
# print(output[0].shape)
