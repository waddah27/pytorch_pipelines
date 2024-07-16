import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):
    """
    A single head attention layer.

    Args:
        embedding_dim (int): The dimensionality of the input embeddings.
        attention_dim (int): The dimensionality of the attention space.

    Attributes:
        Q (nn.Linear): The linear layer for the queries.
        K (nn.Linear): The linear layer for the keys.
        V (nn.Linear): The linear layer for the values.
    """

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.Q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.K = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.V = nn.Linear(embedding_dim, attention_dim, bias=False)


    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        """
        Forward pass of the attention layer.

        Args:
            embedded (TensorType[float]): The input embeddings.

        Returns:
            TensorType[float]: The attention weights.
        """

        # Return your answer to 4 decimal places
        q = self.Q(embedded)
        k = torch.transpose(self.K(embedded), 1,2)
        v = self.V(embedded)
        scores = q @ torch.transpose(k, 1,2) # torch.matmul(q,torch.transpose(k, 1,2)) = Q.K^T
        B, T, A = k.shape # batch_size, seq_len (or context_len), attention_dim (d_k)
        scores = scores/(A ** 0.5)

        # masking
        lower_triangle = torch.tril(torch.ones(T, T))
        mask = lower_triangle == 0
        scores = scores.masked_fill(mask, float('-inf'))

        # softmax
        scores = nn.functional.softmax(scores, dim=2) # (B, T, T)

        return scores @ v

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer.

    Args:
        embedding_dim (int): The dimensionality of the input embeddings.
        attention_dim (int): The dimensionality of the attention space.
        num_heads (int): The number of attention heads.
        testing_mode (bool, optional): Whether to set the seed for reproducibility. Defaults to False.
    """

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int, testing_mode: bool = False):
        super().__init__()
        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads"
        if testing_mode:
            torch.manual_seed(0)
        head_size = attention_dim // num_heads
        self.single_head_attention = SingleHeadAttention(embedding_dim, head_size)
        self.Heads = nn.ModuleList([self.single_head_attention for _ in range(num_heads)])

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        """
        Forward pass of the multi-head attention layer.

        Args:
            embedded (TensorType[float]): The input embeddings.

        Returns:
            TensorType[float]: The attention weights.
        """
        scores = []
        for i in range(len(self.Heads)):
            scores.append(self.Heads[i](embedded))

        scores = torch.cat(scores, dim=2)
        return scores


if __name__ == '__main__':
    B, T, E = 2, 2, 3 # batch_size, seq_len (or context_len), embedding_dim
    A = 4 # attention_dim
    n_heads = 2 # num_heads
    attention = MultiHeadAttention(E, A, n_heads, testing_mode=True)
    input = torch.randn(B, T, E)
    print(attention(input))