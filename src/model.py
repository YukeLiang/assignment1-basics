import math
import torch
import torch.nn as nn

class Linear(nn.Module): 

    def __init__(self, in_features: int, out_features:int, device: torch.device=None, dtype: torch.dtype=None):
        """
        Construct a linear transformation module. This function should accept the following parameters:
        in_features: int  final dimension of the input
        out_features: int  final dimension of the output
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()

        # weights initialization 
        # 𝒩︀(𝜇 = 0, 𝜎2 = 2 𝑑in+𝑑out) truncated at [−3𝜎, 3𝜎]
        w = torch.empty(out_features, in_features, dtype = dtype, device = device)
        # 1. Calculate the base target variance (Xavier criterion)
        target_variance = 2.0 / (out_features + in_features)
        # 2. Convert variance to standard deviation
        std = math.sqrt(target_variance)
        torch.nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std, generator=None)

        self.weights = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('... i, ji ->... j', x, self.weights)



class Embedding(nn.Module): 
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
        """ 
        Construct an embedding module. This function should accept the following parameters:
        num_embeddings: int  Size of the vocabulary
        embedding_dim: int  Dimension of the embedding vectors, i.e., 𝑑model
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # initialize weights 
        w = torch.empty(num_embeddings, embedding_dim, device = device, dtype = dtype)
        #  𝒩︀(𝜇 = 0, 𝜎2 = 1) truncated at [−3, 3]
        truncated_w = torch.nn.init.trunc_normal_(w, mean=0.0, std=1, a=-3.0, b=3.0, generator=None)
        self.weights = nn.Parameter(truncated_w)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        """
        token_ids here is in shape of (batch, sequence_len)
        """
        return self.weights[token_ids]

if __name__ == "__main__":
    # weight = torch.arange(20).reshape(4, 5)  # shape (4, 5)
    # idx = torch.tensor([3,])
    # print(weight[idx].shape)

    weight = torch.arange(20).reshape(4, 5)   # vocab_size=4, d_model=5
    idx = torch.tensor([[2, 0, 3], [1, 1, 0]])  # shape (2, 3)
    idx_t = idx.T
    print(weight[idx].shape) # 2,3,5
    print(weight[idx_t].shape) # 3,2,5