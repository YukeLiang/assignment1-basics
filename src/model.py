import math
import torch
import torch.nn as nn

class Linear(nn.Module): 
    def __init__(self, d_in: int, d_out:int, device: torch.device=None, dtype: torch.dtype=None):
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
        w = torch.empty(d_out, d_in, dtype = dtype, device = device)
        # 1. Calculate the base target variance (Xavier criterion)
        target_variance = 2.0 / (d_out + d_in)
        # 2. Convert variance to standard deviation
        std = math.sqrt(target_variance)
        torch.nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std, generator=None)

        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('... i, ji ->... j', x, self.weight)



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


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
            Construct the RMSNorm module. This function should accept the following parameters:
            d_model: int  Hidden dimension of the model
            eps: float = 1e-5  Epsilon value for numerical stability
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        self.eps = eps
        # init wight is all 1s
        self.weight = nn.Parameter(torch.ones(d_model, device = device, dtype = dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + 1e-5)
        result = x / rms * self.weight
        return result.to(in_dtype)

class SwiGLU(nn.Module): 
    # This is position wise feed forward networks 
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        
        self.w_si = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w_glu = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w_down = Linear(d_ff, d_model, device=device, dtype=dtype)
        
    # Feed forward layer 
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x1 = self.w_si(x)
        x2 = self.w_glu(x)
        # siLu and GLU
        up = x1 * torch.sigmoid(x1) * x2
        # down projection
        return self.w_down(up)

if __name__ == "__main__":
    # weight = torch.arange(20).reshape(4, 5)  # shape (4, 5)
    # idx = torch.tensor([3,])
    # print(weight[idx].shape)

    # weight = torch.arange(20).reshape(4, 5)   # vocab_size=4, d_model=5
    # idx = torch.tensor([[2, 0, 3], [1, 1, 0]])  # shape (2, 3)
    # idx_t = idx.T
    # print(weight[idx].shape) # 2,3,5
    # print(weight[idx_t].shape) # 3,2,5

    # x = torch.tensor([[2, 0, 3], [1, 1, 0]])  # shape (2, 3)
    # x = x.to(torch.float32)
    # rms = torch.sqrt(torch.mean(x.pow(2), dim=-1) + 1e-5)
    # print(rms)

    norm = RMSNorm(d_model=4)
    x = torch.randn(2, 3, 4, requires_grad=True)
    
    out = norm(x)
    loss = out.sum()
    loss.backward()
    
    print(norm.weight.grad)   # should NOT be None
    print(x.grad)             # should also not be None