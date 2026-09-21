import torch

class Linear(nn.Module): 

    def __init__(self, in_features: int, out_features:int, device: torch.device=None, dtype: torch.dtype=None):
        """
        Construct a linear transformation module. This function should accept the following parameters:
        in_features: int  final dimension of the input
        out_features: int  final dimension of the output
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        """
        super.__init__
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        pass 


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass