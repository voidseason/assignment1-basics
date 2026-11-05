import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

class Linear(nn.Module):
    def __init__(self,
    in_features:int,
    out_features:int,
    device:torch.device|None = None,
    dtype:torch.dtype|None = None
    ):
        super.__init__()
        self.W = nn.Parameter(
                torch.empty(in_features, out_features, device=device, dtype=dtype)
            )
        # 按照3.3.1 Parameter Initialization的要求来初始化
        std = math.sqrt(2.0 / (in_features + out_features))
        trunc_normal_(self.W,mean=0.0,std=std,a=-3*std,b=3*std)
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return x @ self.W

class RNSNorm(nn.Module):
    def __init__(self,
    d_model:int,
    eps:float = 1e-5,
    device:torch.device|None = None,
    dtype:torch.dtype|None = None
    ):
        super().__init__()
        self.eps = eps
        # 按照3.3.1 Parameter Initialization的要求来初始化
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self,x:torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.weight
        return result.to(in_dtype)

