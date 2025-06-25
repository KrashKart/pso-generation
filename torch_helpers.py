import torch
from torch import Tensor

def computeEqCOTorch(temps: Tensor, H2: Tensor, CO: Tensor, H2O: Tensor, CO2: Tensor) -> Tensor:
    """
    Computes the EqCO for the reaction with torch tensors
    """

    exp_k = torch.exp(4577.8 / (temps + 273) - 4.33)
    k = torch.where(temps > -173, exp_k, 1e+18)
    
    a = (1 - k) * CO ** 2
    b = CO * (H2 + CO2 + k * (CO + H2O)) 
    c = H2 * CO2 - k * H2O * CO
    discriminants = torch.clamp(b **2 - 4 * a * c, min = 0.0)
    disc_sqrt = torch.sqrt(discriminants)
    root1 = (0.5 / a) * (-b - disc_sqrt)
    root2 = (0.5 / a) * (-b + disc_sqrt)
    roots = torch.max(root1, root2)
    roots_min = torch.min(torch.stack([root1, root2, torch.ones(root1.size(0))]), dim=0)[0]

    EqCO = torch.where(roots > 1, roots_min, roots) * 100
    return EqCO

def compareEqExpCO(eq: Tensor, exp: Tensor) -> int:
    deltaCO = eq - exp
    return (deltaCO < 0).sum().item()