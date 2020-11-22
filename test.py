import torch

a= torch.randn((3,1))
print(a)
a= a.squeeze(-1)
print(a)