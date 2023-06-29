import torch
print(torch.__version__)

x = torch.rand(5, 3)
print(x)

print( torch.cuda.is_available() )
print( torch.cuda.device_count())