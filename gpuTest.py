import torch
print("PyTorch version:", torch.__version__)
print("CUDA version built with PyTorch:", torch.version.cuda)

print(torch.cuda.is_available())