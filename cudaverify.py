import torch
print(torch.__version__)
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Device Name (GPU 0): {torch.cuda.get_device_name(0)}")