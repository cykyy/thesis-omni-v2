import torch

if torch.cuda.is_available():
    print("CUDA is available")
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print("Name:", torch.cuda.get_device_name(i))
        print("Total memory (GB):", torch.cuda.get_device_properties(i).total_memory / 1024**3)
        print("Allocated (GB):", torch.cuda.memory_allocated(i) / 1024**3)
        print("Reserved (GB):", torch.cuda.memory_reserved(i) / 1024**3)
else:
    print("No CUDA GPU available")
