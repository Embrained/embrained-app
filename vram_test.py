import torch
import sys

def test_vram():
    print("-" * 60)
    print("CUDA VRAM STRESS TEST")
    print("-" * 60)

    # 1. Check for CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Check your PyTorch installation.")
        return

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    total_phys_vram = props.total_memory / (1024 ** 3)  # Convert to GB

    print(f"Device: {props.name}")
    print(f"Total Physical VRAM: {total_phys_vram:.2f} GB")
    print("Starting allocation test...\n")

    allocated_tensors = []
    chunk_size_mb = 100
    # Create a dummy tensor of roughly 100MB (float32 = 4 bytes)
    # 100 MB = 100 * 1024 * 1024 bytes
    # Elements needed = (100 * 1024 * 1024) / 4
    num_elements = int((chunk_size_mb * 1024 * 1024) / 4)
    
    try:
        while True:
            # Allocate memory on GPU
            t = torch.ones(num_elements, dtype=torch.float32, device=device)
            allocated_tensors.append(t)
            
            # Calculate current usage
            current_alloc_gb = (len(allocated_tensors) * chunk_size_mb) / 1024
            
            # Print status on same line to keep console clean
            sys.stdout.write(f"\rAllocated: {current_alloc_gb:.2f} GB")
            sys.stdout.flush()

    except RuntimeError as e:
        print("\n\n--- HIT MEMORY LIMIT ---")
        if "out of memory" in str(e):
            print("Successfully triggered OutOfMemoryError.")
        else:
            print(f"Stopped due to unexpected error: {e}")

    # Final stats
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    print("-" * 60)
    print(f"MAX USABLE VRAM:   {max_allocated:.2f} GB")
    print(f"TOTAL RESERVED:    {max_reserved:.2f} GB")
    print(f"SYSTEM OVERHEAD:   {total_phys_vram - max_reserved:.2f} GB (Windows UI/Driver)")
    print("-" * 60)
    
    # Cleanup
    allocated_tensors = []
    torch.cuda.empty_cache()
    print("Memory cleared.")

if __name__ == "__main__":
    test_vram()