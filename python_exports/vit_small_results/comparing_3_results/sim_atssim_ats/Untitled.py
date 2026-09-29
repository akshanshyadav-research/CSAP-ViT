# Exported from experiments/vit_small_results/comparing_3_results/sim_atssim_ats/Untitled.ipynb
# Historical code: review paths and side effects before executing.


# %% Original cell 0
import random
import time
import os

def generate_floating_point_file(filename="floating_points.txt", count=110_000_000):
    """
    Generate a text file with specified number of floating-point entries.
    
    Args:
        filename (str): Name of the output file
        count (int): Number of floating-point numbers to generate (default: 11 crores)
    """
    
    print(f"Generating {count:,} floating-point numbers...")
    print(f"Output file: {filename}")
    
    start_time = time.time()
    
    # Open file for writing
    with open(filename, 'w') as file:
        # Write numbers in batches for better performance
        batch_size = 10000
        batches = count // batch_size
        remaining = count % batch_size
        
        # Generate numbers in batches
        for batch in range(batches):
            # Generate batch of random floating-point numbers
            numbers = []
            for _ in range(batch_size):
                # Generate random float between -1000.0 and 1000.0
                num = random.uniform(-1000.0, 1000.0)
                numbers.append(f"{num:.6f}")
            
            # Write batch to file (one number per line)
            file.write('\n'.join(numbers) + '\n')
            
            # Progress indicator
            if (batch + 1) % 1000 == 0:
                progress = ((batch + 1) * batch_size / count) * 100
                print(f"Progress: {progress:.1f}% ({(batch + 1) * batch_size:,} numbers)")
        
        # Handle remaining numbers
        if remaining > 0:
            numbers = []
            for _ in range(remaining):
                num = random.uniform(-1000.0, 1000.0)
                numbers.append(f"{num:.6f}")
            file.write('\n'.join(numbers))
    
    end_time = time.time()
    
    # Get file size
    file_size = os.path.getsize(filename)
    file_size_mb = file_size / (1024 * 1024)
    file_size_gb = file_size_mb / 1024
    
    print(f"\nFile generation completed!")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"File size: {file_size_mb:.2f} MB ({file_size_gb:.3f} GB)")
    print(f"Numbers generated: {count:,}")
    print(f"Average numbers per second: {count / (end_time - start_time):,.0f}")

def generate_different_ranges():
    """
    Alternative function to generate numbers with different ranges/patterns
    """
    filename = "mixed_floating_points.txt"
    count = 110_000_000
    
    print(f"Generating {count:,} floating-point numbers with mixed ranges...")
    
    start_time = time.time()
    
    with open(filename, 'w') as file:
        batch_size = 10000
        batches = count // batch_size
        
        for batch in range(batches):
            numbers = []
            for i in range(batch_size):
                # Create different types of floating-point numbers
                choice = i % 4
                if choice == 0:
                    # Small decimals
                    num = random.uniform(-10.0, 10.0)
                elif choice == 1:
                    # Large numbers
                    num = random.uniform(-1000000.0, 1000000.0)
                elif choice == 2:
                    # Very small numbers
                    num = random.uniform(-0.001, 0.001)
                else:
                    # Scientific notation range
                    num = random.uniform(-1e6, 1e6)
                
                numbers.append(f"{num:.8f}")
            
            file.write('\n'.join(numbers) + '\n')
            
            if (batch + 1) % 1000 == 0:
                progress = ((batch + 1) * batch_size / count) * 100
                print(f"Progress: {progress:.1f}%")
    
    end_time = time.time()
    file_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"Mixed range file completed in {end_time - start_time:.2f} seconds")
    print(f"File size: {file_size:.2f} MB")

if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    random.seed(42)
    
    print("Choose an option:")
    print("1. Generate 11 crore uniform random floats (-1000 to 1000)")
    print("2. Generate 11 crore mixed range floats")
    print("3. Generate custom count")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        generate_floating_point_file()
    elif choice == "2":
        generate_different_ranges()
    elif choice == "3":
        try:
            custom_count = int(input("Enter number of floats to generate: "))
            custom_filename = input("Enter filename (press Enter for default): ").strip()
            if not custom_filename:
                custom_filename = "floating_points.txt"
            generate_floating_point_file(custom_filename, custom_count)
        except ValueError:
            print("Invalid number entered!")
    else:
        print("Invalid choice! Running default option...")
        generate_floating_point_file()
    
    print("\nScript completed!")


# %% Original cell 1
import os
import time
import numpy as np
from collections import defaultdict
import gc

# GPU libraries - install with: pip install cupy-cuda11x torch (adjust CUDA version)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy detected - GPU acceleration available")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not found - install with: pip install cupy-cuda11x")

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print(f"PyTorch CUDA available - GPU: {torch.cuda.get_device_name()}")
    else:
        print("PyTorch CUDA not available")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found - install with: pip install torch")

def get_gpu_info():
    """Display GPU information"""
    print("\n" + "="*60)
    print("GPU INFORMATION")
    print("="*60)
    
    if CUPY_AVAILABLE:
        try:
            print(f"CuPy GPU: {cp.cuda.get_device_name()}")
            print(f"GPU Memory: {cp.cuda.Device().mem_info[1] / (1024**3):.1f} GB total")
            free_mem = cp.cuda.Device().mem_info[0] / (1024**3)
            print(f"Free Memory: {free_mem:.1f} GB")
        except:
            print("CuPy GPU info unavailable")
    
    if TORCH_AVAILABLE:
        try:
            print(f"PyTorch GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
            allocated = torch.cuda.memory_allocated() / (1024**3)
            cached = torch.cuda.memory_reserved() / (1024**3)
            print(f"Allocated: {allocated:.1f} GB, Cached: {cached:.1f} GB")
        except:
            print("PyTorch GPU info unavailable")
    
    print("="*60)

def check_duplicates_cupy_gpu(input_filename, output_filename="gpu_duplicates.txt", 
                              chunk_size=10_000_000):
    """
    GPU-accelerated duplicate checker using CuPy.
    Processes large files in chunks using GPU memory.
    
    Args:
        input_filename (str): Input file with floating-point numbers
        output_filename (str): Output file for duplicates
        chunk_size (int): Numbers to process per GPU batch
    """
    
    if not CUPY_AVAILABLE:
        print("CuPy not available! Install with: pip install cupy-cuda11x")
        return
    
    print(f"\n🚀 GPU Processing with CuPy")
    print(f"Input: {input_filename}")
    print(f"Output: {output_filename}")
    print(f"Chunk size: {chunk_size:,}")
    
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found!")
        return
    
    start_time = time.time()
    
    # Dictionary to store counts (kept on CPU to avoid GPU memory issues)
    number_counts = defaultdict(int)
    total_processed = 0
    chunk_count = 0
    
    try:
        with open(input_filename, 'r') as file:
            chunk_data = []
            
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        num = float(line)
                        chunk_data.append(num)
                        total_processed += 1
                    except ValueError:
                        print(f"Warning: Invalid number at line {line_num}: {line}")
                
                # Process chunk when full
                if len(chunk_data) >= chunk_size:
                    chunk_count += 1
                    print(f"Processing chunk {chunk_count} ({len(chunk_data):,} numbers)...")
                    
                    # Move data to GPU
                    gpu_start = time.time()
                    gpu_array = cp.array(chunk_data, dtype=cp.float32)
                    
                    # Round to 6 decimal places for consistent comparison
                    gpu_rounded = cp.round(gpu_array, 6)
                    
                    # Get unique values and counts on GPU
                    unique_vals, counts = cp.unique(gpu_rounded, return_counts=True)
                    
                    # Move results back to CPU
                    unique_cpu = cp.asnumpy(unique_vals)
                    counts_cpu = cp.asnumpy(counts)
                    
                    # Update global counts
                    for val, count in zip(unique_cpu, counts_cpu):
                        number_counts[f"{val:.6f}"] += int(count)
                    
                    gpu_time = time.time() - gpu_start
                    print(f"  GPU processing: {gpu_time:.2f}s, Total: {total_processed:,}")
                    
                    # Clear GPU memory
                    del gpu_array, gpu_rounded, unique_vals, counts
                    cp.get_default_memory_pool().free_all_blocks()
                    
                    # Clear CPU chunk
                    chunk_data.clear()
                    gc.collect()
            
            # Process remaining data
            if chunk_data:
                chunk_count += 1
                print(f"Processing final chunk {chunk_count} ({len(chunk_data):,} numbers)...")
                
                gpu_array = cp.array(chunk_data, dtype=cp.float32)
                gpu_rounded = cp.round(gpu_array, 6)
                unique_vals, counts = cp.unique(gpu_rounded, return_counts=True)
                
                unique_cpu = cp.asnumpy(unique_vals)
                counts_cpu = cp.asnumpy(counts)
                
                for val, count in zip(unique_cpu, counts_cpu):
                    number_counts[f"{val:.6f}"] += int(count)
                
                # Clear GPU memory
                del gpu_array, gpu_rounded, unique_vals, counts
                cp.get_default_memory_pool().free_all_blocks()
    
    except Exception as e:
        print(f"Error during processing: {e}")
        return
    
    processing_time = time.time() - start_time
    
    # Find duplicates
    duplicates = {num: count for num, count in number_counts.items() if count > 1}
    
    print(f"\n📊 RESULTS:")
    print(f"Total numbers processed: {total_processed:,}")
    print(f"Unique numbers: {len(number_counts):,}")
    print(f"Duplicate numbers: {len(duplicates):,}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Speed: {total_processed/processing_time:,.0f} numbers/second")
    
    # Write results
    write_results(duplicates, output_filename, total_processed, len(number_counts))

def check_duplicates_pytorch_gpu(input_filename, output_filename="pytorch_duplicates.txt",
                                chunk_size=10_000_000):
    """
    GPU-accelerated duplicate checker using PyTorch.
    
    Args:
        input_filename (str): Input file with floating-point numbers
        output_filename (str): Output file for duplicates
        chunk_size (int): Numbers to process per GPU batch
    """
    
    if not TORCH_AVAILABLE:
        print("PyTorch CUDA not available!")
        return
    
    print(f"\n🚀 GPU Processing with PyTorch")
    print(f"Input: {input_filename}")
    print(f"Output: {output_filename}")
    print(f"Chunk size: {chunk_size:,}")
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found!")
        return
    
    start_time = time.time()
    number_counts = defaultdict(int)
    total_processed = 0
    chunk_count = 0
    
    try:
        with open(input_filename, 'r') as file:
            chunk_data = []
            
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        num = float(line)
                        chunk_data.append(num)
                        total_processed += 1
                    except ValueError:
                        continue
                
                if len(chunk_data) >= chunk_size:
                    chunk_count += 1
                    print(f"Processing chunk {chunk_count} ({len(chunk_data):,} numbers)...")
                    
                    gpu_start = time.time()
                    
                    # Move to GPU
                    tensor = torch.tensor(chunk_data, dtype=torch.float32, device=device)
                    
                    # Round for consistency
                    rounded = torch.round(tensor * 1000000) / 1000000  # 6 decimal places
                    
                    # Get unique values - PyTorch unique is different from NumPy
                    unique_vals = torch.unique(rounded)
                    
                    # Count occurrences manually (PyTorch doesn't have return_counts like NumPy)
                    for val in unique_vals:
                        count = torch.sum(rounded == val).item()
                        val_str = f"{val.item():.6f}"
                        number_counts[val_str] += count
                    
                    gpu_time = time.time() - gpu_start
                    print(f"  GPU processing: {gpu_time:.2f}s, Total: {total_processed:,}")
                    
                    # Clear GPU memory
                    del tensor, rounded, unique_vals
                    torch.cuda.empty_cache()
                    
                    chunk_data.clear()
                    gc.collect()
            
            # Process remaining data
            if chunk_data:
                chunk_count += 1
                print(f"Processing final chunk ({len(chunk_data):,} numbers)...")
                
                tensor = torch.tensor(chunk_data, dtype=torch.float32, device=device)
                rounded = torch.round(tensor * 1000000) / 1000000
                unique_vals = torch.unique(rounded)
                
                for val in unique_vals:
                    count = torch.sum(rounded == val).item()
                    val_str = f"{val.item():.6f}"
                    number_counts[val_str] += count
                
                del tensor, rounded, unique_vals
                torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Error during processing: {e}")
        return
    
    processing_time = time.time() - start_time
    duplicates = {num: count for num, count in number_counts.items() if count > 1}
    
    print(f"\n📊 RESULTS:")
    print(f"Total numbers processed: {total_processed:,}")
    print(f"Unique numbers: {len(number_counts):,}")
    print(f"Duplicate numbers: {len(duplicates):,}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Speed: {total_processed/processing_time:,.0f} numbers/second")
    
    write_results(duplicates, output_filename, total_processed, len(number_counts))

def write_results(duplicates, output_filename, total_numbers, unique_count):
    """Write duplicate results to file"""
    if duplicates:
        print(f"\n💾 Writing {len(duplicates):,} duplicates to: {output_filename}")
        
        with open(output_filename, 'w') as outfile:
            outfile.write("Number,Count\n")
            outfile.write("=" * 50 + "\n")
            
            # Sort by count (descending)
            sorted_duplicates = sorted(duplicates.items(), 
                                     key=lambda x: (-x[1], float(x[0])))
            
            for number, count in sorted_duplicates:
                outfile.write(f"{number},{count}\n")
        
        # Show top duplicates
        print(f"\n🔥 Top 10 most frequent duplicates:")
        for i, (number, count) in enumerate(sorted_duplicates[:10], 1):
            print(f"{i:2d}. {number} appears {count:,} times")
        
        file_size = os.path.getsize(output_filename) / (1024 * 1024)
        print(f"\nOutput file size: {file_size:.2f} MB")
    else:
        print("✅ No duplicate numbers found!")
        with open(output_filename, 'w') as outfile:
            outfile.write("No duplicate numbers found.\n")
            outfile.write(f"Total numbers: {total_numbers:,}\n")
            outfile.write(f"All numbers are unique: {unique_count:,}\n")

def optimize_chunk_size():
    """Suggest optimal chunk size based on available GPU memory"""
    if CUPY_AVAILABLE:
        try:
            free_mem_gb = cp.cuda.Device().mem_info[0] / (1024**3)
            # Use about 50% of free memory, accounting for float32 (4 bytes per number)
            suggested_size = int((free_mem_gb * 0.5 * 1024**3) / 4)
            return min(suggested_size, 50_000_000)  # Cap at 50M numbers
        except:
            pass
    
    if TORCH_AVAILABLE:
        try:
            # Conservative estimate
            return 10_000_000
        except:
            pass
    
    return 1_000_000  # Default fallback

if __name__ == "__main__":
    print("🔥 GPU-Accelerated Duplicate Checker 🔥")
    print("=" * 60)
    
    # Check GPU availability
    get_gpu_info()
    
    if not (CUPY_AVAILABLE or TORCH_AVAILABLE):
        print("\n❌ No GPU libraries available!")
        print("Install with:")
        print("  pip install cupy-cuda11x  # (adjust CUDA version)")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
        exit()
    
    # Get input parameters
    input_file = input("\nEnter input filename (default: floating_points.txt): ").strip()
    if not input_file:
        input_file = "floating_points.txt"
    
    if not os.path.exists(input_file):
        print(f"❌ Error: File '{input_file}' not found!")
        exit()
    
    # File info
    file_size_gb = os.path.getsize(input_file) / (1024**3)
    print(f"\n📁 Input file size: {file_size_gb:.2f} GB")
    
    # Choose GPU library
    print("\nChoose GPU processing method:")
    methods = []
    if CUPY_AVAILABLE:
        methods.append("1. CuPy (Recommended - faster for large arrays)")
        print("1. CuPy (Recommended - faster for large arrays)")
    if TORCH_AVAILABLE:
        methods.append("2. PyTorch (Good alternative)")
        print("2. PyTorch (Good alternative)")
    
    choice = input("Enter choice: ").strip()
    
    # Optimize chunk size
    suggested_chunk = optimize_chunk_size()
    chunk_input = input(f"Chunk size (default: {suggested_chunk:,}): ").strip()
    chunk_size = int(chunk_input) if chunk_input.isdigit() else suggested_chunk
    
    output_file = input("Output filename (default: gpu_duplicates.txt): ").strip()
    if not output_file:
        output_file = "gpu_duplicates.txt"
    
    print(f"\n🚀 Starting GPU processing...")
    print(f"Estimated processing time: {file_size_gb * 2:.1f} minutes")
    
    # Process based on choice
    if choice == "1" and CUPY_AVAILABLE:
        check_duplicates_cupy_gpu(input_file, output_file, chunk_size)
    elif choice == "2" and TORCH_AVAILABLE:
        check_duplicates_pytorch_gpu(input_file, output_file, chunk_size)
    elif CUPY_AVAILABLE:
        check_duplicates_cupy_gpu(input_file, output_file, chunk_size)
    elif TORCH_AVAILABLE:
        check_duplicates_pytorch_gpu(input_file, output_file, chunk_size)
    
    print("\n✅ GPU processing completed!")
