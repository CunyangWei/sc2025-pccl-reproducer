import torch
import torch.distributed as dist
from mpi4py import MPI
import numpy as np
import os
from argparse import ArgumentParser
import csv
from pccl import ProcessGroups, all_to_all_2D, _all_to_all
from pccl.build_kernels import build as build_pccl
from benchmark_raw_collectives.utils import time_something, init, allclose

def get_gpu_counts_and_job_id():
    gpu_count = int(os.getenv("SLURM_NTASKS", "1"))  # Default to 1 if not found
    slurm_job_id = os.getenv("SLURM_JOB_ID", "unknown")
    return gpu_count, slurm_job_id

def validate_all_to_all_correctness(output_tensor, input_tensor, world_size):
    """
    Validate that all_to_all operation is correct.
    Each process should receive block i from all other processes at position i.
    """
    # For all_to_all, process i should receive block i from all processes
    block_size = input_tensor.numel() // world_size
    
    # Create expected output tensor
    expected_output = torch.zeros_like(output_tensor)
    
    for src_rank in range(world_size):
        dest_block_start = src_rank * block_size
        dest_block_end = dest_block_start + block_size
        
        # The input tensor of src_rank at block rank should appear in output at block src_rank
        expected_output[dest_block_start:dest_block_end] = float(src_rank) * torch.ones(block_size, dtype=output_tensor.dtype, device=output_tensor.device)
    
    return allclose(output_tensor, expected_output)

if __name__ == "__main__":
    init()
    parser = ArgumentParser()
    parser.add_argument("--num-gpus-per-node", 
                        type=int, 
                        required=True, 
                        help="specify number of GPUs/GCDs per node")
    parser.add_argument("--machine",
                        type=str,
                        required=True,
                        help="specify the machine you are running on. Will be used to create folders")
    parser.add_argument("--library", type=str, choices=["pccl", "mpi", "xccl"])
    parser.add_argument("--pccl-disable-cpp-backend", 
                    dest="use_pccl_cpp_backend",
                    action="store_false",
                    help="Disable the C++ backend in PCCL. Will fall back to the inefficient Python backend.")
    parser.add_argument("--test", 
                        action="store_true",
                        help="test for correctness")
    parser.add_argument("--pccl-algorithm", 
                        type=str,
                        choices=["spread_out", "pairwise_exchange", "bruck", "hypre", "radix_bruck", "uniform_modified_radix_bruck", "nccl"],
                        default="spread_out",
                        help="Choose the all-to-all algorithm for PCCL")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["bf16", "fp32"],
                        default="fp32")
    parser.add_argument("--radix",
                        type=int,
                        default=-1,
                        help="specify the radix for the radix_bruck algorithm")
    parser.add_argument("--outer-nccl",
                        action="store_true",
                        help="use NCCL communicator for inter-node (outer group) communication")

    args = parser.parse_args()
    
    if args.use_pccl_cpp_backend:
        if dist.get_rank() == 0:
            build_pccl()
            MPI.COMM_WORLD.Barrier()
        else:
            MPI.COMM_WORLD.Barrier()
            build_pccl()

    gpu_count, slurm_job_id = get_gpu_counts_and_job_id()
    sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    # sizes = np.array([1])
    unit = "MB"
    algorithm = args.pccl_algorithm
    
    if args.library == "pccl":
        # creating 2D process groups for intra- and inter-node communication
        if algorithm == "nccl":
            pg = ProcessGroups(args.num_gpus_per_node, 
                               dist.get_world_size() // args.num_gpus_per_node,
				                  inner_group_backend="nccl",
				                  outer_group_backend="nccl")
        elif args.outer_nccl:
            pg = ProcessGroups(args.num_gpus_per_node, 
                               dist.get_world_size() // args.num_gpus_per_node,
                               inner_group_backend="nccl",
                               outer_group_backend="nccl")
        else:
            print("mpi Alltoall")
            pg = ProcessGroups(args.num_gpus_per_node, 
                                   dist.get_world_size() // args.num_gpus_per_node,
                                   inner_group_backend="nccl",
                                   outer_group_backend="mpi",)
        args.library += "_cpp" if args.use_pccl_cpp_backend else "_py"
        args.library += f"_{algorithm}"
        if args.outer_nccl and algorithm != "nccl":
            args.library += "_outer_nccl"
        
        # For radix_bruck or uniform_modified_radix_bruck, calculate and append the actual radix value to the column name
        if algorithm == "radix_bruck" or algorithm == "uniform_modified_radix_bruck":
            # Calculate radix the same way as in the PCCL functions
            if args.radix == -1:
                print("Calculating radix from group size")
                _, outer_group_size = pg.get_world_size()
                actual_radix = max(2, int(np.ceil(np.sqrt(outer_group_size))))
            else:
                print(f"Using radix {args.radix} from command line")
                actual_radix = args.radix
            args.library += f"_{actual_radix}"
        
        function = all_to_all_2D
    elif args.library == "mpi":
        pg = MPI.COMM_WORLD
        function = _all_to_all
    elif args.library == "xccl":
        pg = None # None is mapped to comm-world in torch.dist + xccl
        function = _all_to_all
        
    if args.use_pccl_cpp_backend:
        args.library += "_cpp"

    data_folder = f"./data/all_to_all/{args.machine}"
    os.makedirs(data_folder, exist_ok=True)

    csv_filename = os.path.join(data_folder,
                                f"gpus_{gpu_count}_slurm_{slurm_job_id}.csv")
    
    # Read existing CSV data if file exists
    existing_data = {}
    base_columns = ["gpu_count", "slurm_job_id", "tensor_size", "unit"]
    current_library_column = f"time_{args.library}"
    
    if os.path.exists(csv_filename):
        with open(csv_filename, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Use tensor_size as key for matching rows
                key = row["tensor_size"]
                existing_data[key] = row
    
    # Collect new timing data
    new_timing_data = {}

    for size in sizes:
            if dist.get_rank() == 0:
                print(f"tensor size = {size} {unit}")
            mult = 2**20 if unit == "MB" else 2**10
            
            tensor_numel = size * (mult) // 4 if args.dtype == "fp32" else size * (mult) // 2
            dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
            
            # For all_to_all, input and output tensors have the same size
            output_tensor = torch.empty((tensor_numel,), dtype=dtype, device="cuda")
            
            # Create input tensor with rank-specific pattern for testing
            if args.test:
                input_tensor = torch.full((tensor_numel,), float(dist.get_rank()), dtype=dtype, device="cuda")
                # Set different values for each block to help with validation
                block_size = tensor_numel // dist.get_world_size()
                for i in range(dist.get_world_size()):
                    start_idx = i * block_size
                    end_idx = start_idx + block_size
                    input_tensor[start_idx:end_idx] = float(dist.get_rank()) * 10 + float(i)
            else:
                input_tensor = torch.randn((tensor_numel,), dtype=dtype, device="cuda")
            
            # Create gold standard if testing
            if args.test:
                output_tensor_gold = torch.empty_like(output_tensor)
                if args.library == "mpi":
                    # Use MPI's built-in alltoall for gold standard
                    torch.cuda.current_stream().synchronize()
                    MPI.COMM_WORLD.Alltoall(input_tensor, output_tensor_gold)
                else:
                    # Use torch distributed for gold standard
                    input_list = list(torch.chunk(input_tensor, dist.get_world_size()))
                    output_list = list(torch.chunk(output_tensor_gold, dist.get_world_size()))
                    dist.all_to_all(output_list, input_list)

            kwargs = {"use_pccl_cpp_backend": args.use_pccl_cpp_backend}
            if args.library.startswith("pccl"):
                kwargs["algorithm"] = algorithm
                # For radix_bruck or uniform_modified_radix_bruck, add default radix calculation (sqrt of outer group size)
                if algorithm == "radix_bruck" or algorithm == "uniform_modified_radix_bruck":
                    # Default radix will be calculated in the PCCL functions based on group size
                    kwargs["radix"] = -1  # -1 means use default calculation
                
            time = time_something(function, 
                                  output_tensor, 
                                  input_tensor, 
                                  group=pg, 
                                  **kwargs)
            
            # Test for correctness
            if args.test:
                if allclose(output_tensor, output_tensor_gold):
                    if dist.get_rank() == 0:
                        print(f"✓ Correctness test passed for size {size} {unit}")
                else:
                    if dist.get_rank() == 0:
                        print(f"✗ Correctness test failed for size {size} {unit}")
                        print(f"Max absolute difference: {torch.max(torch.abs(output_tensor - output_tensor_gold)).item()}")
                        print(f"Max relative difference: {torch.max(torch.abs((output_tensor - output_tensor_gold) / (output_tensor_gold + 1e-8))).item()}")
                    
            # Store timing data for this size
            new_timing_data[str(size)] = time

    # Merge existing data with new timing data and write to CSV
    # if dist.get_rank() == 0:
    #     # Determine all columns that should be present
    #     all_columns = set(base_columns)
    #     if existing_data:
    #         # Add all existing timing columns
    #         for row in existing_data.values():
    #             all_columns.update(row.keys())
    #     # Add current library column
    #     all_columns.add(current_library_column)
        
    #     # Convert to sorted list for consistent ordering
    #     all_columns = sorted(all_columns)
        
    #     # Prepare merged data
    #     merged_data = []
    #     for size in sizes:
    #         size_str = str(size)
            
    #         # Start with base data
    #         if size_str in existing_data:
    #             row_data = existing_data[size_str].copy()
    #         else:
    #             row_data = {
    #                 "gpu_count": gpu_count,
    #                 "slurm_job_id": slurm_job_id,
    #                 "tensor_size": size,
    #                 "unit": unit
    #             }
            
    #         # Add/update timing data for current library
    #         row_data[current_library_column] = new_timing_data[size_str]
            
    #         # Ensure all columns are present (fill missing with empty string)
    #         for col in all_columns:
    #             if col not in row_data:
    #                 row_data[col] = ""
            
    #         merged_data.append(row_data)
        
    #     # Write merged data to CSV
    #     with open(csv_filename, "w", newline="") as f:
    #         writer = csv.DictWriter(f, fieldnames=all_columns)
    #         writer.writeheader()
    #         writer.writerows(merged_data)

    if dist.get_rank() == 0:
        print(f"Benchmark completed. Results saved to {csv_filename}")