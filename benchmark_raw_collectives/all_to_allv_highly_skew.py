import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
from mpi4py import MPI
import numpy as np
import os
from argparse import ArgumentParser
import csv
from typing import List, Union, Dict
from pccl import ProcessGroups
from pccl.all_to_allv_highly_skew import all_to_allv_withcomm, _all_to_allv
from pccl.build_kernels import build as build_pccl
from benchmark_raw_collectives.utils import time_something, init, allclose

def get_gpu_counts_and_job_id():
    gpu_count = int(os.getenv("SLURM_NTASKS", "1"))  # Default to 1 if not found
    slurm_job_id = os.getenv("SLURM_JOB_ID", "unknown")
    return gpu_count, slurm_job_id
def parse_splits_from_csv(csv_path: str, world_size: int, rank: int):
    """
    Read the input/output shapes and split sizes for the current rank from matrix.csv.
    Assumes each CSV row contains the following fields (from real MOE training logs):
    - direction: 'forward' or 'backward'
    - group_rank: intra-group rank (can be matched directly when equal to global rank)
    - input_shape: e.g. "[N, H]"
    - output_shape: e.g. "[M, H]"
    - input_split_sizes: e.g. "[a b c ...]" or comma-separated "[a, b, ...]"
    - output_split_sizes: same as above
    Returns:
      input_rows, hidden_size, output_rows, sendcounts_rows(list), recvcounts_rows(list)
    """
    def parse_shape(s: str):
        s = s.strip().strip('"').strip()
        s = s.strip('[]')
        parts = [p.strip() for p in s.split(',')]
        if len(parts) != 2:
            # May also be space-separated
            parts = [p.strip() for p in s.split()]
        return int(parts[0]), int(parts[1])

    def parse_int_list(s: str, expected_len: int):
        s = s.strip().strip('"').strip()
        s = s.strip('[]')
        if ',' in s:
            parts = [p.strip() for p in s.split(',') if p.strip()]
        else:
            parts = [p.strip() for p in s.split() if p.strip()]
        vals = [int(x) for x in parts]
        if len(vals) != expected_len:
            raise ValueError(f"Split sizes length {len(vals)} != world_size {expected_len}")
        return vals

    selected = None
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            try:
                grp = row.get('group_rank') if 'group_rank' in row else row.get(' global_rank')
                glb = row.get('global_rank') if 'global_rank' in row else row.get(' global_rank')
                rank_val = grp if grp is not None else glb
                if int(rank_val) != rank:
                    continue
                selected = row
                break
            except Exception:
                continue
    if selected is None:
        raise RuntimeError(f"No matching row in {csv_path} for rank {rank}")

    in_rows, hidden = parse_shape(selected['input_shape'])
    out_rows, hidden2 = parse_shape(selected['output_shape'])
    if hidden != hidden2:
        raise ValueError("input/output hidden size mismatch")
    sendcounts_rows = parse_int_list(selected['input_split_sizes'], world_size)
    recvcounts_rows = parse_int_list(selected['output_split_sizes'], world_size)
    if sum(sendcounts_rows) != in_rows:
        raise ValueError("Sum of input_split_sizes != input_rows")
    if sum(recvcounts_rows) != out_rows:
        raise ValueError("Sum of output_split_sizes != output_rows")
    return in_rows, hidden, out_rows, sendcounts_rows, recvcounts_rows


def parse_all_cycles_from_csv(csv_path: str, world_size: int, rank: int):
    """
    Read data for all cycles from matrix.csv.
    Each cycle contains 16 rows of data (rank 0-15), which may span multiple rows.
    If input_shape or output_shape does not contain exactly two numbers, skip that cycle.
    Returns:
      cycles: List[Dict] - each cycle contains data for all ranks
    """
    def parse_shape(s: str):
        s = s.strip().strip('"').strip()
        s = s.strip('[]')
        parts = [p.strip() for p in s.split(',')]
        if len(parts) != 2:
            # May also be space-separated
            parts = [p.strip() for p in s.split()]
        return int(parts[0]), int(parts[1])

    def is_valid_shape(s: str):
        """Check whether the shape string contains exactly two numbers"""
        try:
            s = s.strip().strip('"').strip()
            s = s.strip('[]')
            parts = [p.strip() for p in s.split(',')]
            if len(parts) != 2:
                # May also be space-separated
                parts = [p.strip() for p in s.split()]
            if len(parts) != 2:
                return False
            # Try to convert to integers
            int(parts[0])
            int(parts[1])
            return True
        except:
            return False

    def parse_int_list(s: str, expected_len: int):
        s = s.strip().strip('"').strip()
        s = s.strip('[]')
        if ',' in s:
            parts = [p.strip() for p in s.split(',') if p.strip()]
        else:
            parts = [p.strip() for p in s.split() if p.strip()]
        vals = [int(x) for x in parts]
        if len(vals) != expected_len:
            raise ValueError(f"Split sizes length {len(vals)} != world_size {expected_len}")
        return vals

    def is_cycle_valid(cycle):
        """Check whether a cycle is valid (input_shape and output_shape are two numbers for all ranks)"""
        for rank_data in cycle.values():
            input_shape = rank_data.get('input_shape', '')
            output_shape = rank_data.get('output_shape', '')
            if not is_valid_shape(input_shape) or not is_valid_shape(output_shape):
                return False
        return True

    cycles = []
    current_cycle = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            try:

                grp = row.get('group_rank') if 'group_rank' in row else row.get(' global_rank')
                glb = row.get('global_rank') if 'global_rank' in row else row.get(' global_rank')
                rank_val = grp if grp is not None else glb
                current_rank = int(rank_val)

                # If we encounter rank 0, a new cycle has started
                if current_rank == 0 and current_cycle:
                    # Validate whether the previous cycle is valid
                    if is_cycle_valid(current_cycle):
                        cycles.append(current_cycle)
                    else:
                        # print(f"Warning: Skipping invalid cycle with incomplete shape data")
                        pass
                    current_cycle = {}

                # Add the current rank data to the current cycle
                current_cycle[current_rank] = row

            except Exception as e:
                print(f"Warning: Error parsing row: {e}")
                continue

    # Add the last cycle
    if current_cycle:
        if is_cycle_valid(current_cycle):
            cycles.append(current_cycle)
        else:
            # print(f"Warning: Skipping invalid cycle with incomplete shape data")
            pass

    return cycles


def get_cycle_data_for_rank(cycles: List[Dict], rank: int, cycle_idx: int):
    """
    Get data for the specified rank and cycle from cycles.
    Returns:
      input_rows, hidden_size, output_rows, sendcounts_rows(list), recvcounts_rows(list)
    """
    def parse_shape(s: str):
        s = s.strip().strip('"').strip()
        s = s.strip('[]')
        parts = [p.strip() for p in s.split(',')]
        if len(parts) != 2:
            # May also be space-separated
            parts = [p.strip() for p in s.split()]
        return int(parts[0]), int(parts[1])

    def parse_int_list(s: str, expected_len: int):
        s = s.strip().strip('"').strip()
        s = s.strip('[]')
        if ',' in s:
            parts = [p.strip() for p in s.split(',') if p.strip()]
        else:
            parts = [p.strip() for p in s.split() if p.strip()]
        vals = [int(x) for x in parts]
        if len(vals) != expected_len:
            raise ValueError(f"Split sizes length {len(vals)} != world_size {expected_len}")
        return vals

    if cycle_idx >= len(cycles):
        raise IndexError(f"Cycle index {cycle_idx} out of range. Total cycles: {len(cycles)}")

    cycle = cycles[cycle_idx]
    if rank not in cycle:
        raise KeyError(f"Rank {rank} not found in cycle {cycle_idx}")

    row = cycle[rank]
    in_rows, hidden = parse_shape(row['input_shape'])
    out_rows, hidden2 = parse_shape(row['output_shape'])
    if hidden != hidden2:
        raise ValueError("input/output hidden size mismatch")

    sendcounts_rows = parse_int_list(row['input_split_sizes'], len(cycle))
    recvcounts_rows = parse_int_list(row['output_split_sizes'], len(cycle))

    if sum(sendcounts_rows) != in_rows:
        raise ValueError("Sum of input_split_sizes != input_rows")
    if sum(recvcounts_rows) != out_rows:
        raise ValueError("Sum of output_split_sizes != output_rows")

    return in_rows, hidden, out_rows, sendcounts_rows, recvcounts_rows

def parse_int_list_simple(s: str, expected_len: int) -> List[int]:
    """
    Simple parser: convert a string like "[a, b, c]" or "a b c" into a Python list[int].
    """
    s = s.strip().strip('"').strip()
    s = s.strip('[]')
    if ',' in s:
        parts = [p.strip() for p in s.split(',') if p.strip()]
    else:
        parts = [p.strip() for p in s.split() if p.strip()]
    vals = [int(x) for x in parts]
    if len(vals) != expected_len:
        raise ValueError(f"Split sizes length {len(vals)} != world_size {expected_len}")
    return vals

def create_gold_standard_alltoallv(input_tensor: torch.Tensor,
                                   sendcounts: List[Union[int, np.int64]],
                                   recvcounts: List[Union[int, np.int64]],
                                   library: str):
    """
    Use torch.distributed.all_to_all_single as the unified gold-standard reference.
    """
    output_tensor_gold = torch.zeros((sum(recvcounts),), dtype=input_tensor.dtype, device=input_tensor.device)
    torch.cuda.current_stream().synchronize()
    input_tensor = input_tensor.contiguous()
    output_tensor_gold = output_tensor_gold.contiguous()
    # PyTorch all_to_all_single: (out, inp, out_split_sizes, in_split_sizes)
    dist.all_to_all_single(output_tensor_gold, input_tensor, recvcounts, sendcounts)
    return output_tensor_gold

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
    parser.add_argument("--library", type=str, choices=["pccl", "mpi", "nccl"])
    parser.add_argument("--pccl-disable-cpp-backend",
                    dest="use_pccl_cpp_backend",
                    action="store_false",
                    help="Disable the C++ backend in PCCL. Will fall back to the inefficient Python backend.")
    parser.add_argument("--test",
                        action="store_true",
                        help="test for correctness")
    parser.add_argument("--pccl-algorithm",
                        type=str,
                        choices=["spread_out", "pairwise_sendrecv", "pairwise_exchange", "pairwise_scattered", "openmpi_pairwise", "openmpi_basic_linear", "openmpi_basic_inplace", "openmpi_inter", "openmpi_persistent"],
                        default="spread_out",
                        help="Choose the all-to-allv algorithm for PCCL")
    # Fixed to bf16; pattern/size loop is no longer needed
    parser.add_argument("--profile",
                        action="store_true",
                        help="profile the alltoallv")

    args = parser.parse_args()

    # sizes = np.array([32, 64, 128, 256])  # MB
    sizes = np.array([32])  # MB

    if args.use_pccl_cpp_backend:
        if dist.get_rank() == 0:
            build_pccl()
            MPI.COMM_WORLD.Barrier()
        else:
            MPI.COMM_WORLD.Barrier()
            build_pccl()

    gpu_count, slurm_job_id = get_gpu_counts_and_job_id()
    algorithm = args.pccl_algorithm

    if args.library == "pccl":
        # creating 2D process groups for intra- and inter-node communication
        pg = ProcessGroups(args.num_gpus_per_node,
                           dist.get_world_size() // args.num_gpus_per_node,
                           inner_group_backend="nccl",
                        #    outer_group_backend="mpi", create_inner_mpi_group=True)
                           outer_group_backend="nccl", create_inner_mpi_group=True)
        args.library += "_cpp" if args.use_pccl_cpp_backend else "_py"
        args.library += f"_2D"
        function = all_to_allv_withcomm
        # function = all_to_allv_2D
    elif args.library == "mpi":
        pg = MPI.COMM_WORLD
        function = _all_to_allv
    elif args.library == "nccl":
        pg = None  # None is mapped to comm-world in torch.dist + nccl
        function = _all_to_allv

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if args.use_pccl_cpp_backend:
        args.library += "_cpp"
        args.library += f"_{algorithm}"

    dtype = torch.bfloat16
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()

    # Read/prepare CSV data schema for writing results (no pattern/size now)
    existing_data = {}
    base_columns = ["gpu_count"]
    current_library_column = f"time_{args.library}"

    if args.library == "pccl":
        args.library += "_cpp" if args.use_pccl_cpp_backend else "_py"
        args.library += f"_{algorithm}"

    # Prepare output file path and load existing rows
    data_folder = f"./data/all_to_all/{args.machine}"
    os.makedirs(data_folder, exist_ok=True)

    # Read matrix.csv and parse data for all cycles
    csv_path = os.path.join(os.path.dirname(__file__), f"test_alltoall_{world_size}.csv")
    rank = dist.get_rank()

    # Parse all cycles
    cycles = parse_all_cycles_from_csv(csv_path, world_size, rank)
    total_cycles = len(cycles)

    if rank == 0:
        print(f"Found {total_cycles} cycles in matrix.csv")

    # Fix dtype to bf16
    dtype = torch.bfloat16
    device = f"cuda:{rank % torch.cuda.device_count()}"

    # Create profile object (if profiling is enabled)
    prof = None
    if args.profile:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()
    for size_mb in sizes:
        if rank == 0:
            print(f"\n=== Size {size_mb}MB ===")
        csv_filename = os.path.join(data_folder, f"gpus_{gpu_count}_slurm_{slurm_job_id}_{size_mb}.csv")
        if os.path.exists(csv_filename):
            with open(csv_filename, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = f"{row.get('rows_in','')}_{row.get('rows_out','')}_{row.get('hidden','')}"
                    existing_data[key] = row
        # Store results for all cycles
        all_results = []
        # Iterate over all test sizes
        for cycle_idx in range(min(2500, total_cycles)):
        # for cycle_idx in range(total_cycles):
            if rank == 0:
                print(f"\n=== Processing Cycle {cycle_idx + 1}/{total_cycles} ===")
            try:
                # Get data for the current cycle
                in_rows, hidden, out_rows, sendcounts_rows, recvcounts_rows = get_cycle_data_for_rank(
                    cycles, rank, cycle_idx
                )
                scale = size_mb / 64.0
                hidden = int(hidden * scale)

                # 2D shape: [rows, hidden]
                input_tensor = torch.randn((in_rows, hidden), dtype=dtype, device=device)
                output_tensor = torch.zeros((out_rows, hidden), dtype=dtype, device=device)

                # In 2D mode, counts are expressed as row counts; all_to_allv_2D internally scales by hidden
                sendcounts_rows_list = [int(x) for x in sendcounts_rows]
                recvcounts_rows_list = [int(x) for x in recvcounts_rows]

                # Gold standard: use 1D flattened reference implementation, passing element counts
                if args.test:
                    flat_sendcounts = [int(x) * hidden for x in sendcounts_rows_list]
                    flat_recvcounts = [int(x) * hidden for x in recvcounts_rows_list]
                    gold_input = input_tensor.view(-1)
                    output_tensor_gold = create_gold_standard_alltoallv(gold_input, flat_sendcounts, flat_recvcounts, args.library)

                # Build the global communication matrix for the current cycle (in row counts, unscaled)
                # Shape: [world_size, world_size]; row r is the input_split_sizes of rank r
                current_cycle = cycles[cycle_idx]
                comm_matrix_rows = np.zeros((world_size, world_size), dtype=np.int64)
                for r in range(world_size):
                    row_r = current_cycle.get(r)
                    if row_r is None:
                        raise KeyError(f"Cycle {cycle_idx} missing data for rank {r}")
                    splits_r = parse_int_list_simple(row_r['input_split_sizes'], world_size)
                    comm_matrix_rows[r, :] = np.asarray(splits_r, dtype=np.int64)

                # Compute imbalance metrics (based on row/column totals)
                send_totals = comm_matrix_rows.sum(axis=1).astype(np.float64)
                recv_totals = comm_matrix_rows.sum(axis=0).astype(np.float64)

                def _safe_mean(vec):
                    return float(np.mean(vec)) if vec.size > 0 else 0.0

                mean_send = _safe_mean(send_totals)
                mean_recv = _safe_mean(recv_totals)
                p2m_send = float(np.max(send_totals)) / mean_send if mean_send > 0 else 0.0
                p2m_recv = float(np.max(recv_totals)) / mean_recv if mean_recv > 0 else 0.0
                peak_to_mean = max(p2m_send, p2m_recv)

                def _cv(vec):
                    mu = _safe_mean(vec)
                    return float(np.std(vec)) / mu if mu > 0 else 0.0

                cv_out = _cv(send_totals)
                cv_in = _cv(recv_totals)
                cv_val = max(cv_in, cv_out)

                def _gini(vec):
                    n = vec.size
                    s = float(np.sum(vec))
                    if n == 0 or s <= 0:
                        return 0.0
                    sorted_vec = np.sort(vec.astype(np.float64))
                    cumsum = np.cumsum(sorted_vec)
                    return float(1.0 - 2.0 * float(np.sum(cumsum)) / (n * s) + 1.0 / n)

                gini_out = _gini(send_totals)
                gini_in = _gini(recv_totals)
                gini_val = max(gini_in, gini_out)

                kwargs = {"use_pccl_cpp_backend": args.use_pccl_cpp_backend}
                if args.library.startswith("pccl"):
                    kwargs["algorithm"] = algorithm
                    kwargs["comm_matrix_rows"] = comm_matrix_rows

                # Time and execute
                if args.library.startswith("pccl"):
                    # SCRIPT0: use all_to_allv_2D (2D + row counts)
                    func = all_to_allv_withcomm
                    # func = all_to_allv_2D
                    out_arg = output_tensor
                    in_arg = input_tensor
                    sc = sendcounts_rows_list
                    rc = recvcounts_rows_list
                elif args.library == "nccl" and not args.use_pccl_cpp_backend:
                    # SCRIPT2: use _all_to_allv (internally dist.all_to_all_single), 2D + row counts
                    func = _all_to_allv
                    out_arg = output_tensor
                    in_arg = input_tensor
                    sc = sendcounts_rows_list
                    rc = recvcounts_rows_list
                else:
                    # SCRIPT1: use _all_to_allv, 1D flattened + element counts
                    func = _all_to_allv
                    out_arg = output_tensor.view(-1)
                    in_arg = input_tensor.view(-1)
                    sc = [x * hidden for x in sendcounts_rows_list]
                    rc = [x * hidden for x in recvcounts_rows_list]


                elapsed_time = time_something(func,
                                            out_arg,
                                            in_arg,
                                            sc,
                                            rc,
                                            group=pg,
                                            warmup_iters=5, timed_iters=20,
                                            prof=prof,
                                            **kwargs)

                # Correctness test
                cycle_passed = True
                if args.test:
                    # Flatten the 2D output and compare against the gold standard
                    local_pass = allclose(output_tensor.view(-1), output_tensor_gold)
                    local_result = 1 if local_pass else 0
                    global_result = MPI.COMM_WORLD.allreduce(local_result, MPI.MIN)
                    cycle_passed = (global_result == 1)

                    if global_result == 1:
                        if rank == 0:
                            print(f"✓ Cycle {cycle_idx + 1} correctness test passed")
                    else:
                        if rank == 0:
                            print(f"✗ Cycle {cycle_idx + 1} correctness test failed")
                        if not local_pass:
                            max_abs_diff = torch.max(torch.abs(output_tensor.view(-1) - output_tensor_gold)).item()
                            max_rel_diff = torch.max(torch.abs((output_tensor.view(-1) - output_tensor_gold) / (output_tensor_gold + 1e-8))).item()
                            print(f"Rank {rank} - Max absolute difference: {max_abs_diff}")
                            print(f"Rank {rank} - Max relative difference: {max_rel_diff}")

                if rank == 0:
                    print(f"Cycle {cycle_idx + 1} completed: {elapsed_time:.2f}ms, peak_to_mean={peak_to_mean}, rows_out={out_rows}, hidden={hidden}")

                # Store results
                all_results.append({
                    "cycle_idx": cycle_idx,
                    "elapsed_time": elapsed_time,
                    "peak_to_mean": peak_to_mean,
                    "cv": cv_val,
                    "gini": gini_val
                })

            except Exception as e:
                if rank == 0:
                    print(f"Error processing cycle {cycle_idx + 1}: {e}")
                all_results.append({
                    "cycle_idx": cycle_idx,
                    "elapsed_time": 0.0,
                    "error": str(e),
                    "peak_to_mean": "",
                    "cv": "",
                    "gini": ""
                })

        # Stop profiling and save results
        if prof is not None:
            prof.stop()
            if rank == 0:
                # Save profiling results
                prof.export_chrome_trace(f"profile_{args.library}_in.json")
                print(f"Profile saved to profile_{args.library}.json")



        # # Write results to CSV file
        if rank == 0:
            existing_data = {}
            existing_columns = set()
            if os.path.exists(csv_filename):
                with open(csv_filename, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        existing_columns.update(reader.fieldnames)
                    for row in reader:
                        key = row.get("cycle_idx")
                        if key is None or key == "":
                            continue
                        existing_data[key] = row

            all_columns = set(base_columns)
            all_columns.update({"cycle_idx", current_library_column, "peak_to_mean", "cv", "gini"})
            all_columns.update(existing_columns)

            for result in all_results:
                key = str(result["cycle_idx"])
                row_data = existing_data.get(
                    key,
                    {
                        "gpu_count": gpu_count,
                        "cycle_idx": result["cycle_idx"],
                    },
                )
                row_data[current_library_column] = f"{result['elapsed_time']:.4f}"
                if isinstance(result.get("peak_to_mean"), (int, float)):
                    row_data["peak_to_mean"] = f"{result['peak_to_mean']:.6f}"
                elif "peak_to_mean" not in row_data:
                    row_data["peak_to_mean"] = result.get("peak_to_mean", "")
                if isinstance(result.get("cv"), (int, float)):
                    row_data["cv"] = f"{result['cv']:.6f}"
                elif "cv" not in row_data:
                    row_data["cv"] = result.get("cv", "")
                if isinstance(result.get("gini"), (int, float)):
                    row_data["gini"] = f"{result['gini']:.6f}"
                elif "gini" not in row_data:
                    row_data["gini"] = result.get("gini", "")
                if "error" in result:
                    row_data["error"] = result["error"]
                    all_columns.add("error")
                existing_data[key] = row_data

            fieldnames = sorted(all_columns)
            with open(csv_filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(
                    existing_data[k] for k in sorted(existing_data.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
                )

            # Summary statistics
            total_cycles = len(all_results)
            avg_time = sum(r["elapsed_time"] for r in all_results) / total_cycles if total_cycles > 0 else 0

            print(f"\n=== Benchmark Summary ===")
            print(f"Total cycles: {total_cycles}")
            print(f"Average time: {avg_time:.2f}ms")
            print(f"Results saved to {csv_filename}")
