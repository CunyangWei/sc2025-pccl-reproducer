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
from pccl.all_to_allv_lightly_skew import all_to_allv_2D, _all_to_allv
from pccl.all_to_allv_simple_2D import all_to_allv_simple_2D
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
    - group_rank: rank within the group (can be matched directly when equal to global rank)
    - input_shape: in the form "[N, H]"
    - output_shape: in the form "[M, H]"
    - input_split_sizes: in the form "[a b c ...]" or comma-separated "[a, b, ...]"
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
    Read all cycles of data from matrix.csv.
    Each cycle contains 16 rows of data (rank 0-15); data may span multiple rows.
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
            # Try converting to integers
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
        """Check whether the cycle is valid (all ranks have two-number input_shape and output_shape)"""
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

                # If we encounter rank 0, it means a new cycle has started
                if current_rank == 0 and current_cycle:
                    # Validate whether the previous cycle is valid
                    if is_cycle_valid(current_cycle):
                        cycles.append(current_cycle)
                    else:
                        # print(f"Warning: Skipping invalid cycle with incomplete shape data")
                        pass
                    current_cycle = {}

                # Add the current rank's data to the current cycle
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


def get_cycle_data_for_rank(cycles: List[Dict], rank: int, cycle_idx: int, world_size: int):
    """
    Get data for the specified rank and cycle from cycles, and expand the communication matrix to the target world_size.
    Returns:
      input_rows, hidden_size, output_rows, sendcounts_rows(list), recvcounts_rows(list), comm_matrix(np.ndarray)
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

    if world_size <= 0:
        raise ValueError(f"Invalid world_size {world_size}")

    if cycle_idx >= len(cycles):
        raise IndexError(f"Cycle index {cycle_idx} out of range. Total cycles: {len(cycles)}")

    cycle = cycles[cycle_idx]
    base_world_size = len(cycle)
    if base_world_size == 0:
        raise ValueError("Cycle is empty")

    # Build the 16x16 (or base_world_size x base_world_size) base communication matrix
    base_matrix = np.zeros((base_world_size, base_world_size), dtype=np.int64)
    for base_rank in range(base_world_size):
        row_data = cycle.get(base_rank)
        if row_data is None:
            raise KeyError(f"Cycle {cycle_idx} missing data for rank {base_rank}")
        splits = np.asarray(parse_int_list(row_data['input_split_sizes'], base_world_size), dtype=np.int64)
        base_matrix[base_rank, :] = splits

    # Tile to base_world_size x world_size, then further tile to world_size x world_size
    row_indices = np.mod(np.arange(world_size), base_world_size)
    col_indices = np.mod(np.arange(world_size), base_world_size)
    comm_matrix = base_matrix[row_indices][:, col_indices]

    # Map rank to the corresponding row in the base matrix to obtain the hidden dimension
    base_rank = rank % base_world_size
    base_row = cycle.get(base_rank)
    if base_row is None:
        raise KeyError(f"Base rank {base_rank} not found in cycle {cycle_idx}")

    in_rows_base, hidden = parse_shape(base_row['input_shape'])
    out_rows_base, hidden2 = parse_shape(base_row['output_shape'])
    if hidden != hidden2:
        raise ValueError("input/output hidden size mismatch")

    sendcounts_rows = comm_matrix[rank, :].astype(np.int64).tolist()
    recvcounts_rows = comm_matrix[:, rank].astype(np.int64).tolist()

    # in_rows/out_rows are computed directly by summing the tiled matrix
    in_rows = int(sum(sendcounts_rows))
    out_rows = int(sum(recvcounts_rows))

    # After tiling, row and column sums should be multiples of the base sums; perform a robustness check
    if in_rows % in_rows_base != 0:
        raise ValueError(f"in_rows {in_rows} is not a multiple of base rows {in_rows_base}")
    if out_rows % out_rows_base != 0:
        raise ValueError(f"out_rows {out_rows} is not a multiple of base rows {out_rows_base}")

    return in_rows, hidden, out_rows, sendcounts_rows, recvcounts_rows, comm_matrix


    # The old MOE workload generation function has been removed; it is no longer needed

def scale_hidden_to_size(in_rows: int,
                         out_rows: int,
                         hidden: int,
                         target_size_mb: Union[int, float],
                         dtype_bytes: int) -> int:
    """
    Scale the hidden dimension based on the target average message size (MB).
    Ensures the scaled tensor size meets the target size, provided the target size
    and base size are multiples of each other.
    """
    if dtype_bytes <= 0:
        raise ValueError(f"Invalid dtype_bytes {dtype_bytes}")
    target_bytes = int(target_size_mb) * (1024 ** 2)
    if target_bytes <= 0:
        raise ValueError(f"Invalid target size {target_size_mb}")

    candidates = [
        ("input", in_rows, hidden),
        ("output", out_rows, hidden),
    ]
    for name, rows, current_hidden in candidates:
        if rows == 0 or current_hidden == 0:
            continue
        base_bytes = rows * current_hidden * dtype_bytes
        if base_bytes == 0:
            continue

        if target_bytes >= base_bytes and target_bytes % base_bytes == 0:
            scale = target_bytes // base_bytes
            scaled_hidden = current_hidden * scale
            if scaled_hidden <= 0:
                raise ValueError(f"Scaled hidden non-positive for {name}")
            return scaled_hidden

        if base_bytes >= target_bytes and base_bytes % target_bytes == 0:
            scale = base_bytes // target_bytes
            if current_hidden % scale != 0:
                continue
            scaled_hidden = current_hidden // scale
            if scaled_hidden <= 0:
                raise ValueError(f"Scaled hidden non-positive for {name}")
            return scaled_hidden

    raise ValueError(f"Cannot scale hidden for target size {target_size_mb} MB "
                     f"with in_rows={in_rows}, out_rows={out_rows}, hidden={hidden}")

def create_gold_standard_alltoallv(input_tensor: torch.Tensor,
                                   sendcounts: List[Union[int, np.int64]],
                                   recvcounts: List[Union[int, np.int64]],
                                   library: str):
    """
    Use torch.distributed.all_to_all_single as the unified gold standard.
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
                        choices=["spread_out", "simple_2D"],
                        default="spread_out",
                        help="Choose the all-to-allv algorithm for PCCL")
    # Fixed to bf16; pattern/size loops are no longer needed
    parser.add_argument("--profile",
                        action="store_true",
                        help="profile the alltoallv")

    # sizes = np.array([16, 32, 64, 128, 256, 512])
    sizes = np.array([64])

    args = parser.parse_args()

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
    elif args.library == "mpi":
        pg = MPI.COMM_WORLD
    elif args.library == "nccl":
        pg = None  # None is mapped to comm-world in torch.dist + nccl

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    library_label_parts = [args.library]
    if args.library == "pccl":
        backend_suffix = "cpp" if args.use_pccl_cpp_backend else "py"
        library_label_parts.extend([backend_suffix, "2D", algorithm])
    elif args.library == "nccl" and args.use_pccl_cpp_backend:
        library_label_parts.append("cpp")
    library_label = "_".join(library_label_parts)

    # Prepare output file path
    data_folder = f"./data/all_to_all/{args.machine}"
    os.makedirs(data_folder, exist_ok=True)
    csv_filename = os.path.join(
        data_folder,
        f"gpus_{gpu_count}_slurm_{slurm_job_id}.csv"
    )

    # Read matrix.csv and parse all cycles of data
    csv_path = os.path.join(os.path.dirname(__file__), "moe_balance.csv")
    # csv_path = os.path.join(os.path.dirname(__file__), "moe_alltoall_16_5000.csv")
    rank = dist.get_rank()

    # Parse all cycles
    cycles = parse_all_cycles_from_csv(csv_path, world_size, rank)
    total_cycles = len(cycles)

    if rank == 0:
        print(f"Found {total_cycles} cycles in matrix.csv")

    # Fix dtype to bf16
    dtype = torch.bfloat16
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    device = f"cuda:{rank % torch.cuda.device_count()}"

    size_values = [int(s) for s in (sizes.tolist() if hasattr(sizes, "tolist") else sizes)]
    elapsed_time_by_size = {size: [] for size in size_values}
    correctness_flags_by_size = {size: [] for size in size_values}
    elapsed_time_by_cycle = {size: {} for size in size_values}
    peak_to_mean_by_cycle = {size: {} for size in size_values}

    # Create profile object (if profiling is enabled)
    prof = None
    if args.profile:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()

    cycle_payloads = []
    for cycle_idx in range(total_cycles):
        try:
            in_rows, hidden, out_rows, sendcounts_rows, recvcounts_rows, comm_matrix = get_cycle_data_for_rank(
                cycles, rank, cycle_idx, world_size
            )
            cycle_payloads.append(
                {
                    "cycle_idx": cycle_idx,
                    "in_rows": in_rows,
                    "hidden": hidden,
                    "out_rows": out_rows,
                    "sendcounts_rows": [int(x) for x in sendcounts_rows],
                    "recvcounts_rows": [int(x) for x in recvcounts_rows],
                    "comm_matrix": comm_matrix,
                }
            )
        except Exception as exc:
            cycle_payloads.append(None)
            if rank == 0:
                print(f"Skipping cycle {cycle_idx + 1} due to error: {exc}")

    for size_mb in size_values:
        if rank == 0:
            print(f"\n=== Size {size_mb}MB ===")

        for payload in cycle_payloads:
            if payload is None:
                continue

            cycle_idx = payload["cycle_idx"]
            in_rows = payload["in_rows"]
            hidden = payload["hidden"]
            out_rows = payload["out_rows"]
            sendcounts_rows_list = payload["sendcounts_rows"]
            recvcounts_rows_list = payload["recvcounts_rows"]
            comm_matrix = payload["comm_matrix"]

            send_totals = comm_matrix.sum(axis=1).astype(np.float64)
            recv_totals = comm_matrix.sum(axis=0).astype(np.float64)

            def _safe_mean(vec):
                return float(np.mean(vec)) if vec.size > 0 else 0.0

            mean_send = _safe_mean(send_totals)
            mean_recv = _safe_mean(recv_totals)
            p2m_send = float(np.max(send_totals)) / mean_send if mean_send > 0 else 0.0
            p2m_recv = float(np.max(recv_totals)) / mean_recv if mean_recv > 0 else 0.0
            peak_to_mean = max(p2m_send, p2m_recv)
            peak_to_mean_by_cycle[size_mb][cycle_idx] = peak_to_mean

            try:
                hidden_scaled = scale_hidden_to_size(in_rows, out_rows, hidden, size_mb, dtype_bytes)
            except ValueError as exc:
                if rank == 0:
                    print(f"  Cycle {cycle_idx + 1}: skipping (size={size_mb}MB) -> {exc}")
                continue

            if rank == 0:
                print(
                    f"  Cycle {cycle_idx + 1}: rows_in={in_rows}, rows_out={out_rows}, "
                    f"hidden={hidden} -> hidden_scaled={hidden_scaled}"
                )

            input_tensor = torch.randn((in_rows, hidden_scaled), dtype=dtype, device=device)
            output_tensor = torch.zeros((out_rows, hidden_scaled), dtype=dtype, device=device)

            flat_sendcounts = [count * hidden_scaled for count in sendcounts_rows_list]
            flat_recvcounts = [count * hidden_scaled for count in recvcounts_rows_list]

            output_tensor_gold = None
            if args.test:
                gold_input = input_tensor.view(-1)
                output_tensor_gold = create_gold_standard_alltoallv(
                    gold_input,
                    flat_sendcounts,
                    flat_recvcounts,
                    args.library,
                )

            kwargs = {"use_pccl_cpp_backend": args.use_pccl_cpp_backend}
            if args.library == "pccl":
                kwargs["algorithm"] = algorithm
                kwargs["comm_matrix_rows"] = comm_matrix

            if args.library == "pccl" and algorithm == "spread_out":
                func = all_to_allv_2D
                out_arg = output_tensor
                in_arg = input_tensor
                sc = sendcounts_rows_list
                rc = recvcounts_rows_list
            elif args.library == "pccl" and algorithm == "simple_2D":
                func = all_to_allv_simple_2D
                out_arg = output_tensor
                in_arg = input_tensor
                sc = sendcounts_rows_list
                rc = recvcounts_rows_list
            elif args.library == "nccl" and not args.use_pccl_cpp_backend:
                func = _all_to_allv
                out_arg = output_tensor
                in_arg = input_tensor
                sc = sendcounts_rows_list
                rc = recvcounts_rows_list
            else:
                func = _all_to_allv
                out_arg = output_tensor.view(-1)
                in_arg = input_tensor.view(-1)
                sc = flat_sendcounts
                rc = flat_recvcounts

            try:
                elapsed_time = time_something(
                    func,
                    out_arg,
                    in_arg,
                    sc,
                    rc,
                    group=pg,
                    warmup_iters=5,
                    timed_iters=20,
                    prof=prof,
                    **kwargs,
                )
            except Exception as exc:
                if rank == 0:
                    print(f"    Error executing cycle {cycle_idx + 1} (size={size_mb}MB): {exc}")
                continue

            success = True
            if args.test:
                local_pass = allclose(output_tensor.view(-1), output_tensor_gold)
                local_result = 1 if local_pass else 0
                global_result = MPI.COMM_WORLD.allreduce(local_result, MPI.MIN)
                success = (global_result == 1)
                if success:
                    if rank == 0:
                        print("    ✓ correctness passed")
                else:
                    if rank == 0:
                        print("    ✗ correctness failed")
                    if not local_pass:
                        max_abs_diff = torch.max(torch.abs(output_tensor.view(-1) - output_tensor_gold)).item()
                        max_rel_diff = torch.max(
                            torch.abs(
                                (output_tensor.view(-1) - output_tensor_gold) / (output_tensor_gold + 1e-8)
                            )
                        ).item()
                        print(f"Rank {rank} - Max absolute difference: {max_abs_diff}")
                        print(f"Rank {rank} - Max relative difference: {max_rel_diff}")

            if success:
                elapsed_time_by_size[size_mb].append(elapsed_time)
                elapsed_time_by_cycle[size_mb].setdefault(cycle_idx, []).append(elapsed_time)
                if rank == 0:
                    print(f"    Elapsed: {elapsed_time:.2f}ms")

            correctness_flags_by_size[size_mb].append(success)
        #     break
        # break

    if prof is not None:
        prof.stop()
        if rank == 0:
            trace_name = f"profile_{library_label}.json"
            prof.export_chrome_trace(trace_name)
            print(f"Profile saved to {trace_name}")

    if rank == 0:
        existing_rows: Dict[tuple, Dict[str, str]] = {}
        existing_fieldnames: List[str] = []
        if os.path.exists(csv_filename):
            with open(csv_filename, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames or []
                for row in reader:
                    key = (row.get("world_size", ""), row.get("size", ""))
                    existing_rows[key] = row

        print("\n=== Benchmark Summary ===")
        for size_mb in size_values:
            times = elapsed_time_by_size[size_mb]
            successes = correctness_flags_by_size[size_mb]
            total_attempts = len(successes)
            successful_runs = sum(1 for passed in successes if passed)

            if times:
                mean_time = float(np.mean(times))
                print(
                    f"Size {size_mb}MB: mean={mean_time:.2f}ms "
                    f"(successful {len(times)}/{total_attempts})"
                )
                key = (str(world_size), str(size_mb))
                row = existing_rows.get(key, {"world_size": str(world_size), "size": str(size_mb)})
                row["world_size"] = str(world_size)
                row["size"] = str(size_mb)
                row[library_label] = f"{mean_time:.4f}"
                existing_rows[key] = row
            else:
                if total_attempts > 0:
                    print(f"Size {size_mb}MB: no successful samples ({successful_runs}/{total_attempts} passed)")
                else:
                    print(f"Size {size_mb}MB: not attempted")

        if existing_rows:
            fieldnames: List[str] = []
            seen_cols = set()

            def add_column(col: str):
                if col and col not in seen_cols:
                    fieldnames.append(col)
                    seen_cols.add(col)

            add_column("world_size")
            add_column("size")

            for col in existing_fieldnames:
                add_column(col)

            add_column(library_label)

            for row in existing_rows.values():
                for col in row.keys():
                    add_column(col)

            sorted_rows = sorted(
                existing_rows.values(),
                key=lambda r: (
                    int(r.get("world_size", "0")) if r.get("world_size", "").isdigit() else r.get("world_size", ""),
                    int(r.get("size", "0")) if r.get("size", "").isdigit() else r.get("size", ""),
                ),
            )

            with open(csv_filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sorted_rows)

            print("\n=== Final Results Table ===")
            # Print header
            print(",".join(fieldnames))
            # Print each data row
            for row in sorted_rows:
                row_values = [row.get(col, "") for col in fieldnames]
                print(",".join(row_values))

            print(f"Results saved to {csv_filename}")
        else:
            print("No results to save.")

        # Write per-cycle performance results
        csv_cycles_filename = os.path.join(
            data_folder,
            f"gpus_{gpu_count}_slurm_{slurm_job_id}_cycles.csv"
        )

        existing_cycle_rows: Dict[tuple, Dict[str, str]] = {}
        existing_cycle_fieldnames: List[str] = []
        if os.path.exists(csv_cycles_filename):
            with open(csv_cycles_filename, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_cycle_fieldnames = reader.fieldnames or []
                for row in reader:
                    key = (
                        row.get("world_size", ""),
                        row.get("size", ""),
                        row.get("cycle", ""),
                    )
                    existing_cycle_rows[key] = row

        # print("\n=== Per-Cycle Results ===")
        for size_mb in size_values:
            cycle_times = elapsed_time_by_cycle.get(size_mb, {})
            for cycle_idx, measurements in cycle_times.items():
                if not measurements:
                    continue
                mean_time = float(np.mean(measurements))
                cycle_label = str(cycle_idx + 1)
                # print(
                #     f"Size {size_mb}MB, Cycle {cycle_label}: mean={mean_time:.2f}ms"
                # )
                key = (str(world_size), str(size_mb), cycle_label)
                row = existing_cycle_rows.get(
                    key,
                    {
                        "world_size": str(world_size),
                        "size": str(size_mb),
                        "cycle": cycle_label,
                    },
                )
                peak_to_mean_val = peak_to_mean_by_cycle.get(size_mb, {}).get(cycle_idx)
                if peak_to_mean_val is not None:
                    row["peak_to_mean"] = f"{peak_to_mean_val:.4f}"
                row[library_label] = f"{mean_time:.4f}"
                existing_cycle_rows[key] = row

        if existing_cycle_rows:
            cycle_fieldnames: List[str] = []
            seen_cols = set()

            def add_cycle_column(col: str):
                if col and col not in seen_cols:
                    cycle_fieldnames.append(col)
                    seen_cols.add(col)

            add_cycle_column("world_size")
            add_cycle_column("size")
            add_cycle_column("cycle")
            add_cycle_column("peak_to_mean")

            for col in existing_cycle_fieldnames:
                add_cycle_column(col)

            add_cycle_column(library_label)

            for row in existing_cycle_rows.values():
                for col in row.keys():
                    add_cycle_column(col)

            sorted_cycle_rows = sorted(
                existing_cycle_rows.values(),
                key=lambda r: (
                    int(r.get("world_size", "0")) if r.get("world_size", "").isdigit() else r.get("world_size", ""),
                    int(r.get("size", "0")) if r.get("size", "").isdigit() else r.get("size", ""),
                    int(r.get("cycle", "0")) if r.get("cycle", "").isdigit() else r.get("cycle", ""),
                ),
            )

            with open(csv_cycles_filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=cycle_fieldnames)
                writer.writeheader()
                writer.writerows(sorted_cycle_rows)

            print(f"Per-cycle results saved to {csv_cycles_filename}")
        else:
            print("No per-cycle results to save.")
