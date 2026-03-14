import torch
import torch.distributed as dist
from mpi4py import MPI
import numpy as np
from typing import Optional, Union, List, Dict, Tuple
from .request import Request
from .process_groups import ProcessGroups
# Note: To avoid circular dependencies, the sameinput/sameoutput implementations
# are not imported at the module level; they are lazily imported inside conditional branches.

def _all_to_allv(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    sendcounts: List[Union[int, np.int64]],
    recvcounts: List[Union[int, np.int64]],
    group: Optional[Union[dist.ProcessGroup, MPI.Comm]] = None,
    async_op: bool = False,
    use_pccl_cpp_backend: bool = False,
    algorithm: str = "pairwise_sendrecv",
    send_displs: Optional[List[Union[int, np.int64]]] = None,
    recv_displs: Optional[List[Union[int, np.int64]]] = None
) -> Optional[Request]:

    # Case 1: Use PyTorch distributed process group or default group
    if group is None or isinstance(group, dist.ProcessGroup):
        if use_pccl_cpp_backend:

            # Use PCCL C++ backend's NCCL point-to-point implementation for optimal performance
            import pccl as pccl_cpp
            from .nccl_comm import CommHandler
            nccl_comm = CommHandler.get_communicator_from_process_group(group)

            # Compute displacement arrays - MPI/NCCL needs to know the starting position of each data block in the buffer
            if send_displs is None:
                send_displs = [0]  # Displacement array for send data
                for i in range(len(sendcounts) - 1):
                    send_displs.append(send_displs[-1] + sendcounts[i])  # Cumulative offset

            if recv_displs is None:
                recv_displs = [0]  # Displacement array for receive data
                for i in range(len(recvcounts) - 1):
                    recv_displs.append(recv_displs[-1] + recvcounts[i])  # Cumulative offset

            # Call the C++ implemented NCCL point-to-point communication
            request = pccl_cpp.all_to_allv_nccl_p2p(output_tensor, input_tensor, sendcounts,
                recvcounts, send_displs, recv_displs, nccl_comm.comm, nccl_comm.rank, nccl_comm.nranks
            )
            return request
        else:
            # raise ValueError("use_pccl_cpp_backend must be True")
            request = dist.all_to_all_single(output_tensor, input_tensor, recvcounts, sendcounts, group, async_op)
            return request

    # Case 2: Use MPI communicator - suitable for cross-node communication
    elif isinstance(group, MPI.Comm):
        # if 0:
        #     import pccl as pccl_cpp

        #     # Compute displacement arrays - MPI needs to know the starting position of each data block in the buffer
        #     send_displs = [0]  # Displacement array for send data
        #     for i in range(len(sendcounts) - 1):
        #         send_displs.append(send_displs[-1] + sendcounts[i])  # Cumulative offset

        #     recv_displs = [0]  # Displacement array for receive data
        #     for i in range(len(recvcounts) - 1):
        #         recv_displs.append(recv_displs[-1] + recvcounts[i])  # Cumulative offset
        #     request = pccl_cpp.all_to_allv_mpi(output_tensor, input_tensor, sendcounts, recvcounts,
        #                                       send_displs, recv_displs, group, algorithm)
        #     return request

        if async_op:
            # Async operation - use Ialltoallv
            # Synchronize CUDA stream to ensure GPU data is ready, avoiding race conditions
            # torch.cuda.current_stream().synchronize()

            input_tensor = input_tensor.contiguous()
            output_tensor = output_tensor.contiguous()

            # Check if we have CUDA-aware MPI support
            has_cuda_mpi = hasattr(MPI, 'CUDA_FLOAT') or hasattr(MPI, 'GPU_FLOAT')

            # Determine MPI data type based on tensor dtype
            if input_tensor.dtype == torch.float32:
                if has_cuda_mpi and hasattr(MPI, 'CUDA_FLOAT'):
                    mpi_dtype = MPI.CUDA_FLOAT
                elif has_cuda_mpi and hasattr(MPI, 'GPU_FLOAT'):
                    mpi_dtype = MPI.GPU_FLOAT
                else:
                    mpi_dtype = MPI.FLOAT
            elif input_tensor.dtype == torch.bfloat16:
                mpi_dtype = MPI.BYTE  # bfloat16 not directly supported, use BYTE
            elif input_tensor.dtype == torch.float16:
                mpi_dtype = MPI.BYTE  # float16 not directly supported, use BYTE
            elif input_tensor.dtype == torch.int64:
                mpi_dtype = MPI.LONG_LONG
            else:
                mpi_dtype = MPI.BYTE

            # Convert element counts to byte counts for MPI.BYTE
            if mpi_dtype == MPI.BYTE:
                # Calculate bytes per element
                bytes_per_element = input_tensor.element_size()
                sendcounts_bytes = [count * bytes_per_element for count in sendcounts]
                recvcounts_bytes = [count * bytes_per_element for count in recvcounts]
            else:
                sendcounts_bytes = sendcounts
                recvcounts_bytes = recvcounts

            # Compute displacement arrays needed by MPI - same logic as NCCL implementation
            send_displs = [0]
            for i in range(len(sendcounts_bytes) - 1):
                send_displs.append(send_displs[-1] + sendcounts_bytes[i])

            recv_displs = [0]
            for i in range(len(recvcounts_bytes) - 1):
                recv_displs.append(recv_displs[-1] + recvcounts_bytes[i])

            # Async GPU communication - use Ialltoallv
            if input_tensor.dtype == torch.float32:
                request = group.Ialltoallv(
                    [input_tensor, sendcounts_bytes, send_displs, mpi_dtype],
                    [output_tensor, recvcounts_bytes, recv_displs, mpi_dtype]
                )
            elif input_tensor.dtype == torch.float16:
                request = group.Ialltoallv(
                    [input_tensor, sendcounts_bytes, send_displs, MPI.BYTE],
                    [output_tensor, recvcounts_bytes, recv_displs, MPI.BYTE]
                )
            elif input_tensor.dtype == torch.int64:
                request = group.Ialltoallv(
                    [input_tensor, sendcounts_bytes, send_displs, MPI.LONG_LONG],
                    [output_tensor, recvcounts_bytes, recv_displs, MPI.LONG_LONG]
                )
            else:
                request = group.Ialltoallv(
                    [input_tensor, sendcounts_bytes, send_displs, MPI.BYTE],
                    [output_tensor, recvcounts_bytes, recv_displs, MPI.BYTE]
                )
        else:
            # Synchronize CUDA stream to ensure GPU data is ready, avoiding race conditions
            # torch.cuda.current_stream().synchronize()

            input_tensor = input_tensor.contiguous()
            output_tensor = output_tensor.contiguous()

            # Check if we have CUDA-aware MPI support
            has_cuda_mpi = hasattr(MPI, 'CUDA_FLOAT') or hasattr(MPI, 'GPU_FLOAT')

            # Determine MPI data type based on tensor dtype
            if input_tensor.dtype == torch.float32:
                if has_cuda_mpi and hasattr(MPI, 'CUDA_FLOAT'):
                    mpi_dtype = MPI.CUDA_FLOAT
                elif has_cuda_mpi and hasattr(MPI, 'GPU_FLOAT'):
                    mpi_dtype = MPI.GPU_FLOAT
                else:
                    mpi_dtype = MPI.FLOAT
            elif input_tensor.dtype == torch.bfloat16:
                mpi_dtype = MPI.BYTE  # bfloat16 not directly supported, use BYTE
            elif input_tensor.dtype == torch.float16:
                mpi_dtype = MPI.BYTE  # float16 not directly supported, use BYTE
            elif input_tensor.dtype == torch.int64:
                mpi_dtype = MPI.LONG_LONG
            else:
                mpi_dtype = MPI.BYTE

            # Convert element counts to byte counts for MPI.BYTE
            if mpi_dtype == MPI.BYTE:
                # Calculate bytes per element
                bytes_per_element = input_tensor.element_size()
                sendcounts_bytes = [count * bytes_per_element for count in sendcounts]
                recvcounts_bytes = [count * bytes_per_element for count in recvcounts]
            else:
                sendcounts_bytes = sendcounts
                recvcounts_bytes = recvcounts

            # Compute displacement arrays needed by MPI - same logic as NCCL implementation
            send_displs = [0]
            for i in range(len(sendcounts_bytes) - 1):
                send_displs.append(send_displs[-1] + sendcounts_bytes[i])

            recv_displs = [0]
            for i in range(len(recvcounts_bytes) - 1):
                recv_displs.append(recv_displs[-1] + recvcounts_bytes[i])

            # Try direct GPU communication
            if input_tensor.dtype == torch.float32:
                request = group.Alltoallv(
                    [input_tensor, sendcounts_bytes, send_displs, mpi_dtype],
                    [output_tensor, recvcounts_bytes, recv_displs, mpi_dtype]
                )
            elif input_tensor.dtype == torch.float16:
                request = group.Alltoallv(
                    [input_tensor, sendcounts_bytes, send_displs, MPI.BYTE],
                    [output_tensor, recvcounts_bytes, recv_displs, MPI.BYTE]
                )
            elif input_tensor.dtype == torch.int64:
                request = group.Alltoallv(
                    [input_tensor, sendcounts_bytes, send_displs, MPI.LONG_LONG],
                    [output_tensor, recvcounts_bytes, recv_displs, MPI.LONG_LONG]
                )
            else:
                request = group.Alltoallv(
                    [input_tensor, sendcounts_bytes, send_displs, MPI.BYTE],
                    [output_tensor, recvcounts_bytes, recv_displs, MPI.BYTE]
                )
    else:
        raise TypeError(
            f"Unsupported group type: {type(group)}. "
            "Expected torch.distributed.ProcessGroup or mpi4py.MPI.Comm."
        )
    return request

def all_to_allv_2D(output_tensor: torch.Tensor,
                   input_tensor: torch.Tensor,
                   sendcounts: List[Union[int, np.int64]],
                   recvcounts: List[Union[int, np.int64]],
                   group: Optional['ProcessGroups'] = None,
                   async_op: bool = False,
                   use_pccl_cpp_backend: bool = False,
                   algorithm: str = "spread_out",
                   overload_threshold: float = 1.0,
                   split_threshold: float = 0.0,
                   comm_matrix_rows: Optional[Union[List[List[int]], np.ndarray]] = None):
    """
    Hierarchical 2D All-to-Allv implementation optimized for multi-node multi-GPU systems.

    This function implements a four-phase hierarchical all-to-allv algorithm that leverages
    the 2D topology (intra-node GPUs × inter-node) to minimize communication overhead and
    improve load balancing compared to naive all-to-allv implementations.

    Algorithm Overview:
    -------------------
    Phase 1: Data Reorganization (by outer_group)
        - Reorganize input data from [rank0_data|rank1_data|...] layout
        - To: [outer_group0_data|outer_group1_data|...] layout
        - Each outer_group contains ranks with same intra-node position across all nodes

    Phase 2: Intra-Node All-to-Allv
        - Redistribute data within each node using NCCL
        - Each intra-node rank acts as a "proxy" for one outer_group
        - Proxy ranks aggregate all data destined for their outer_group

    Phase 3: Data Reorganization (by destination node)
        - Reorganize data from [src_rank0_data|src_rank1_data|...] layout
        - To: [dest_node0_data|dest_node1_data|...] layout
        - Prepares data for efficient inter-node communication

    Phase 4: Inter-Node All-to-Allv
        - Transfer data between nodes using NCCL
        - Each proxy rank sends/receives data for its outer_group
        - Output is naturally organized by source rank due to Phase 3 reorganization

    Returns:
    --------
    None (output written to output_tensor)
    """

    assert not async_op, "Non blocking version not implemented"

    # Tensor dimension validation and element count calculation
    # Supports both 1D tensors and 2D tensors (rows × features)
    # For 2D tensors: counts represent number of rows, scaled by row_size for element counts
    if input_tensor.dim() == 2:
        assert output_tensor.dim() == 2, "When input is 2D, output must be 2D as well"
        assert input_tensor.size(1) == output_tensor.size(1), "2D input/output must have the same feature dimension"
        row_size = int(input_tensor.size(1))  # Feature dimension size
    elif input_tensor.dim() == 1 and output_tensor.dim() == 1:
        row_size = 1  # 1D tensor: each element is a "row"
    else:
        raise AssertionError("all_to_allv_2D supports 1D tensors or 2D tensors with matching second dimension")

    # Flatten tensors for unified processing
    input_flat = input_tensor.view(-1)
    output_flat = output_tensor.view(-1)

    # Convert counts from rows to elements
    sendcounts_elems = [int(c) * row_size for c in sendcounts]
    recvcounts_elems = [int(c) * row_size for c in recvcounts]

    # Extract system topology information
    intra_node_group_size, inter_node_group_size = group.get_world_size()
    # intra_node_group_size: Number of GPUs per node (e.g., 4)
    # inter_node_group_size: Number of nodes (e.g., 16)
    world_size = intra_node_group_size * inter_node_group_size  # Total number of GPUs

    # Validate input consistency and completeness
    assert len(sendcounts) == world_size, f"sendcounts length {len(sendcounts)} != world_size {world_size}"
    assert len(recvcounts) == world_size, f"recvcounts length {len(recvcounts)} != world_size {world_size}"
    assert sum(sendcounts_elems) == input_flat.numel(), f"Sum of sendcounts (elements) {sum(sendcounts_elems)} != input size {input_flat.numel()}"
    assert sum(recvcounts_elems) == output_flat.numel(), f"Sum of recvcounts (elements) {sum(recvcounts_elems)} != output size {output_flat.numel()}"

    # Determine current process position in 2D grid
    rank = dist.get_rank()  # Global rank ID
    my_intra_rank = rank % intra_node_group_size  # Intra-node GPU ID (e.g., 0-3)
    my_node_idx = rank // intra_node_group_size    # Node ID (e.g., 0-15)

    # Construct communication matrix with element counts
    # comm_matrix[i][j] = number of elements rank i sends to rank j
    if comm_matrix_rows is not None:
        full_cm_rows = np.asarray(comm_matrix_rows, dtype=np.int64)
        assert full_cm_rows.shape == (world_size, world_size), (
            f"comm_matrix_rows shape {full_cm_rows.shape} != ({world_size}, {world_size})"
        )
        comm_matrix = (full_cm_rows * row_size).astype(np.int64)  # Scale by row_size
    else:
        raise ValueError("comm_matrix_rows is required")



    # Compute per-destination prefix offsets for our row (used by stage 1 send displs)
    send_offsets = np.zeros(world_size + 1, dtype=np.int64)
    for i in range(world_size):
        send_offsets[i + 1] = send_offsets[i] + int(comm_matrix[rank, i])

    # Create streams and events for overlapped execution
    s_inter1 = torch.cuda.Stream()
    s_reorg1 = torch.cuda.Stream()
    s_intra = torch.cuda.Stream()
    s_reorg3 = torch.cuda.Stream()
    s_inter2 = torch.cuda.Stream()
    e_reorg1_done = torch.cuda.Event(enable_timing=False)
    e_intra_done = torch.cuda.Event(enable_timing=False)
    e_reorg3_done = torch.cuda.Event(enable_timing=False)

    # ----------------------------------------------------------------------------------
    # Early inter-node exchange (stage 1): send each GPU's own outer-group data first.
    # This overlaps with intra-node redistribution and reduces the second inter-node load.
    # We place received data directly into final output buffer at correct global offsets.
    # ----------------------------------------------------------------------------------
    with torch.cuda.stream(s_inter1):
        # Pre-compute global recv prefix offsets for this rank (column-wise prefix over sources)
        recv_offsets_prefix = np.zeros(world_size + 1, dtype=np.int64)
        for src_rank in range(world_size):
            recv_offsets_prefix[src_rank + 1] = recv_offsets_prefix[src_rank] + int(comm_matrix[src_rank, rank])

        # Build counts and displacements for stage 1: only same intra-rank sources per node
        stage1_sendcounts = [0] * inter_node_group_size
        stage1_recvcounts = [0] * inter_node_group_size
        stage1_send_displs = [0] * inter_node_group_size
        stage1_recv_displs = [0] * inter_node_group_size
        for node in range(inter_node_group_size):
            dest_rank = node * intra_node_group_size + my_intra_rank
            src_rank_same_intra = node * intra_node_group_size + my_intra_rank
            # amount we send to that node (our row -> that rank with same intra)
            send_cnt = int(comm_matrix[rank, dest_rank])
            # amount we receive from that node (their same-intra rank -> us)
            recv_cnt = int(comm_matrix[src_rank_same_intra, rank])
            stage1_sendcounts[node] = send_cnt
            stage1_recvcounts[node] = recv_cnt
            # send offset within input_flat row (prefix over destination ranks)
            stage1_send_displs[node] = int(send_offsets[dest_rank])
            # recv offset within final output buffer (prefix over source ranks)
            stage1_recv_displs[node] = int(recv_offsets_prefix[src_rank_same_intra])

        # Launch stage 1 inter-node all_to_allv asynchronously to overlap with intra-node phase
        stage1_req = _all_to_allv(
            output_flat,
            input_flat,
            stage1_sendcounts,
            stage1_recvcounts,
            group.get_outer_group(),
            async_op=True,
            use_pccl_cpp_backend=use_pccl_cpp_backend,
            send_displs=stage1_send_displs,
            recv_displs=stage1_recv_displs
        )

    # ==================================================================================
    # PHASE 1: Data Reorganization by Outer Group (Optimized with Streams + Pre-computed Offsets)
    # ==================================================================================
    # Purpose: Transform data layout from rank-major to outer_group-major order
    #
    # Concept: Outer Group
    # - An "outer group" consists of ranks with the same intra-node position across all nodes
    # - For example, outer_group=0 contains: rank0, rank4, rank8, ..., rank(4*(N-1))
    # - This grouping enables efficient hierarchical communication
    #
    # Data Layout Transformation:
    # - Input layout:  [data_for_rank0 | data_for_rank1 | ... | data_for_rank(W-1)]
    # - Output layout: [data_for_outer_group0 | data_for_outer_group1 | ... | data_for_outer_group(G-1)]
    #
    # Where "data_for_outer_group_i" contains:
    #   [data_for_rank_i | data_for_rank_(G+i) | data_for_rank_(2G+i) | ... | data_for_rank_(G*(N-1)+i)]
    #   (G = intra_node_group_size, N = inter_node_group_size)

    # Pre-compute cumulative send offsets (eliminates O(world_size²) repeated sums)
    send_offsets = np.zeros(world_size + 1, dtype=np.int64)
    for i in range(world_size):
        send_offsets[i + 1] = send_offsets[i] + int(comm_matrix[rank, i])

    input_permuted = torch.empty_like(input_flat)

    # Phase 1 reorg on its own stream and signal when done
    with torch.cuda.stream(s_reorg1):
        write_offset = 0
        for outer_group_id in range(intra_node_group_size):
            for dest_node in range(inter_node_group_size):
                dest_rank = dest_node * intra_node_group_size + outer_group_id
                count = int(comm_matrix[rank, dest_rank])
                if count > 0:
                    read_offset = send_offsets[dest_rank]
                    input_permuted[write_offset:write_offset + count].copy_(
                        input_flat[read_offset:read_offset + count], non_blocking=True
                    )
                    write_offset += count
        e_reorg1_done.record(s_reorg1)

    # ==================================================================================
    # PHASE 2: Intra-Node All-to-Allv (Proxy Aggregation)
    # ==================================================================================
    # Purpose: Redistribute data within each node so that each GPU acts as a "proxy"
    #          for one outer_group, aggregating all data destined for that group
    #
    # Proxy Concept:
    # - Within each node, GPU i (where i = rank % intra_node_group_size) serves as the
    #   proxy for outer_group i
    # - This proxy will handle all inter-node communication for outer_group i
    # - Example: rank 0 proxies for outer_group 0 (ranks 0, 4, 8, 12, ...)
    #            rank 1 proxies for outer_group 1 (ranks 1, 5, 9, 13, ...)
    #
    # Data Flow:
    # - Each GPU sends its outer_group i data to intra-node GPU i
    # - Each GPU i receives all outer_group i data from all intra-node GPUs
    # - Uses fast intra-node communication (NCCL via NVLink/Infinity Fabric)

    # Calculate sendcounts for intra-node all_to_allv
    # intra_sendcounts[i] = amount of data to send to intra-node GPU i
    #                     = all data destined for outer_group i
    intra_sendcounts = [0] * intra_node_group_size
    for dest_node in range(inter_node_group_size):
        for local_rank in range(intra_node_group_size):
            dest_rank = dest_node * intra_node_group_size + local_rank
            count = int(comm_matrix[rank, dest_rank])
            # local_rank corresponds to outer_group ID
            intra_sendcounts[local_rank] += count

    # Calculate recvcounts for intra-node all_to_allv
    # intra_recvcounts[i] = amount of data to receive from intra-node GPU i
    # As a proxy for outer_group = my_intra_rank, we receive all data destined
    # for that outer_group from all intra-node GPUs
    intra_recvcounts = [0] * intra_node_group_size
    for src_local_rank in range(intra_node_group_size):
        src_rank = my_node_idx * intra_node_group_size + src_local_rank
        # Sum all data from src_rank destined for outer_group = my_intra_rank (across all nodes)
        for dest_node in range(inter_node_group_size):
            dest_rank = dest_node * intra_node_group_size + my_intra_rank
            count = int(comm_matrix[src_rank, dest_rank])
            intra_recvcounts[src_local_rank] += count

    # Execute intra-node all_to_allv using NCCL (fast intra-node communication)
    output_intermediate = torch.empty(sum(intra_recvcounts), dtype=input_flat.dtype, device=input_flat.device)
    # Intra-node all_to_allv on dedicated stream, wait Phase 1 completion
    with torch.cuda.stream(s_intra):
        s_intra.wait_event(e_reorg1_done)
        _all_to_allv(
            output_intermediate,
            input_permuted,
            intra_sendcounts,
            intra_recvcounts,
            group.get_inner_group(),
            async_op=False,
            use_pccl_cpp_backend=use_pccl_cpp_backend
        )
        e_intra_done.record(s_intra)

    # ==================================================================================
    # PHASE 3: Data Reorganization by Destination Node (Optimized with Pre-computed Offsets)
    # ==================================================================================
    # Purpose: Transform data layout from source-rank-major to destination-node-major order
    #          to prepare for efficient inter-node communication
    #
    # Current State (output_intermediate):
    # - Current rank (my_intra_rank) is serving as proxy for outer_group = my_intra_rank
    # - Data layout: [data_from_intra_rank0 | data_from_intra_rank1 | ... | data_from_intra_rank(G-1)]
    # - Each "data_from_intra_rank_i" contains:
    #   [dest_node0_data | dest_node1_data | ... | dest_node(N-1)_data]
    #   where each chunk is destined for outer_group = my_intra_rank on that node
    #
    # Conceptual View:
    # - output_intermediate can be viewed as: (intra_node, inter_node, variable_size_blocks)
    #
    # Target Layout:
    # - Materialize contiguous blocks per destination node directly into the stage-2 send buffer
    #   while skipping our own intra-rank slice (already handled by stage 1).
    #

    # Precompute offsets for Phase 3 and run copies on its own stream after intra completes
    recv_base_offsets = np.zeros(intra_node_group_size + 1, dtype=np.int64)
    for src_local_rank in range(intra_node_group_size):
        src_rank = my_node_idx * intra_node_group_size + src_local_rank
        total = 0
        for dn in range(inter_node_group_size):
            dr = dn * intra_node_group_size + my_intra_rank
            total += int(comm_matrix[src_rank, dr])
        recv_base_offsets[src_local_rank + 1] = recv_base_offsets[src_local_rank] + total

    recv_node_offsets = np.zeros((intra_node_group_size, inter_node_group_size + 1), dtype=np.int64)
    for src_local_rank in range(intra_node_group_size):
        src_rank = my_node_idx * intra_node_group_size + src_local_rank
        for dest_node in range(inter_node_group_size):
            dest_rank = dest_node * intra_node_group_size + my_intra_rank
            recv_node_offsets[src_local_rank, dest_node + 1] = \
                recv_node_offsets[src_local_rank, dest_node] + int(comm_matrix[src_rank, dest_rank])

    node_totals = np.zeros(inter_node_group_size, dtype=np.int64)
    inner_prefix = np.zeros((inter_node_group_size, intra_node_group_size + 1), dtype=np.int64)
    for dest_node in range(inter_node_group_size):
        dest_rank = dest_node * intra_node_group_size + my_intra_rank
        for lr in range(intra_node_group_size):
            src_rank_lr = my_node_idx * intra_node_group_size + lr
            cnt = int(comm_matrix[src_rank_lr, dest_rank])
            inner_prefix[dest_node, lr + 1] = inner_prefix[dest_node, lr] + cnt
        node_totals[dest_node] = inner_prefix[dest_node, intra_node_group_size]

    inter_sendcounts2 = [0] * inter_node_group_size
    inter_recvcounts2 = [0] * inter_node_group_size
    for dest_node in range(inter_node_group_size):
        my_cnt = int(inner_prefix[dest_node, my_intra_rank + 1] - inner_prefix[dest_node, my_intra_rank])
        inter_sendcounts2[dest_node] = int(node_totals[dest_node] - my_cnt)

    for src_node in range(inter_node_group_size):
        total = 0
        for lr in range(intra_node_group_size):
            if lr == my_intra_rank:
                continue
            src_rank_lr = src_node * intra_node_group_size + lr
            total += int(comm_matrix[src_rank_lr, rank])
        inter_recvcounts2[src_node] = total

    stage2_send_displs = [0] * inter_node_group_size
    acc = 0
    for dest_node in range(inter_node_group_size):
        stage2_send_displs[dest_node] = acc
        acc += inter_sendcounts2[dest_node]
    sendbuf2_size = int(acc)
    send_buffer = (
        torch.empty(sendbuf2_size, dtype=input_flat.dtype, device=input_flat.device)
        if sendbuf2_size > 0 else torch.empty(0, dtype=input_flat.dtype, device=input_flat.device)
    )
    send_write_positions = np.array(stage2_send_displs, dtype=np.int64)

    stage2_recv_displs = [0] * inter_node_group_size
    acc = 0
    for src_node in range(inter_node_group_size):
        stage2_recv_displs[src_node] = acc
        acc += inter_recvcounts2[src_node]
    recvbuf2_size = int(acc)
    recv_buffer = (
        torch.empty(recvbuf2_size, dtype=input_flat.dtype, device=input_flat.device)
        if recvbuf2_size > 0 else torch.empty(0, dtype=input_flat.dtype, device=input_flat.device)
    )

    with torch.cuda.stream(s_reorg3):
        s_reorg3.wait_event(e_intra_done)
        for dest_node in range(inter_node_group_size):
            for src_local_rank in range(intra_node_group_size):
                src_rank = my_node_idx * intra_node_group_size + src_local_rank
                dest_rank = dest_node * intra_node_group_size + my_intra_rank
                count = int(comm_matrix[src_rank, dest_rank])
                if count <= 0:
                    continue
                read_offset = recv_base_offsets[src_local_rank] + recv_node_offsets[src_local_rank, dest_node]
                if sendbuf2_size > 0 and src_local_rank != my_intra_rank:
                    out_off = int(send_write_positions[dest_node])
                    send_buffer[out_off:out_off + count].copy_(
                        output_intermediate[read_offset:read_offset + count], non_blocking=True
                    )
                    send_write_positions[dest_node] += count
        e_reorg3_done.record(s_reorg3)

    if sendbuf2_size > 0:
        for dest_node in range(inter_node_group_size):
            assert send_write_positions[dest_node] == stage2_send_displs[dest_node] + inter_sendcounts2[dest_node], (
                f"send buffer fill mismatch for dest_node {dest_node}"
            )

    # ==================================================================================
    # PHASE 4: Inter-Node All-to-Allv (Final Data Transfer)
    # ==================================================================================

    with torch.cuda.stream(s_inter2):
        s_inter2.wait_event(e_reorg3_done)
        if sendbuf2_size > 0 or recvbuf2_size > 0:
            req = _all_to_allv(
                recv_buffer,
                send_buffer,
                inter_sendcounts2,
                inter_recvcounts2,
                group.get_outer_group(),
                async_op=True,
                use_pccl_cpp_backend=use_pccl_cpp_backend,
                send_displs=stage2_send_displs,
                recv_displs=stage2_recv_displs
            )
            if req is not None:
                req.wait()
        if recvbuf2_size > 0:
            for src_node in range(inter_node_group_size):
                total_len = int(inter_recvcounts2[src_node])
                if total_len == 0:
                    continue
                in_base = int(stage2_recv_displs[src_node])
                pre_len = 0
                for lr in range(my_intra_rank):
                    src_rank_lr = src_node * intra_node_group_size + lr
                    pre_len += int(comm_matrix[src_rank_lr, rank])
                post_len = total_len - pre_len
                my_cnt = int(comm_matrix[src_node * intra_node_group_size + my_intra_rank, rank])
                out_base = int(recv_offsets_prefix[src_node * intra_node_group_size])
                if pre_len > 0:
                    output_flat[out_base: out_base + pre_len].copy_(
                        recv_buffer[in_base: in_base + pre_len], non_blocking=True
                    )
                if post_len > 0:
                    out_start = out_base + pre_len + my_cnt
                    in_start = in_base + pre_len
                    output_flat[out_start: out_start + post_len].copy_(
                        recv_buffer[in_start: in_start + post_len], non_blocking=True
                    )

    # Ensure Stage 1 inter-node exchange has completed before returning
    try:
        stage1_req.wait()
    except Exception:
        pass
    return None


def all_to_allv_simple_2D(output_tensor: torch.Tensor,
                   input_tensor: torch.Tensor,
                   sendcounts: List[Union[int, np.int64]],
                   recvcounts: List[Union[int, np.int64]],
                   group: Optional['ProcessGroups'] = None,
                   async_op: bool = False,
                   use_pccl_cpp_backend: bool = False,
                   algorithm: str = "spread_out",
                   overload_threshold: float = 1.0,
                   split_threshold: float = 0.0,
                   comm_matrix_rows: Optional[Union[List[List[int]], np.ndarray]] = None):

    assert not async_op, "Non blocking version not implemented"
    # assert input_tensor.dim() == 1 and output_tensor.dim() == 1, "all_to_allv_2D only admits 1D tensors"
    # Supports 1D or 2D (packed by row width) tensors. For 2D, counts represent row counts and need to be scaled by row width to element counts.
    if input_tensor.dim() == 2:
        assert output_tensor.dim() == 2, "When input is 2D, output must be 2D as well"
        assert input_tensor.size(1) == output_tensor.size(1), "2D input/output must have the same feature dimension"
        row_size = int(input_tensor.size(1))
    elif input_tensor.dim() == 1 and output_tensor.dim() == 1:
        row_size = 1
    else:
        raise AssertionError("all_to_allv_2D supports 1D tensors or 2D tensors with matching second dimension")
    input_flat = input_tensor.view(-1)
    output_flat = output_tensor.view(-1)
    sendcounts_elems = [int(c) * row_size for c in sendcounts]
    recvcounts_elems = [int(c) * row_size for c in recvcounts]

    # Get system topology information
    intra_node_group_size, inter_node_group_size = group.get_world_size()  # GPUs per node, number of nodes
    world_size = intra_node_group_size * inter_node_group_size  # Total number of GPUs

    # Validate input data consistency and completeness (using element counts for verification)
    assert len(sendcounts) == world_size, f"sendcounts length {len(sendcounts)} != world_size {world_size}"
    assert len(recvcounts) == world_size, f"recvcounts length {len(recvcounts)} != world_size {world_size}"
    assert sum(sendcounts_elems) == input_flat.numel(), f"Sum of sendcounts (elements) {sum(sendcounts_elems)} != input size {input_flat.numel()}"
    assert sum(recvcounts_elems) == output_flat.numel(), f"Sum of recvcounts (elements) {sum(recvcounts_elems)} != output size {output_flat.numel()}"

    # Determine current process position in the 2D grid
    rank = dist.get_rank()  # Get global process rank
    my_intra_rank = rank % intra_node_group_size  # Intra-node GPU ID (0-3)
    my_node_idx = rank // intra_node_group_size    # Node ID

    if comm_matrix_rows is not None:
        full_cm_rows = np.asarray(comm_matrix_rows, dtype=np.int64)
        assert full_cm_rows.shape == (world_size, world_size), (
            f"comm_matrix_rows shape {full_cm_rows.shape} != ({world_size}, {world_size})"
        )
        comm_matrix = (full_cm_rows * row_size).astype(np.int64)
    else:
        raise ValueError("comm_matrix_rows is required")

    # ==================== Step 1: Data Reorganization - Group by outer_group ====================
    # Corresponds to all_to_all's: input.view(inter_node, intra_node, -1).transpose(0,1).reshape(-1)
    #
    # Original data layout: [data_for_rank0][data_for_rank1]...[data_for_rank(world_size-1)]
    # After reorganization: [all_for_outer_group=0][all_for_outer_group=1]...[all_for_outer_group=3]
    #
    # Where "all_for_outer_group=i" contains:
    #   [data_for_rank_i][data_for_rank_(4+i)][data_for_rank_(8+i)]...[data_for_rank_(4*(nodes-1)+i)]

    input_permuted = torch.empty_like(input_flat)

    write_offset = 0
    # Iterate by outer_group
    for outer_group_id in range(intra_node_group_size):
        # Iterate over all nodes, collecting data destined for the rank with outer_group=outer_group_id on each node
        for dest_node in range(inter_node_group_size):
            dest_rank = dest_node * intra_node_group_size + outer_group_id
            count = int(comm_matrix[rank, dest_rank])
            if count > 0:
                # Compute position in the original input_flat
                read_offset = sum(int(comm_matrix[rank, r]) for r in range(dest_rank))
                # Copy data
                input_permuted[write_offset:write_offset + count].copy_(
                    input_flat[read_offset:read_offset + count]
                )
                write_offset += count

    # ==================== Step 2: Intra-node all_to_allv ====================
    # Goal: Forward data to the intra-node proxy rank responsible for the corresponding outer_group
    #
    # Logic:
    # - An outer_group consists of ranks with the same rank % intra_node_group_size
    # - Intra-node rank i is responsible for forwarding all outer_group=i communication
    # - Example: rank 0 forwards data destined for outer_group=0 (i.e., ranks 0, 4, 8, ...)

    # Compute sendcounts and recvcounts for intra-node all_to_allv
    # intra_sendcounts[i] = amount of data this rank sends to intra-node rank i (i.e., data destined for outer_group=i)
    intra_sendcounts = [0] * intra_node_group_size
    for dest_node in range(inter_node_group_size):
        for local_rank in range(intra_node_group_size):
            dest_rank = dest_node * intra_node_group_size + local_rank
            count = int(comm_matrix[rank, dest_rank])
            # local_rank corresponds to the outer_group ID
            intra_sendcounts[local_rank] += count

    # Compute recvcounts: how much data this rank (as proxy) will receive
    # Received data = all data from intra-node ranks destined for this rank's corresponding outer_group
    intra_recvcounts = [0] * intra_node_group_size
    for src_local_rank in range(intra_node_group_size):
        src_rank = my_node_idx * intra_node_group_size + src_local_rank
        # Iterate over all destination nodes, computing data from src_rank destined for outer_group=my_intra_rank
        for dest_node in range(inter_node_group_size):
            dest_rank = dest_node * intra_node_group_size + my_intra_rank
            count = int(comm_matrix[src_rank, dest_rank])
            intra_recvcounts[src_local_rank] += count

    # Execute intra-node all_to_allv
    output_intermediate = torch.empty(sum(intra_recvcounts), dtype=input_flat.dtype, device=input_flat.device)
    _all_to_allv(
        output_intermediate,
        input_permuted,
        intra_sendcounts,
        intra_recvcounts,
        group.get_inner_group(),
        async_op=False,
        use_pccl_cpp_backend=use_pccl_cpp_backend
    )

    # ==================== Step 3: Data Reorganization - Group by destination node ====================
    # Corresponds to all_to_all's: output_intermediate.view(intra_node, inter_node, -1).transpose(0,1).reshape(-1)
    #
    # Current output_intermediate layout (current rank=my_intra_rank, acting as outer_group proxy):
    #   [data_from_intra_rank0][data_from_intra_rank1]...[data_from_intra_rank3]
    # Where "data_from_intra_rank_i" contains:
    #   [destined_for_node0_outer_group=my_intra_rank][for_node1]...[for_node(N-1)]
    #
    # Conceptually: output_intermediate can be viewed as (intra_node, inter_node, variable_size_blocks)
    #
    # After reorganization: [all_data_for_node0][all_data_for_node1]...[all_data_for_node(N-1)]
    # Where "all_data_for_node_i" contains:
    #   [intra_rank0_data_for_node_i][intra_rank1_data_for_node_i]...[intra_rank3_data_for_node_i]
    #
    # Corresponds to transpose(0,1): swapping the (intra_node, inter_node) dimensions

    input_permuted2 = torch.empty_like(output_intermediate)

    write_offset = 0
    # Iterate by destination node (first dimension after transpose)
    for dest_node in range(inter_node_group_size):
        # Iterate by source intra-node rank (second dimension after transpose)
        for src_local_rank in range(intra_node_group_size):
            src_rank = my_node_idx * intra_node_group_size + src_local_rank
            dest_rank = dest_node * intra_node_group_size + my_intra_rank
            count = int(comm_matrix[src_rank, dest_rank])
            if count > 0:
                # Compute position in output_intermediate
                # output_intermediate layout: [from_rank0][from_rank1]...[from_rank3]
                # Each "from_rank_i" is internally ordered by destination node
                read_offset = 0
                # All data from the preceding src_local_rank ranks
                for prev_src_local in range(src_local_rank):
                    prev_src_rank = my_node_idx * intra_node_group_size + prev_src_local
                    for dn in range(inter_node_group_size):
                        dr = dn * intra_node_group_size + my_intra_rank
                        read_offset += int(comm_matrix[prev_src_rank, dr])
                # Add data from the current src_rank for preceding destination nodes
                for prev_node in range(dest_node):
                    prev_dr = prev_node * intra_node_group_size + my_intra_rank
                    read_offset += int(comm_matrix[src_rank, prev_dr])

                # Copy data
                input_permuted2[write_offset:write_offset + count].copy_(
                    output_intermediate[read_offset:read_offset + count]
                )
                write_offset += count

    # ==================== Step 4: Inter-node all_to_allv ====================
    # Corresponds to all_to_all's: _all_to_all(output_tensor, input_permuted, outer_group)
    #
    # At this point input_permuted2 is organized as: [for_node0][for_node1]...[for_node(N-1)]
    # After inter-node all_to_allv, each rank on each node (as proxy) receives:
    #   [from_node0][from_node1]...[from_node(N-1)]
    # Where "from_node_i" contains: data from all ranks on that node destined for
    #   outer_group=my_intra_rank on the current rank's node
    #
    # Due to Step 3's reorganization, the data is already ordered by source intra-node rank,
    # so the final output is naturally in source rank order: [from_rank0][from_rank1]...

    # Compute sendcounts for inter-node all_to_allv
    # inter_sendcounts[dest_node] = total data to send to that node
    inter_sendcounts = [0] * inter_node_group_size
    for dest_node in range(inter_node_group_size):
        # Sum data from all ranks on this node destined for outer_group=my_intra_rank on dest_node
        for src_local_rank in range(intra_node_group_size):
            src_rank = my_node_idx * intra_node_group_size + src_local_rank
            dest_rank = dest_node * intra_node_group_size + my_intra_rank
            inter_sendcounts[dest_node] += int(comm_matrix[src_rank, dest_rank])

    # Compute recvcounts for inter-node all_to_allv
    # inter_recvcounts[src_node] = amount of data received from that node
    #
    # Symmetry: inter_sendcounts computes data from this node to dest_node for outer_group=my_intra_rank
    # Therefore inter_recvcounts should compute: data from src_node to this node for outer_group=my_intra_rank
    #
    # But wait! The current rank itself is the proxy for outer_group=my_intra_rank
    # So it should receive: data from all ranks on src_node destined for the current rank!
    #
    # Actually, more broadly: data from all ranks on src_node destined for all ranks
    # with outer_group=my_intra_rank on this node
    # But the only rank with outer_group=my_intra_rank on this node is the current rank itself
    #
    # So it is simply: data from all ranks on src_node destined for the current rank
    inter_recvcounts = [0] * inter_node_group_size
    for src_node in range(inter_node_group_size):
        # Data from all ranks on src_node destined for the current rank
        for src_local_rank in range(intra_node_group_size):
            src_rank = src_node * intra_node_group_size + src_local_rank
            inter_recvcounts[src_node] += int(comm_matrix[src_rank, rank])

    torch.cuda.synchronize()
    # Execute inter-node all_to_allv, writing directly into the final output_flat
    # Due to Step 3's reorganization, output data is naturally in source rank order, no further reorganization needed
    _all_to_allv(
        output_flat,
        input_permuted2,
        inter_sendcounts,
        inter_recvcounts,
        group.get_outer_group(),
        async_op=False,
        use_pccl_cpp_backend=use_pccl_cpp_backend
    )

    return None
