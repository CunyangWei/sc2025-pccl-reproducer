import torch
import torch.distributed as dist
from mpi4py import MPI
import numpy as np
from typing import Optional, Union, List, Dict, Tuple
from .request import Request
from .process_groups import ProcessGroups
# Note: To avoid circular dependencies, sameinput/sameoutput implementations
# are not imported at the module top level; they are lazily imported inside
# conditional branches within functions.

def _all_to_allv(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    sendcounts: List[Union[int, np.int64]],
    recvcounts: List[Union[int, np.int64]],
    group: Optional[Union[dist.ProcessGroup, MPI.Comm]] = None,
    async_op: bool = False,
    use_pccl_cpp_backend: bool = False,
    algorithm: str = "pairwise_sendrecv"
) -> Optional[Request]:

    # Case 1: Use PyTorch distributed process group or default group
    if group is None or isinstance(group, dist.ProcessGroup):
        if use_pccl_cpp_backend:

            # Use PCCL C++ backend's NCCL point-to-point implementation for optimal performance
            import pccl as pccl_cpp
            from .nccl_comm import CommHandler
            nccl_comm = CommHandler.get_communicator_from_process_group(group)

            # Compute displacement arrays - MPI/NCCL needs the starting position of each data block in the buffer
            send_displs = [0]  # Send displacement array
            for i in range(len(sendcounts) - 1):
                send_displs.append(send_displs[-1] + sendcounts[i])  # Cumulative offset

            recv_displs = [0]  # Receive displacement array
            for i in range(len(recvcounts) - 1):
                recv_displs.append(recv_displs[-1] + recvcounts[i])  # Cumulative offset

            # Call the C++ implemented NCCL point-to-point communication
            request = pccl_cpp.all_to_allv_nccl_p2p(output_tensor, input_tensor, sendcounts,
                recvcounts, send_displs, recv_displs, nccl_comm.comm, nccl_comm.rank, nccl_comm.nranks
            )
            return request
        else:
            request = dist.all_to_all_single(output_tensor, input_tensor, recvcounts, sendcounts, group, async_op)
            return request


    # Case 2: Use MPI communicator - suitable for cross-node communication
    elif isinstance(group, MPI.Comm):
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

            # Compute displacement arrays needed by MPI - same logic as the NCCL implementation
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

            # Compute displacement arrays needed by MPI - same logic as the NCCL implementation
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

    # Determine the current process's position in the 2D mesh
    rank = dist.get_rank()  # Get global process rank
    my_intra_rank = rank % intra_node_group_size  # Intra-node GPU index (0-3)
    my_node_idx = rank // intra_node_group_size    # Node index

    if comm_matrix_rows is not None:
        full_cm_rows = np.asarray(comm_matrix_rows, dtype=np.int64)
        assert full_cm_rows.shape == (world_size, world_size), (
            f"comm_matrix_rows shape {full_cm_rows.shape} != ({world_size}, {world_size})"
        )
        comm_matrix = (full_cm_rows * row_size).astype(np.int64)
    else:
        raise ValueError("comm_matrix_rows is required")

    # ==================== Step 1: Data reorganization - group by outer_group ====================
    # Corresponds to all_to_all: input.view(inter_node, intra_node, -1).transpose(0,1).reshape(-1)
    #
    # Original data layout: [for rank0][for rank1]...[for rank(world_size-1)]
    # After reorganization: [all for outer_group=0][all for outer_group=1]...[all for outer_group=3]
    #
    # Where "all for outer_group=i" contains:
    #   [for rank i][for rank (4+i)][for rank (8+i)]...[for rank (4*(nodes-1)+i)]

    input_permuted = torch.empty_like(input_flat)

    write_offset = 0
    # Iterate by outer_group
    for outer_group_id in range(intra_node_group_size):
        # Iterate over all nodes, collecting data destined for the rank at outer_group=outer_group_id on each node
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
    # Goal: Forward data to the proxy rank within the node responsible for the corresponding outer_group
    #
    # Logic:
    # - An outer_group consists of ranks with the same rank % intra_node_group_size
    # - Intra-node rank i is responsible for forwarding all communication for outer_group=i
    # - For example: rank 0 forwards data destined for outer_group=0 (i.e., ranks 0, 4, 8, ...)

    # Compute sendcounts and recvcounts for intra-node all_to_allv
    # intra_sendcounts[i] = amount of data this rank sends to intra-node rank i (i.e., data for outer_group=i)
    intra_sendcounts = [0] * intra_node_group_size
    for dest_node in range(inter_node_group_size):
        for local_rank in range(intra_node_group_size):
            dest_rank = dest_node * intra_node_group_size + local_rank
            count = int(comm_matrix[rank, dest_rank])
            # local_rank corresponds to the outer_group index
            intra_sendcounts[local_rank] += count

    # Compute recvcounts: how much data this rank (as proxy) will receive
    # Received data = data from all intra-node ranks destined for this rank's outer_group
    intra_recvcounts = [0] * intra_node_group_size
    for src_local_rank in range(intra_node_group_size):
        src_rank = my_node_idx * intra_node_group_size + src_local_rank
        # Iterate over all destination nodes, computing data volume from src_rank to outer_group=my_intra_rank
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

    # ==================== Step 3: Data reorganization - group by destination node ====================
    # Corresponds to all_to_all: output_intermediate.view(intra_node, inter_node, -1).transpose(0,1).reshape(-1)
    #
    # Current output_intermediate layout (current rank=my_intra_rank, acting as outer_group proxy):
    #   [data received from intra-node rank0][from rank1]...[from rank3]
    # Where "data received from intra-node rank i" contains:
    #   [that rank's data for node0's outer_group=my_intra_rank][for node1]...[for node(N-1)]
    #
    # Think of it as: output_intermediate has shape (intra_node, inter_node, variable-length block)
    #
    # After reorganization: [all data for node0][all data for node1]...[all data for node(N-1)]
    # Where "all data for node i" contains:
    #   [intra-node rank0's data for node_i][rank1's for node_i]...[rank3's for node_i]
    #
    # Corresponds to transpose(0,1): swap (intra_node, inter_node) dimensions

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
                # output_intermediate layout: [from rank0][from rank1]...[from rank3]
                # Each "from rank i" section is ordered by destination node internally
                read_offset = 0
                # All data from the preceding src_local_rank ranks
                for prev_src_local in range(src_local_rank):
                    prev_src_rank = my_node_idx * intra_node_group_size + prev_src_local
                    for dn in range(inter_node_group_size):
                        dr = dn * intra_node_group_size + my_intra_rank
                        read_offset += int(comm_matrix[prev_src_rank, dr])
                # Add data from preceding destination nodes within the current src_rank
                for prev_node in range(dest_node):
                    prev_dr = prev_node * intra_node_group_size + my_intra_rank
                    read_offset += int(comm_matrix[src_rank, prev_dr])

                # Copy data
                input_permuted2[write_offset:write_offset + count].copy_(
                    output_intermediate[read_offset:read_offset + count]
                )
                write_offset += count

    # ==================== Step 4: Inter-node all_to_allv ====================
    # Corresponds to all_to_all: _all_to_all(output_tensor, input_permuted, outer_group)
    #
    # At this point input_permuted2 is organized as: [for node0][for node1]...[for node(N-1)]
    # After inter-node all_to_allv, each rank on each node (acting as proxy) receives:
    #   [from node0][from node1]...[from node(N-1)]
    # Where "from node i" contains: data from all ranks on that node destined for the current
    # rank's node, with outer_group=my_intra_rank
    #
    # Due to Step 3's reorganization, this data is already sorted by source intra-node rank,
    # so the final output is in source rank order: [from rank0][from rank1]...

    # Compute sendcounts for inter-node all_to_allv
    # inter_sendcounts[dest_node] = total data volume destined for that node
    inter_sendcounts = [0] * inter_node_group_size
    for dest_node in range(inter_node_group_size):
        # Accumulate data from all ranks on this node destined for dest_node's outer_group=my_intra_rank
        for src_local_rank in range(intra_node_group_size):
            src_rank = my_node_idx * intra_node_group_size + src_local_rank
            dest_rank = dest_node * intra_node_group_size + my_intra_rank
            inter_sendcounts[dest_node] += int(comm_matrix[src_rank, dest_rank])

    # Compute recvcounts for inter-node all_to_allv
    # inter_recvcounts[src_node] = data volume received from that node
    #
    # Symmetry: inter_sendcounts computes data from this node to dest_node for outer_group=my_intra_rank
    # Therefore inter_recvcounts should compute: data from src_node to this node for outer_group=my_intra_rank
    #
    # But wait! The current rank itself is the proxy for outer_group=my_intra_rank
    # So it should receive: all data from all ranks on src_node destined for the current rank!
    #
    # Actually, more broadly: all data from all ranks on src_node destined for this node's outer_group=my_intra_rank ranks
    # But the only rank in outer_group=my_intra_rank on this node is the current rank itself
    #
    # So it is: all data from all ranks on src_node destined for the current rank
    inter_recvcounts = [0] * inter_node_group_size
    for src_node in range(inter_node_group_size):
        # All data from all ranks on src_node destined for the current rank
        for src_local_rank in range(intra_node_group_size):
            src_rank = src_node * intra_node_group_size + src_local_rank
            inter_recvcounts[src_node] += int(comm_matrix[src_rank, rank])

    torch.cuda.synchronize()
    # Execute inter-node all_to_allv, writing directly to the final output_flat
    # Due to Step 3's reorganization, output data is naturally in source rank order, no further reordering needed
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

    # Determine the current process's position in the 2D mesh
    rank = dist.get_rank()  # Get global process rank
    my_intra_rank = rank % intra_node_group_size  # Intra-node GPU index (0-3)
    my_node_idx = rank // intra_node_group_size    # Node index

    if comm_matrix_rows is not None:
        full_cm_rows = np.asarray(comm_matrix_rows, dtype=np.int64)
        assert full_cm_rows.shape == (world_size, world_size), (
            f"comm_matrix_rows shape {full_cm_rows.shape} != ({world_size}, {world_size})"
        )
        comm_matrix = (full_cm_rows * row_size).astype(np.int64)
    else:
        raise ValueError("comm_matrix_rows is required")

    # ==================== Step 1: Data reorganization - group by outer_group ====================
    # Corresponds to all_to_all: input.view(inter_node, intra_node, -1).transpose(0,1).reshape(-1)
    #
    # Original data layout: [for rank0][for rank1]...[for rank(world_size-1)]
    # After reorganization: [all for outer_group=0][all for outer_group=1]...[all for outer_group=3]
    #
    # Where "all for outer_group=i" contains:
    #   [for rank i][for rank (4+i)][for rank (8+i)]...[for rank (4*(nodes-1)+i)]

    input_permuted = torch.empty_like(input_flat)

    write_offset = 0
    # Iterate by outer_group
    for outer_group_id in range(intra_node_group_size):
        # Iterate over all nodes, collecting data destined for the rank at outer_group=outer_group_id on each node
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
    # Goal: Forward data to the proxy rank within the node responsible for the corresponding outer_group
    #
    # Logic:
    # - An outer_group consists of ranks with the same rank % intra_node_group_size
    # - Intra-node rank i is responsible for forwarding all communication for outer_group=i
    # - For example: rank 0 forwards data destined for outer_group=0 (i.e., ranks 0, 4, 8, ...)

    # Compute sendcounts and recvcounts for intra-node all_to_allv
    # intra_sendcounts[i] = amount of data this rank sends to intra-node rank i (i.e., data for outer_group=i)
    intra_sendcounts = [0] * intra_node_group_size
    for dest_node in range(inter_node_group_size):
        for local_rank in range(intra_node_group_size):
            dest_rank = dest_node * intra_node_group_size + local_rank
            count = int(comm_matrix[rank, dest_rank])
            # local_rank corresponds to the outer_group index
            intra_sendcounts[local_rank] += count

    # Compute recvcounts: how much data this rank (as proxy) will receive
    # Received data = data from all intra-node ranks destined for this rank's outer_group
    intra_recvcounts = [0] * intra_node_group_size
    for src_local_rank in range(intra_node_group_size):
        src_rank = my_node_idx * intra_node_group_size + src_local_rank
        # Iterate over all destination nodes, computing data volume from src_rank to outer_group=my_intra_rank
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

    # ==================== Step 3: Data reorganization - group by destination node ====================
    # Corresponds to all_to_all: output_intermediate.view(intra_node, inter_node, -1).transpose(0,1).reshape(-1)
    #
    # Current output_intermediate layout (current rank=my_intra_rank, acting as outer_group proxy):
    #   [data received from intra-node rank0][from rank1]...[from rank3]
    # Where "data received from intra-node rank i" contains:
    #   [that rank's data for node0's outer_group=my_intra_rank][for node1]...[for node(N-1)]
    #
    # Think of it as: output_intermediate has shape (intra_node, inter_node, variable-length block)
    #
    # After reorganization: [all data for node0][all data for node1]...[all data for node(N-1)]
    # Where "all data for node i" contains:
    #   [intra-node rank0's data for node_i][rank1's for node_i]...[rank3's for node_i]
    #
    # Corresponds to transpose(0,1): swap (intra_node, inter_node) dimensions

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
                # output_intermediate layout: [from rank0][from rank1]...[from rank3]
                # Each "from rank i" section is ordered by destination node internally
                read_offset = 0
                # All data from the preceding src_local_rank ranks
                for prev_src_local in range(src_local_rank):
                    prev_src_rank = my_node_idx * intra_node_group_size + prev_src_local
                    for dn in range(inter_node_group_size):
                        dr = dn * intra_node_group_size + my_intra_rank
                        read_offset += int(comm_matrix[prev_src_rank, dr])
                # Add data from preceding destination nodes within the current src_rank
                for prev_node in range(dest_node):
                    prev_dr = prev_node * intra_node_group_size + my_intra_rank
                    read_offset += int(comm_matrix[src_rank, prev_dr])

                # Copy data
                input_permuted2[write_offset:write_offset + count].copy_(
                    output_intermediate[read_offset:read_offset + count]
                )
                write_offset += count

    # ==================== Step 4: Inter-node all_to_allv ====================
    # Corresponds to all_to_all: _all_to_all(output_tensor, input_permuted, outer_group)
    #
    # At this point input_permuted2 is organized as: [for node0][for node1]...[for node(N-1)]
    # After inter-node all_to_allv, each rank on each node (acting as proxy) receives:
    #   [from node0][from node1]...[from node(N-1)]
    # Where "from node i" contains: data from all ranks on that node destined for the current
    # rank's node, with outer_group=my_intra_rank
    #
    # Due to Step 3's reorganization, this data is already sorted by source intra-node rank,
    # so the final output is in source rank order: [from rank0][from rank1]...

    # Compute sendcounts for inter-node all_to_allv
    # inter_sendcounts[dest_node] = total data volume destined for that node
    inter_sendcounts = [0] * inter_node_group_size
    for dest_node in range(inter_node_group_size):
        # Accumulate data from all ranks on this node destined for dest_node's outer_group=my_intra_rank
        for src_local_rank in range(intra_node_group_size):
            src_rank = my_node_idx * intra_node_group_size + src_local_rank
            dest_rank = dest_node * intra_node_group_size + my_intra_rank
            inter_sendcounts[dest_node] += int(comm_matrix[src_rank, dest_rank])

    # Compute recvcounts for inter-node all_to_allv
    # inter_recvcounts[src_node] = data volume received from that node
    #
    # Symmetry: inter_sendcounts computes data from this node to dest_node for outer_group=my_intra_rank
    # Therefore inter_recvcounts should compute: data from src_node to this node for outer_group=my_intra_rank
    #
    # But wait! The current rank itself is the proxy for outer_group=my_intra_rank
    # So it should receive: all data from all ranks on src_node destined for the current rank!
    #
    # Actually, more broadly: all data from all ranks on src_node destined for this node's outer_group=my_intra_rank ranks
    # But the only rank in outer_group=my_intra_rank on this node is the current rank itself
    #
    # So it is: all data from all ranks on src_node destined for the current rank
    inter_recvcounts = [0] * inter_node_group_size
    for src_node in range(inter_node_group_size):
        # All data from all ranks on src_node destined for the current rank
        for src_local_rank in range(intra_node_group_size):
            src_rank = src_node * intra_node_group_size + src_local_rank
            inter_recvcounts[src_node] += int(comm_matrix[src_rank, rank])

    torch.cuda.synchronize()
    # Execute inter-node all_to_allv, writing directly to the final output_flat
    # Due to Step 3's reorganization, output data is naturally in source rank order, no further reordering needed
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
