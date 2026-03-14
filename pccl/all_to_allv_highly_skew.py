import torch
import torch.distributed as dist
from mpi4py import MPI
import numpy as np
from typing import Optional, Union, List, Dict, Tuple
from .request import Request
from .process_groups import ProcessGroups
# Note: To avoid circular dependencies, do not import sameinput/sameoutput implementations
# at module top level; use lazy imports inside conditional branches within functions.

# ================== Metadata Tensor Structure Definition ==================
META_SIZE = 3               # Size of the data block (bytes)
META_ORIG_TENSOR_OFFSET = 5 # Offset in the original cross_node_input_tensor
META_BUFFER_OFFSET = 6      # Offset of the data block in the current send/receive buffer
# =================================================================================

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
    
    # Case 1: Using PyTorch distributed process group or default group
    if group is None or isinstance(group, dist.ProcessGroup):
        if use_pccl_cpp_backend:
            
            # Use PCCL C++ backend NCCL point-to-point implementation for optimal performance
            import pccl as pccl_cpp
            from .nccl_comm import CommHandler
            nccl_comm = CommHandler.get_communicator_from_process_group(group)
            
            # Compute displacement arrays - MPI/NCCL needs to know the start position of each data block in the buffer
            send_displs = [0]  # Send data displacement array
            for i in range(len(sendcounts) - 1):
                send_displs.append(send_displs[-1] + sendcounts[i])  # Cumulative offset

            recv_displs = [0]  # Receive data displacement array
            for i in range(len(recvcounts) - 1):
                recv_displs.append(recv_displs[-1] + recvcounts[i])  # Cumulative offset

            # Call C++ implemented NCCL point-to-point communication
            request = pccl_cpp.all_to_allv_nccl_p2p(output_tensor, input_tensor, sendcounts,
                recvcounts, send_displs, recv_displs, nccl_comm.comm, nccl_comm.rank, nccl_comm.nranks
            )
            return request
        else:
            request = dist.all_to_all_single(output_tensor, input_tensor, recvcounts, sendcounts, group, async_op)
            return request
            
    
    # Case 2: Using MPI communicator - suitable for cross-node communication
    elif isinstance(group, MPI.Comm):
        if async_op:
            # Async operation - using Ialltoallv
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
            
            # Compute MPI displacement arrays - same logic as NCCL implementation
            send_displs = [0]
            for i in range(len(sendcounts_bytes) - 1):
                send_displs.append(send_displs[-1] + sendcounts_bytes[i])

            recv_displs = [0]
            for i in range(len(recvcounts_bytes) - 1):
                recv_displs.append(recv_displs[-1] + recvcounts_bytes[i])

            # Async GPU communication - using Ialltoallv
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

            # Compute MPI displacement arrays - same logic as NCCL implementation
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

def _precompute_inter_node_recv_counts_for_this_proxy(
    gathered_recv_rows: np.ndarray,
    intra_node_group_size: int,
    inter_node_group_size: int,
    my_node_idx: int,
    my_intra_rank: int,
    overload_threshold: float = 1.2,
    split_threshold: float = 0.2,
) -> Tuple[List[int], Dict[int, Dict[int, list]]]:
    """
    Compute the number of elements this proxy GPU (my_intra_rank) will receive from each
    source node during the outer all-to-allv phase, while caching each source node's
    single-destination assignment result for replay reuse in the third phase.
    Approach: For each src_node, replay the sender's single-destination assignment.
    - Return value 1: Accumulate only block sizes assigned to my_intra_rank (actual_recv_counts)
    - Return value 2: Per src_node per_dest_assignment cache (proxy_gpu -> blocks list)
    """
    P = intra_node_group_size
    actual_recv_counts = [0] * inter_node_group_size
    per_srcnode_assignments: Dict[int, Dict[int, list]] = {}

    for src_node in range(inter_node_group_size):
        if src_node == my_node_idx:
            actual_recv_counts[src_node] = 0
            per_srcnode_assignments[src_node] = {i: [] for i in range(P)}
            continue

        data_blocks = []
        total_to_this_dest = 0
        for src_gpu in range(P):
            global_src = src_node * P + src_gpu
            for dest_gpu in range(P):
                size = int(gathered_recv_rows[dest_gpu, global_src])
                if size > 0:
                    data_blocks.append({
                        'src_gpu': src_gpu,
                        'dest_node': my_node_idx,
                        'dest_gpu': dest_gpu,
                        'size': size,
                        'original_offset': 0,
                        'global_src': global_src,
                        'split_offset': 0,
                    })
                    total_to_this_dest += size

        if not data_blocks:
            actual_recv_counts[src_node] = 0
            per_srcnode_assignments[src_node] = {i: [] for i in range(P)}
            continue

        avg_load_per_proxy = total_to_this_dest / float(P) if P > 0 else 0.0
        per_dest_assignment = _smart_allocation_with_splitting_single_dest(
            data_blocks, avg_load_per_proxy, P, overload_threshold, split_threshold
        )
        my_list = per_dest_assignment.get(my_intra_rank, [])
        actual_recv_counts[src_node] = int(sum(int(b['size']) for b in my_list))
        per_srcnode_assignments[src_node] = per_dest_assignment

    return actual_recv_counts, per_srcnode_assignments

def _extract_and_execute_intra_node_communication(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    sendcounts: List[Union[int, np.int64]],
    recvcounts: List[Union[int, np.int64]],
    intra_node_group_size: int,
    inter_node_group_size: int,
    my_node_idx: int,
    group: 'ProcessGroups',
    use_pccl_cpp_backend: bool = False,
    algorithm: str = "spread_out"):
    
    # 1. Extract intra-node communication send and receive counts (using slices to avoid data copies)
    node_start = my_node_idx * intra_node_group_size
    node_end_exclusive = node_start + intra_node_group_size
    
    # Directly use slices to reference the intra-node portion of the original lists, avoiding data copies
    intra_node_sendcounts = sendcounts[node_start:node_end_exclusive]
    intra_node_recvcounts = recvcounts[node_start:node_end_exclusive]
    
    # 2. Select contiguous intra-node slices directly on original tensors and execute alltoallv (zero-copy)
    send_start_offset = 0
    recv_start_offset = 0
    for i in range(node_start):
        send_start_offset += sendcounts[i]
        recv_start_offset += recvcounts[i]
    total_intra_send = 0
    total_intra_recv = 0
    for i in range(node_start, node_end_exclusive):
        total_intra_send += sendcounts[i]
        total_intra_recv += recvcounts[i]
    input_intra_view = input_tensor[send_start_offset: send_start_offset + total_intra_send]
    output_intra_view = output_tensor[recv_start_offset: recv_start_offset + total_intra_recv]
    
    _all_to_allv(output_intra_view, input_intra_view,
                 intra_node_sendcounts, intra_node_recvcounts,
                 group.get_inner_group(), async_op=True,  # Keep existing async call
                 use_pccl_cpp_backend=use_pccl_cpp_backend,
                 algorithm=algorithm)

def _collect_node_data_blocks(cross_node_comm_matrix: np.ndarray,
                             intra_node_group_size: int,
                             inter_node_group_size: int,
                             my_node_idx: int,
                             target_dest_node: Optional[int] = None) -> tuple:
    """
    Phase 1: Data collection and statistics - preparing data for load balancing.

    Design rationale:
    This function is the first step of the load balancing algorithm. It collects information
    about all outbound data from every GPU on the current node destined for other nodes.
    The core idea is to gather the originally scattered send tasks from individual GPUs
    into a unified view, preparing for subsequent intelligent assignment.

    Key design decisions:
    1. Group by destination node - the foundation of load balancing; data for the same
       destination node can be handled by the same proxy GPU.
    2. Record detailed data block information - including source GPU, destination GPU,
       size, original offset, etc., ensuring complete data tracking.
    3. Compute ideal load - provides a target value for proxy GPU assignment to achieve
       load balancing.
    4. Sort by size - smaller blocks first, beneficial for the subsequent greedy
       allocation algorithm.
    """
    world_size = intra_node_group_size * inter_node_group_size
    
    # Collect all outbound data blocks grouped by destination node (optionally: only for a specific destination node)
    data_blocks_by_dest_node = {}  # dest_node -> list of data blocks
    total_outbound_data = 0        # Total outbound data volume

    # Iterate over all GPUs on the current node
    for src_gpu in range(intra_node_group_size):
        global_src = my_node_idx * intra_node_group_size + src_gpu  # Compute global GPU index

        # Compute offset of the current source GPU in the original input tensor
        src_offset = 0
        for dest_rank in range(world_size):
            # Read using local matrix row index (src_gpu)
            chunk_size = cross_node_comm_matrix[src_gpu, dest_rank]
            if chunk_size > 0:
                dest_node = dest_rank // intra_node_group_size  # Compute destination node
                dest_gpu = dest_rank % intra_node_group_size    # Compute destination GPU
                
                # Only consider cross-node communication; intra-node communication is handled by other mechanisms
                # When target_dest_node is specified, only collect data destined for that node
                if dest_node != my_node_idx and (target_dest_node is None or dest_node == target_dest_node):
                    if dest_node not in data_blocks_by_dest_node:
                        data_blocks_by_dest_node[dest_node] = []
                    
                    # Create data block descriptor - contains complete routing and positioning information
                    data_block = {
                        'src_gpu': src_gpu,              # Source GPU index (intra-node)
                        'dest_node': dest_node,          # Destination node index
                        'dest_gpu': dest_gpu,            # Destination GPU index (intra-node)
                        'size': chunk_size,              # Data block size
                        'original_offset': src_offset,   # Offset in the original tensor
                        'global_src': global_src,        # Global source GPU index
                        'global_dest': dest_rank,        # Global destination GPU index
                        'split_offset': 0                # Original data block starts from the beginning
                    }
                    
                    data_blocks_by_dest_node[dest_node].append(data_block)
                    total_outbound_data += chunk_size
                
                # Accumulate offset in the original input tensor (including both intra-node and cross-node data)
                src_offset += chunk_size
    
    # Sort data blocks for each destination node by size in ascending order
    for dest_node in data_blocks_by_dest_node:
        data_blocks_by_dest_node[dest_node].sort(key=lambda x: x['size'])
    
    # Compute ideal average load per proxy GPU - target value for load balancing (based on filtered data volume)
    avg_load_per_proxy = total_outbound_data / intra_node_group_size if intra_node_group_size > 0 else 0
    
    return data_blocks_by_dest_node, avg_load_per_proxy

def _smart_allocation_with_splitting_single_dest(data_blocks: list,
                                                avg_load_per_proxy: float,
                                                intra_node_group_size: int,
                                                overload_threshold: float = 1.2,
                                                split_threshold: float = 0.2):
    """
    Single-destination-node version of the smart allocation algorithm: performs proxy GPU
    assignment and optional splitting for a data block list targeting a single destination node.
    Input data blocks must contain keys: src_gpu, dest_node, dest_gpu, size, original_offset,
    global_src, global_dest, split_offset.
    Returns: assignment_tracking: dict[proxy_gpu] -> list[data_block]
    """
    # Copy list to avoid in-place modification of the upstream list
    pending_blocks = list(data_blocks)
    # Maintain the existing ascending-size processing order (already sorted in collection phase).
    pending_blocks.sort(key=lambda x: x['size'])

    proxy_loads = [0.0] * intra_node_group_size
    proxy_assignments = {}  # (dest_node, dest_gpu) -> proxy_gpu (for single destination node, dest_node is constant)
    assignment_tracking = {i: [] for i in range(intra_node_group_size)}

    max_allowed_load_per_proxy = avg_load_per_proxy * overload_threshold if intra_node_group_size > 0 else 0.0

    while pending_blocks:
        data_block = pending_blocks.pop(0)
        dest_key = (data_block['dest_node'], data_block['dest_gpu'])

        assigned_proxy = None
        need_splitting = False

        # Prefer the proxy GPU with the corresponding destination GPU index as the candidate
        if dest_key not in proxy_assignments:
            corresponding_proxy = data_block['dest_gpu']
            if proxy_loads[corresponding_proxy] + data_block['size'] <= max_allowed_load_per_proxy:
                assigned_proxy = corresponding_proxy
                proxy_assignments[dest_key] = assigned_proxy
            else:
                available_capacity = max_allowed_load_per_proxy - proxy_loads[corresponding_proxy]
                if available_capacity > avg_load_per_proxy * split_threshold:
                    need_splitting = True
                    assigned_proxy = corresponding_proxy
                    proxy_assignments[dest_key] = assigned_proxy
        else:
            candidate_proxy = proxy_assignments[dest_key]
            if proxy_loads[candidate_proxy] + data_block['size'] <= max_allowed_load_per_proxy:
                assigned_proxy = candidate_proxy
            else:
                available_capacity = max_allowed_load_per_proxy - proxy_loads[candidate_proxy]
                if available_capacity > avg_load_per_proxy * split_threshold:
                    need_splitting = True
                    assigned_proxy = candidate_proxy

        if assigned_proxy is None:
            least_loaded_proxy = min(range(intra_node_group_size), key=lambda i: proxy_loads[i])
            if proxy_loads[least_loaded_proxy] + data_block['size'] <= max_allowed_load_per_proxy:
                assigned_proxy = least_loaded_proxy
                proxy_assignments[dest_key] = assigned_proxy
            else:
                available_capacity = max_allowed_load_per_proxy - proxy_loads[least_loaded_proxy]
                if available_capacity > avg_load_per_proxy * split_threshold:
                    need_splitting = True
                    assigned_proxy = least_loaded_proxy
                    proxy_assignments[dest_key] = assigned_proxy

        if need_splitting and assigned_proxy is not None:
            available_capacity = max_allowed_load_per_proxy - proxy_loads[assigned_proxy]
            split_size = min(int(available_capacity), data_block['size'])
            remaining_size = data_block['size'] - split_size

            if split_size > 0:
                current_split_offset = data_block.get('split_offset', 0)

                split_block_1 = data_block.copy()
                split_block_1.update({'size': split_size, 'split_offset': current_split_offset})

                proxy_loads[assigned_proxy] += split_size
                assignment_tracking[assigned_proxy].append(split_block_1)
                split_block_1['assigned_proxy'] = assigned_proxy

                if remaining_size > 0:
                    split_block_2 = data_block.copy()
                    split_block_2.update({'size': remaining_size, 'split_offset': current_split_offset + split_size})
                    # Put remaining part back at the front of the queue for priority processing
                    pending_blocks.insert(0, split_block_2)
            else:
                least_loaded = min(range(intra_node_group_size), key=lambda i: proxy_loads[i])
                proxy_loads[least_loaded] += data_block['size']
                if 'split_offset' not in data_block:
                    data_block['split_offset'] = 0
                assignment_tracking[least_loaded].append(data_block)
                data_block['assigned_proxy'] = least_loaded
                if dest_key not in proxy_assignments:
                    proxy_assignments[dest_key] = least_loaded
        elif assigned_proxy is not None:
            proxy_loads[assigned_proxy] += data_block['size']
            if 'split_offset' not in data_block:
                data_block['split_offset'] = 0
            assignment_tracking[assigned_proxy].append(data_block)
            data_block['assigned_proxy'] = assigned_proxy
        else:
            least_loaded = min(range(intra_node_group_size), key=lambda i: proxy_loads[i])
            proxy_loads[least_loaded] += data_block['size']
            if 'split_offset' not in data_block:
                data_block['split_offset'] = 0
            assignment_tracking[least_loaded].append(data_block)
            data_block['assigned_proxy'] = least_loaded
            if dest_key not in proxy_assignments:
                proxy_assignments[dest_key] = least_loaded

    return assignment_tracking

def _prepare_intra_node_data_transfer(input_tensor: torch.Tensor,
                                     assignment_tracking: dict,
                                     intra_node_group_size: int,
                                     my_intra_rank: int) -> tuple:
    """
    Phase 4A: Intra-node data transfer preparation - reorganize data based on assignment tracking.

    Design rationale:
    This is the execution phase of the load balancing algorithm, responsible for converting
    the intelligent assignment results into actual data transfer operations. The core task is
    to reorganize data in the input tensor according to proxy GPU assignments, preparing for
    intra-node data shuffling. The key challenge is handling the complexity introduced by
    data block splitting, ensuring each data fragment is sent to its designated location accurately.

    Key design decisions:
    1. Bidirectional mapping computation - compute both send mapping (what to send) and
       receive mapping (what to receive) simultaneously.
    2. Precise position tracking - maintain complete position and routing information for
       each data block (including split blocks).
    3. Split block support - special handling of split block metadata to ensure accuracy
       of subsequent reassembly.
    4. Buffer reorganization - rearrange data according to alltoallv protocol requirements
       to improve transfer efficiency.

    Data reorganization strategy:
    - Group by proxy GPU: store all data destined for the same proxy GPU contiguously.
    - Maintain complete metadata: each data block carries full routing and split information.
    - Optimize memory access: reduce memory jumps during data copies to improve cache efficiency.

    Algorithm advantages:
    - Precision: ensures every data byte reaches its target location accurately.
    - Flexibility: supports unified handling of regular and split blocks.
    - Efficiency: optimizes data layout to reduce transfer and processing overhead.
    - Traceability: complete metadata supports problem diagnosis and performance analysis.
    """
    # Initialize send and receive count arrays and mapping tables
    send_counts = [0] * intra_node_group_size  # Amount of data to send to each proxy GPU
    recv_counts = [0] * intra_node_group_size  # Amount of data to receive from each source GPU
    send_mapping = []  # Send mapping: (dest proxy GPU, send buffer offset, original offset, size, metadata)
    recv_mapping = []  # Receive mapping: (source GPU, receive buffer offset, size, metadata)
    
    # Compute send counts and create send mapping - process data originating from the current GPU
    total_send_size = 0
    for proxy_gpu, data_blocks in assignment_tracking.items():
        for block in data_blocks:
            if block['src_gpu'] == my_intra_rank:  # Only process data blocks originating from the current GPU
                send_counts[proxy_gpu] += block['size']  # Accumulate data volume to send to this proxy GPU
                # Create detailed send mapping entry with routing and split information
                send_mapping.append({
                    'dest_proxy': proxy_gpu,                    # Destination proxy GPU
                    'original_offset': block['original_offset'], # Position in the original tensor
                    'size': block['size'],                      # Data block size
                    'dest_node': block['dest_node'],            # Final destination node
                    'dest_gpu': block['dest_gpu'],              # Final destination GPU
                    'split_offset': block.get('split_offset', 0)  # Split offset
                })
                total_send_size += block['size']
    
    # Compute receive counts - data the current GPU will receive when acting as a proxy GPU
    total_recv_size = 0
    if my_intra_rank in assignment_tracking:  # If the current GPU is assigned as a proxy GPU
        for block in assignment_tracking[my_intra_rank]:
            recv_counts[block['src_gpu']] += block['size']  # Accumulate data volume received from this source GPU
            # Create receive mapping entry, preserving complete metadata information
            recv_mapping.append({
                'src_gpu': block['src_gpu'],                # Source GPU of data
                'size': block['size'],                      # Data block size
                'dest_node': block['dest_node'],            # Final destination node
                'dest_gpu': block['dest_gpu'],              # Final destination GPU
                'original_offset': block['original_offset'], # Original offset position
                'split_offset': block.get('split_offset', 0)  # Split offset
            })
            total_recv_size += block['size']
    
    # Sort mapping tables to ensure consistent processing order - improves cache efficiency and predictability
    send_mapping.sort(key=lambda x: (x['dest_proxy'], x['original_offset'], x['split_offset']))
    recv_mapping.sort(key=lambda x: (x['src_gpu'], x['original_offset']))
    
    # Create send buffer and reorganize input data
    send_buffer = torch.empty(total_send_size, dtype=input_tensor.dtype, device=input_tensor.device)
    
    if total_send_size > 0:
        send_offset = 0
        for proxy_gpu in range(intra_node_group_size):
            if send_counts[proxy_gpu] > 0:
                for mapping in send_mapping:
                    if mapping['dest_proxy'] == proxy_gpu:
                        size = mapping['size']
                        orig_offset = mapping['original_offset']
                        split_offset = mapping['split_offset']
                        
                        actual_read_offset = orig_offset + split_offset
                        
                        # Validate boundaries
                        if actual_read_offset + size > input_tensor.numel():
                            raise ValueError(f"Split data read out of bounds: "
                                           f"actual_read_offset={actual_read_offset}, "
                                           f"size={size}, input_size={input_tensor.numel()}")
                        
                        # Read split data from the correct position
                        send_buffer[send_offset:send_offset + size] = \
                            input_tensor[actual_read_offset:actual_read_offset + size]
                        
                        # Update send buffer offset in the mapping table
                        mapping['offset_in_send_buffer'] = send_offset
                        send_offset += size
    
    return send_buffer, send_counts, recv_counts, recv_mapping

def _execute_intra_node_data_shuffle(send_buffer: torch.Tensor,
                                    send_counts: List[int],
                                    recv_counts: List[int],
                                    group: Union[dist.ProcessGroup, MPI.Comm],
                                    use_pccl_cpp_backend: bool = False,
                                    algorithm: str = "spread_out") -> tuple:
    """
    Phase 4B: Execute the actual intra-node data transfer.
    
    Perform customized alltoallv operation for intra-node data shuffling,
    handling the reorganized data from proxy assignment phase.
    
    Args:
        send_buffer: reorganized data ready for transfer
        send_counts: amount of data to send to each GPU
        recv_counts: amount of data to receive from each GPU
        group: process group for intra-node communication
        use_pccl_cpp_backend: whether to use PCCL C++ backend
        algorithm: algorithm to use for the transfer
    """
    total_recv_size = sum(recv_counts)
    recv_buffer = torch.empty(total_recv_size, dtype=send_buffer.dtype, device=send_buffer.device)
    
    current_stream = torch.cuda.current_stream(send_buffer.device)

    request = _all_to_allv(recv_buffer, send_buffer, send_counts, recv_counts, 
                    group, async_op=False, use_pccl_cpp_backend=use_pccl_cpp_backend,
                    algorithm=algorithm)
    
    if use_pccl_cpp_backend and request is not None:
        if hasattr(request, 'wait'):
            request.wait()  # Wait for NCCL operation to complete
    
    return recv_buffer, current_stream


def _organize_received_data_for_inter_node(recv_buffer: torch.Tensor,
                                         recv_mapping: list) -> dict:
    """
    Phase 4C: Organize received data for inter-node communication phase.
    
    Parse the received buffer according to recv_mapping and organize data by
    destination node for the subsequent inter-node communication phase.
    
    Args:
        recv_buffer: data received from intra-node shuffle
        recv_mapping: detailed mapping of received data layout (pre-sorted!)
        recv_counts: amount of data received from each source GPU
        intra_node_group_size: number of GPUs in the node
    """
    if recv_buffer.numel() == 0:
        return {}
    
    # Group received data by destination node
    data_by_dest_node = {}
    
    # Parse received buffer according to recv_mapping - DO NOT RE-SORT!
    # recv_mapping is already correctly ordered to match recv_buffer layout
    buffer_offset = 0
    for mapping in recv_mapping:
        dest_node = mapping['dest_node']
        dest_gpu = mapping['dest_gpu']
        size = mapping['size']
        
        if dest_node not in data_by_dest_node:
            data_by_dest_node[dest_node] = []
        
        # Record the data block location and destination
        data_by_dest_node[dest_node].append({
            'offset_in_buffer': buffer_offset,
            'size': size,
            'dest_gpu': dest_gpu,
            'src_gpu': mapping['src_gpu'],
            'split_offset': mapping.get('split_offset', 0),
            'original_offset': mapping['original_offset']
        })
        
        buffer_offset += size
    
    # Prepare inter-node send information
    inter_node_send_info = {}
    for dest_node, blocks in data_by_dest_node.items():
        # Use deterministic order: sorted by (dest_gpu ascending, src_gpu ascending, split_offset ascending, original_offset ascending)
        blocks.sort(key=lambda b: (b['dest_gpu'], b['src_gpu'], b.get('split_offset', 0), b.get('original_offset', 0)))
        inter_node_send_info[dest_node] = {
            'blocks': blocks,
            'total_size': sum(block['size'] for block in blocks)
        }
    
    # return organized_buffer, inter_node_send_info
    return inter_node_send_info

def _pack_data_for_inter_node_transfer(organized_buffer: torch.Tensor,
                                      inter_node_send_info: dict,
                                      inter_node_group_size: int) -> tuple:
    """
    Optimized version: minimize data copies, reuse existing buffers as much as possible.
    """
    inter_send_counts = [0] * inter_node_group_size
    
    # Check if data is already arranged in node order
    sorted_dest_nodes = sorted(inter_node_send_info.keys())

    # Need to repack, but use optimized batch copy
    total_packed_size = organized_buffer.numel()
    packed_send_buffer = torch.empty(total_packed_size, dtype=organized_buffer.dtype, 
                                    device=organized_buffer.device)
    
    # Build batch copy plan
    copy_batches = []
    pack_offset = 0
    
    for dest_node in sorted_dest_nodes:
        if dest_node in inter_node_send_info:
            node_info = inter_node_send_info[dest_node]
            # Use the already-sorted block order from the previous step (deterministic, replayable)
            blocks = node_info['blocks']
            
            inter_send_counts[dest_node] = node_info['total_size']
            
            # Look for contiguous blocks for batch copy
            batch_start = blocks[0]['offset_in_buffer']
            batch_size = blocks[0]['size']
            
            for i in range(1, len(blocks)):
                if blocks[i]['offset_in_buffer'] == batch_start + batch_size:
                    # Extend batch
                    batch_size += blocks[i]['size']
                else:
                    # Execute current batch copy
                    copy_batches.append((batch_start, batch_size, pack_offset))
                    pack_offset += batch_size
                    
                    # Start new batch
                    batch_start = blocks[i]['offset_in_buffer']
                    batch_size = blocks[i]['size']
            
            # Last batch
            copy_batches.append((batch_start, batch_size, pack_offset))
            pack_offset += batch_size
            
            # This version no longer builds or returns cross-node metadata tensors
        
    # Execute batch copies
    for src_offset, size, dst_offset in copy_batches:
        packed_send_buffer[dst_offset:dst_offset + size] = \
            organized_buffer[src_offset:src_offset + size]

    return packed_send_buffer, inter_send_counts

def _execute_inter_node_balanced_transfer(
    packed_send_buffer: torch.Tensor,
    inter_send_counts: List[int],
    group: Union[dist.ProcessGroup, MPI.Comm],
    inter_node_group_size: int,
    use_pccl_cpp_backend: bool = False,
    algorithm: str = "spread_out",
    precomputed_actual_recv_counts: Optional[List[int]] = None) -> tuple:
    
    # ========= Phase 1: Compute receive sizes (prefer precomputed values, no communication needed) =========
    assert precomputed_actual_recv_counts is not None, "precomputed_actual_recv_counts must be provided"
    actual_recv_counts = list(map(int, precomputed_actual_recv_counts))
            
    # ========= Phase 2: Start async data transfer =========
    # Start data transfer immediately based on receive sizes, without waiting for metadata
    total_recv_size = sum(actual_recv_counts)
    inter_recv_buffer = torch.empty(total_recv_size, dtype=packed_send_buffer.dtype,
                                   device=packed_send_buffer.device)
    
    # Start async data transfer
    data_request = None
    data_request = _all_to_allv(
            inter_recv_buffer, packed_send_buffer, 
            inter_send_counts, actual_recv_counts,
            group, async_op=True,
            use_pccl_cpp_backend=True,
            algorithm=algorithm
        )
    
    # Wait for data transfer to complete
    if data_request is not None:
        if hasattr(data_request, 'wait'):
            data_request.wait()  # PyTorch
        elif hasattr(data_request, 'Wait'):
            data_request.Wait()  # MPI
    
    # Return data buffer and receive counts (no metadata)
    return inter_recv_buffer, actual_recv_counts

def _replay_sender_policy_for_all_sources(
    inter_recv_counts: List[int],
    gathered_recv_rows: np.ndarray,
    intra_node_group_size: int,
    inter_node_group_size: int,
    my_node_idx: int,
    my_intra_rank: int,
    overload_threshold: float = 1.2,
    split_threshold: float = 0.2,
    cached_per_srcnode_assignment: Optional[Dict[int, Dict[int, list]]] = None,
) -> tuple:
    """
    Local replay based on recvcounts: for each source node, reconstruct the sender's
    assignment/chunking order to generate routing information.
    Returns (parsed_data_info, distribution_plan).
    """
    parsed_data_info = {}
    distribution_plan = {gpu: [] for gpu in range(intra_node_group_size)}
    # Plan and counts for forwarded data to be received from each local GPU
    forward_receive_plan = {gpu: [] for gpu in range(intra_node_group_size)}
    forward_recv_counts = [0] * intra_node_group_size

    # Compute the starting offset of each source node in inter_recv_buffer (concatenated in node index order)
    node_prefix_offsets = [0]
    for i in range(inter_node_group_size - 1):
        node_prefix_offsets.append(node_prefix_offsets[-1] + int(inter_recv_counts[i]))

    for src_node in range(inter_node_group_size):
        if src_node == my_node_idx or inter_recv_counts[src_node] == 0:
            continue

        # Decide whether to use cached assignment results or reconstruct on the fly
        per_dest_assignment: Dict[int, list]
        if cached_per_srcnode_assignment is not None and src_node in cached_per_srcnode_assignment:
            per_dest_assignment = cached_per_srcnode_assignment[src_node]
        else:
            # Reconstruct on the fly (keeping logic exactly consistent with the sender side)
            data_blocks = []
            total_to_this_dest = 0
            for src_gpu in range(intra_node_group_size):
                global_src = src_node * intra_node_group_size + src_gpu
                for dest_gpu in range(intra_node_group_size):
                    size = int(gathered_recv_rows[dest_gpu, global_src])
                    if size > 0:
                        data_blocks.append({
                            'src_gpu': src_gpu,
                            'dest_node': my_node_idx,
                            'dest_gpu': dest_gpu,
                            'size': size,
                            'original_offset': 0,
                            'global_src': global_src,
                            'split_offset': 0,
                        })
                        total_to_this_dest += size

            if not data_blocks:
                continue

            avg_load_per_proxy = total_to_this_dest / float(intra_node_group_size) if intra_node_group_size > 0 else 0.0
            per_dest_assignment = _smart_allocation_with_splitting_single_dest(
                data_blocks, avg_load_per_proxy, intra_node_group_size,
                overload_threshold, split_threshold
            )

        # 3) Extract segments received by the current receiving GPU (my_intra_rank), assign offsets in deterministic order
        my_segments = list(per_dest_assignment.get(my_intra_rank, []))
        my_segments.sort(key=lambda s: (s['dest_gpu'], s['src_gpu'], s.get('split_offset', 0), s.get('original_offset', 0)))
        base_offset = node_prefix_offsets[src_node]
        running = 0
        for seg in my_segments:
            routing_info = {
                'src_node': src_node,
                'src_gpu': seg['src_gpu'],
                'dest_gpu': seg['dest_gpu'],
                'size': int(seg['size']),
                'offset_in_recv_buffer': base_offset + running,
                'split_offset': int(seg.get('split_offset', 0)),
            }
            running += int(seg['size'])

            routing_info['global_src'] = src_node * intra_node_group_size + routing_info['src_gpu']
            if routing_info['dest_gpu'] == my_intra_rank:
                routing_info['needs_transfer'] = False
            else:
                routing_info['needs_transfer'] = True
                distribution_plan[routing_info['dest_gpu']].append(routing_info)

            if src_node not in parsed_data_info:
                parsed_data_info[src_node] = []
            parsed_data_info[src_node].append(routing_info)

        # 4) Count segments “I need to receive from other local GPUs” (these segments are currently stored on other local GPUs)
        # These correspond to segments in per_dest_assignment[proxy!=my_intra_rank] where dest_gpu == my_intra_rank
        for proxy_gpu in range(intra_node_group_size):
            if proxy_gpu == my_intra_rank:
                continue
            segs = per_dest_assignment.get(proxy_gpu, [])
            if not segs:
                continue
            # Keep only segments destined for this GPU
            filtered = [seg for seg in segs if seg['dest_gpu'] == my_intra_rank]
            if not filtered:
                continue
            # Sender's intra-node packing order: (dest_gpu asc) -> (src_node asc) -> (src_gpu asc) -> (split_offset asc)
            # Here dest_gpu is always my_intra_rank, so sort by (src_node, src_gpu, split_offset, original_offset)
            filtered.sort(key=lambda s: (src_node, s['src_gpu'], s.get('split_offset', 0), s.get('original_offset', 0)))
            for seg in filtered:
                forward_receive_plan[proxy_gpu].append({
                    'src_node': src_node,
                    'src_gpu': seg['src_gpu'],
                    'size': int(seg['size']),
                    'split_offset': int(seg.get('split_offset', 0)),
                })
                forward_recv_counts[proxy_gpu] += int(seg['size'])

    return parsed_data_info, distribution_plan, forward_recv_counts, forward_receive_plan


def _prepare_intra_node_distribution(inter_recv_buffer: torch.Tensor,
                                    distribution_plan: dict,
                                    parsed_data_info: dict,
                                    intra_node_group_size: int,  
                                    my_intra_rank: int) -> tuple:
    """
    Phase 3B: Prepare data for intra-node distribution.
    
    Organize data that needs to be forwarded to other GPUs within the node.
    Create detailed forward mapping for metadata exchange.
    """
    forward_send_counts = [0] * intra_node_group_size
    forward_send_mapping_tensors = {gpu: [] for gpu in range(intra_node_group_size)}  # Optional: retained but not exchanged
    local_data_info = []  # Data that's already on the correct GPU
    
    # Calculate send counts and mapping - data this GPU needs to forward to others
    total_forward_send_size = 0
    for dest_gpu, routing_list in distribution_plan.items():
        if dest_gpu != my_intra_rank:  # Don't send to self
            # Deterministic order: (src_node asc) -> (src_gpu asc) -> (split_offset asc) -> (offset_in_recv_buffer asc)
            sorted_list = sorted(routing_list, key=lambda r: (r['src_node'], r['src_gpu'], r.get('split_offset', 0), r['offset_in_recv_buffer']))
            for routing_info in sorted_list:
                forward_send_counts[dest_gpu] += routing_info['size']
                total_forward_send_size += routing_info['size']
                # Optional: build local mapping (not exchanged)
                forward_mapping_row = torch.tensor([
                    dest_gpu,
                    routing_info['src_gpu'],
                    routing_info['src_node'],
                    routing_info['size'],
                    routing_info.get('split_offset', 0),
                    routing_info['offset_in_recv_buffer'],
                    0
                ], dtype=torch.int64, device='cpu')
                forward_send_mapping_tensors[dest_gpu].append(forward_mapping_row)
    
    # Collect local data that doesn't need forwarding
    for src_node, routing_list in parsed_data_info.items():
        for routing_info in routing_list:
            if routing_info['dest_gpu'] == my_intra_rank and not routing_info['needs_transfer']:
                # This data is already on the correct GPU (this GPU)
                local_data_info.append(routing_info)
    
    # Create forward send buffer
    forward_send_buffer = torch.empty(total_forward_send_size, dtype=inter_recv_buffer.dtype, 
                                     device=inter_recv_buffer.device)
    
    if total_forward_send_size > 0:
        send_offset = 0
        # Pack in deterministic order: dest_gpu ascending
        for dest_gpu in range(intra_node_group_size):
            count = forward_send_counts[dest_gpu]
            if count > 0:
                tensor_idx = 0
                # Use the mapping order saved above for each dest_gpu
                for mapping_row in forward_send_mapping_tensors[dest_gpu]:
                    size = int(mapping_row[META_SIZE].item())
                    src_offset = int(mapping_row[META_ORIG_TENSOR_OFFSET].item())
                    forward_send_buffer[send_offset:send_offset + size] = \
                        inter_recv_buffer[src_offset:src_offset + size]
                    # Record offset in forward buffer
                    forward_send_mapping_tensors[dest_gpu][tensor_idx][META_BUFFER_OFFSET] = send_offset
                    tensor_idx += 1
                    send_offset += size
    
    # Convert lists to Tensors
    final_forward_send_mapping_tensors = {}
    for dest_gpu, tensor_list in forward_send_mapping_tensors.items():
        if tensor_list:
            final_forward_send_mapping_tensors[dest_gpu] = torch.stack(tensor_list)  # Keep on CPU
        else:
            final_forward_send_mapping_tensors[dest_gpu] = torch.empty((0, 7), dtype=torch.int64, device='cpu')
    
    return forward_send_buffer, forward_send_counts, final_forward_send_mapping_tensors, local_data_info

def _execute_intra_node_distribution(forward_send_buffer: torch.Tensor,
                                    forward_send_counts: List[int],
                                    forward_recv_counts: List[int],
                                    group: dist.ProcessGroup) -> torch.Tensor:
    """
    Phase 3C: Execute intra-node data distribution.
    """
    # Ensure counts are Python int type
    forward_send_counts = [int(count) for count in forward_send_counts]
    forward_recv_counts = [int(count) for count in forward_recv_counts]
    
    total_forward_recv_size = sum(forward_recv_counts)
    forward_recv_buffer = torch.empty(total_forward_recv_size, dtype=forward_send_buffer.dtype, 
                                     device=forward_send_buffer.device)
    
    # 1. Split send buffer into tensor list according to send_counts
    input_list = []
    send_offset = 0
    for count in forward_send_counts:
        if count > 0:
            input_list.append(forward_send_buffer[send_offset:send_offset + count])
        else:
            input_list.append(torch.empty(0, dtype=forward_send_buffer.dtype, 
                                        device=forward_send_buffer.device))
        send_offset += count
    
    # 2. Prepare separate tensor list for receiving
    output_list = []
    for count in forward_recv_counts:
        if count > 0:
            output_list.append(torch.empty(count, dtype=forward_send_buffer.dtype, 
                                         device=forward_send_buffer.device))
        else:
            output_list.append(torch.empty(0, dtype=forward_send_buffer.dtype, 
                                         device=forward_send_buffer.device))
    
    # 3. All processes must call all_to_all
    dist.all_to_all(output_list, input_list, group=group)
    
    # 4. Concatenate received data into forward_recv_buffer
    recv_offset = 0
    for i, tensor in enumerate(output_list):
        if forward_recv_counts[i] > 0:
            forward_recv_buffer[recv_offset:recv_offset + forward_recv_counts[i]] = tensor
            recv_offset += forward_recv_counts[i]
    
    return forward_recv_buffer

def _assemble_final_output_cross_node(output_tensor: torch.Tensor,
                                     forward_recv_buffer: torch.Tensor,
                                     local_data_info: list,
                                     inter_recv_buffer: torch.Tensor,
                                     original_recvcounts: List[int],
                                     intra_node_group_size: int,
                                     inter_node_group_size: int
                                     ) -> None:

    world_size = intra_node_group_size * inter_node_group_size
    
    # Compute output offsets using original recvcounts
    output_offsets = [0]
    for i in range(world_size - 1):
        output_offsets.append(output_offsets[-1] + original_recvcounts[i])
    
    # 1. Process local data (data already on the correct GPU after cross-node transfer)
    for routing_info in local_data_info:
        # Compute global source rank
        global_src = routing_info['src_node'] * intra_node_group_size + routing_info['src_gpu']
        size = routing_info['size']
        src_offset = routing_info['offset_in_recv_buffer']
        split_offset = routing_info.get('split_offset', 0)  # Get split offset
        output_start = output_offsets[global_src] + split_offset  # Use split_offset to compute correct position
        
        # Copy from cross-node receive buffer to output tensor
        output_tensor[output_start:output_start + size] = \
            inter_recv_buffer[src_offset:src_offset + size]
    
    # 2. Process forwarded data (data received via intra-node forwarding)
    # Removed the path based on explicit mapping tensors; now uses replay-order write-back (see subsequent steps)


def _assemble_forwarded_segments_without_metadata(
    output_tensor: torch.Tensor,
    forward_recv_buffer: torch.Tensor,
    forward_receive_plan: dict,
    recvcounts_elems: List[int],
    intra_node_group_size: int,
    inter_node_group_size: int
) -> None:
    """
    Write forward_recv_buffer back to output in deterministic order without exchanging
    any mapping metadata. The ordering rules are consistent with the sender's intra-node
    packing order, ensuring both sides can replay consistently.
    """
    world_size = intra_node_group_size * inter_node_group_size
    output_offsets = [0]
    for i in range(world_size - 1):
        output_offsets.append(output_offsets[-1] + recvcounts_elems[i])

    # Construct receive order: sort segments for each source local GPU by (src_node, src_gpu, split_offset)
    ordered_segments = []
    for src_local_gpu in range(intra_node_group_size):
        segs = list(forward_receive_plan.get(src_local_gpu, []))
        if segs:
            segs.sort(key=lambda s: (s['src_node'], s['src_gpu'], s.get('split_offset', 0)))
            for s in segs:
                ordered_segments.append((src_local_gpu, s))

    read_ptr = 0
    for _, seg in ordered_segments:
        size = int(seg['size'])
        if size == 0:
            continue
        src_node = int(seg['src_node'])
        src_gpu = int(seg['src_gpu'])
        split_offset = int(seg.get('split_offset', 0))
        global_src = src_node * intra_node_group_size + src_gpu
        output_start = output_offsets[global_src] + split_offset
        output_tensor[output_start:output_start + size] = \
            forward_recv_buffer[read_ptr:read_ptr + size]
        read_ptr += size

    assert read_ptr == forward_recv_buffer.numel(), \
        f"Forward buffer not fully consumed: read={read_ptr}, total={forward_recv_buffer.numel()}"


def all_to_allv_withcomm(output_tensor: torch.Tensor,
                           input_tensor: torch.Tensor,
                           sendcounts: List[Union[int, np.int64]],
                           recvcounts: List[Union[int, np.int64]],
                           group: Optional[ProcessGroups] = None,
                           async_op: bool = False,
                           use_pccl_cpp_backend: bool = False,
                           algorithm: str = "spread_out",
                           overload_threshold: float = 1.0,
                           split_threshold: float = 0.0,
                           comm_matrix_rows: Optional[Union[List[List[int]], np.ndarray]] = None) -> torch.Tensor:
    """
    Complete three-phase load-balanced all-to-allv implementation - main function.

    Overall design rationale:
    This is the master control function of the entire load balancing algorithm, implementing
    an innovative three-phase communication pattern specifically designed for imbalanced data
    distribution in high-performance computing environments. Traditional all-to-allv is prone
    to network hotspots and GPU load imbalance when facing uneven data distributions. This
    algorithm introduces the concept of proxy GPUs, decomposing communication into three
    carefully designed phases to achieve spatiotemporal load balancing.

    Three-phase architecture design:

    Phase 1 - Intra-node aggregation and load balancing:
    - Data collection: collect outbound data information from all GPUs within the node.
    - Smart assignment: use destination-node cycling and greedy algorithm to assign data
      to proxy GPUs.
    - Data splitting: intelligently split oversized data blocks to avoid load concentration.
    - Intra-node shuffle: efficiently redistribute data within the node via NCCL.

    Phase 2 - Cross-node balanced transfer:
    - Data packing: each proxy GPU packs data destined for the same target node.
    - Load balancing: ensure each NIC's outbound traffic is precomputed and evenly distributed.
    - Communication optimization: reduce N*N communications to N*4, significantly lowering
      network pressure.

    Phase 3 - Intra-node distribution and final assembly:
    - Route parsing: parse received data packets to determine the final destination GPU.
    - Intra-node forwarding: use NCCL to forward data to the final destination GPU.
    - Data reassembly: correctly assemble data from different proxy GPUs into the output tensor.

    Core innovations:
    1. Proxy GPU mechanism - introduces an intermediate layer for load dispersion.
    2. Spatiotemporal decoupling - decomposes load balancing into time and space dimensions.
    3. Adaptive splitting - dynamically handles uneven data distributions.
    4. Metadata integrity - ensures accurate tracking of complex data flows.
    """
    
    # Support 1D or 2D (packed by row width) tensors. For 2D, counts represent row counts and need to be scaled to element counts by row width.
    if input_tensor.dim() == 2:
        assert output_tensor.dim() == 2, "When input is 2D, output must be 2D as well"
        assert input_tensor.size(1) == output_tensor.size(1), "2D input/output must have the same feature dimension"
        row_size = int(input_tensor.size(1))
    elif input_tensor.dim() == 1 and output_tensor.dim() == 1:
        row_size = 1
    else:
        raise AssertionError("all_to_allv_2D supports 1D tensors or 2D tensors with matching second dimension")

    # Create flat views for internal implementation and convert counts uniformly to element counts
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
    
    # Determine position of the current process in the 2D grid
    rank = dist.get_rank()  # Get global process rank
    my_intra_rank = rank % intra_node_group_size  # Intra-node GPU index (0-3)
    my_node_idx = rank // intra_node_group_size    # Node index
    
    # Before the three-phase pipeline, handle direct intra-node communication first
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _extract_and_execute_intra_node_communication(
                input_flat, output_flat, sendcounts_elems, recvcounts_elems,
                intra_node_group_size, inter_node_group_size, my_node_idx,
                group, use_pccl_cpp_backend, algorithm
            )
    
    # Ensure the inner MPI group exists for CPU metadata transfer
    if group.get_inner_mpi_group() is None:
        group._create_inner_mpi_group(intra_node_group_size)
    
    # ==================== Metadata Collection Phase (Intra-node Allgather or use provided matrix) ====================
    # Construct this node's communication matrix, shape: (intra_node_group_size, world_size)
    inner_mpi_group = group.get_inner_mpi_group()
    gathered_rows = np.zeros((intra_node_group_size, world_size), dtype=np.int64)
    gathered_recv_rows = np.zeros((intra_node_group_size, world_size), dtype=np.int64)
    req_send_gather = None
    req_recv_gather = None
    node_start = my_node_idx * intra_node_group_size
    node_end_exclusive = node_start + intra_node_group_size
    if comm_matrix_rows is not None:
        # Use the global communication matrix provided by the caller (in row counts). Scale by row width to element counts and slice to this node's rows
        full_cm_rows = np.asarray(comm_matrix_rows, dtype=np.int64)
        assert full_cm_rows.shape == (world_size, world_size), (
            f"comm_matrix_rows shape {full_cm_rows.shape} != ({world_size}, {world_size})"
        )
        full_cm_elems = (full_cm_rows * row_size).astype(np.int64)
        # Rows for this node's source GPUs (local view of the send matrix)
        comm_matrix = full_cm_elems[node_start:node_end_exclusive, :]
        # Columns destined for this node's GPUs (local view of the receive matrix), stored in gathered_recv_rows rows
        for dest_gpu in range(intra_node_group_size):
            dest_global = my_node_idx * intra_node_group_size + dest_gpu
            gathered_recv_rows[dest_gpu, :] = full_cm_elems[:, dest_global]
    else:
        # Collect all this node's GPU sendcounts/recvcounts rows within the node only, avoiding global Allgather overhead
        sendcounts_array = np.array(sendcounts_elems, dtype=np.int64)
        recvcounts_array = np.array(recvcounts_elems, dtype=np.int64)
        # Non-blocking gather, initiate as early as possible
        req_send_gather = inner_mpi_group.Iallgather(sendcounts_array, gathered_rows)
        req_recv_gather = inner_mpi_group.Iallgather(recvcounts_array, gathered_recv_rows)
    
    
    # Ensure completion before using gathered_rows/gathered_recv_rows (if using Allgather path)
    if req_send_gather is not None:
        req_send_gather.Wait()
        # Use local matrix (only this node's rows) as the communication matrix
        comm_matrix = gathered_rows  # shape: (intra_node_group_size, world_size)

    # ==================== Phase 1: Intra-node Aggregation and Load Balancing ====================
    # Goal: redistribute outbound data from all GPUs in the node to proxy GPUs for load balancing

    # 1.1/1.2 Independent per-destination-node allocation: loop over dest nodes, collect and allocate separately, then merge results
    assignment_tracking = {i: [] for i in range(intra_node_group_size)}
    for dest_node in range(inter_node_group_size):
        if dest_node == my_node_idx:
            continue  # Skip intra-node destinations (already handled by intra-node direct communication)
        # Only collect data blocks destined for this target node
        per_dest_blocks_dict, per_dest_avg_load = _collect_node_data_blocks(
            comm_matrix, intra_node_group_size, inter_node_group_size, my_node_idx, target_dest_node=dest_node
        )
        # Flatten the data block list for this destination node
        per_dest_blocks = per_dest_blocks_dict.get(dest_node, [])
        if not per_dest_blocks:
            continue
        # Use single-destination-node allocation algorithm for proxy assignment with optional splitting
        per_dest_assignment = _smart_allocation_with_splitting_single_dest(
            per_dest_blocks, per_dest_avg_load, intra_node_group_size,
            overload_threshold, split_threshold
        )
        # Merge into the overall assignment_tracking
        for proxy_gpu in range(intra_node_group_size):
            if proxy_gpu in per_dest_assignment:
                assignment_tracking[proxy_gpu].extend(per_dest_assignment[proxy_gpu])
    
    # 1.3 Data reorganization: rearrange cross-node input data based on proxy GPU assignment results
    # Converts smart allocation results into actual data transfer operations, reorganizing the cross-node input tensor
    # Args: cross-node input tensor, proxy GPU assignment tracking, intra-node GPU count, current GPU index
    # Returns:
    #   - send_buffer: torch.Tensor, reorganized send buffer with data arranged by proxy GPU order
    #   - send_counts: list[int], amount of data to send to each proxy GPU
    #   - send_mapping: list[dict], send mapping table with detailed send info (offset, size, target, etc.)
    #   - recv_counts: list[int], amount of data to receive from each source GPU (when current GPU acts as proxy)
    #   - recv_mapping: list[dict], receive mapping table with detailed info and final routing
    send_buffer, send_counts, recv_counts, recv_mapping = _prepare_intra_node_data_transfer(
        input_flat, assignment_tracking, intra_node_group_size, my_intra_rank
    )
    
    # 1.4 Intra-node data shuffle: efficiently redistribute data within the node via NCCL
    # Executes actual intra-node data transfer using alltoallv to move data from source GPUs to proxy GPUs
    
    intra_recv_buffer, intra_stream = _execute_intra_node_data_shuffle(
        send_buffer, send_counts, recv_counts, group.get_inner_group(),
        use_pccl_cpp_backend, algorithm
    )

    # 1.5 Data organization: prepare data and metadata for inter-node transfer phase
    # Parses intra-node received data, reorganizes by target node for inter-node transfer
    # Args: intra-node receive buffer, receive mapping, receive counts, intra-node GPU count
    # Returns:
    #   - organized_buffer: torch.Tensor, data buffer reorganized for inter-node transfer
    #   - inter_node_send_info: dict[dest_node] -> send info with data block list and metadata per dest node
    #     Format: {dest_node: {'blocks': [data block info list], 'total_size': total size}}
    # organized_buffer, inter_node_send_info = _organize_received_data_for_inter_node(
    #     intra_recv_buffer, recv_mapping, recv_counts, intra_node_group_size
    # )
    
    inter_node_send_info = _organize_received_data_for_inter_node(
        intra_recv_buffer, recv_mapping
    )
    
    # ==================== Phase 2: Inter-node Balanced Transfer ====================
    # Implements cross-node communication between proxy GPUs, significantly reducing communication count and balancing network load

    # 2.1 Data packing: each proxy GPU packs data destined for the same target node
    # Packs all data for the same destination node together for batch transfer optimization
    # Args: reorganized data buffer, inter-node send info, number of nodes
    # Returns:
    #   - packed_send_buffer: torch.Tensor, send buffer packed by destination node
    #   - inter_send_counts: list[int], amount of data to send to each destination node
    #   - send_metadata_tensors: dict[dest_node] -> torch.Tensor, detailed routing info tensor per data packet
    #     Contains offset in packet, size, final destination GPU, etc.
    packed_send_buffer, inter_send_counts = _pack_data_for_inter_node_transfer(
        intra_recv_buffer, inter_node_send_info, inter_node_group_size
    )
    intra_stream.synchronize() 
    # torch.cuda.synchronize()
    
    # 2.1.5 Before starting phase 2, ensure recv Iallgather for computing receive sizes has completed (if using Allgather path)
    if req_recv_gather is not None:
        req_recv_gather.Wait()
    precomputed_actual_recv_counts, cached_per_srcnode_assignment = _precompute_inter_node_recv_counts_for_this_proxy(
        gathered_recv_rows, intra_node_group_size, inter_node_group_size,
        my_node_idx, my_intra_rank, overload_threshold, split_threshold
    )

    # 2.2 Inter-node balanced transfer (no sizes alltoall, using precomputed sizes)
    inter_recv_buffer, inter_recv_counts = _execute_inter_node_balanced_transfer(
        packed_send_buffer, inter_send_counts, group.get_outer_group(),
        inter_node_group_size,
        use_pccl_cpp_backend, algorithm,
        precomputed_actual_recv_counts=precomputed_actual_recv_counts
    )
    
    # ==================== Phase 3: Intra-node Distribution and Final Assembly ====================
    # Correctly distributes received data to final destination GPUs and assembles the output tensor

    # 3.1 Routing info parsing: parse received data packets to determine each block's final destination
    # Parses routing info based on actual metadata received during inter-node transfer, not the original comm matrix
    # Args: inter-node recv buffer, recv counts, actual received routing metadata tensors, node topology info, current position info
    # Returns:
    #   - parsed_data_info: dict[src_node] -> routing info list, reconstructed from actual metadata
    #   - distribution_plan: dict[dest_gpu] -> routing info list, data blocks to forward to each destination GPU
    #     Uses actual data distribution rather than original communication pattern
    parsed_data_info, distribution_plan, forward_recv_counts, forward_receive_plan = _replay_sender_policy_for_all_sources(
        inter_recv_counts, gathered_recv_rows, intra_node_group_size,
        inter_node_group_size, my_node_idx, my_intra_rank,
        overload_threshold, split_threshold, cached_per_srcnode_assignment
    )
    
    # 3.2 Intra-node distribution preparation: organize data that needs to be forwarded to other GPUs
    # Prepares final intra-node data distribution, separating data to forward from data already in the correct position
    # Args: inter-node recv buffer, distribution plan, parsed data info, intra-node GPU count, current GPU index
    # Returns:
    #   - forward_send_buffer: torch.Tensor, buffer of data to forward to other GPUs
    #   - forward_send_counts: list[int], amount of data to forward to each intra-node GPU
    #   - forward_send_mapping_tensors: dict[GPU] -> torch.Tensor, forwarding metadata tensors (for communication)
    #   - local_data_info: list[dict], data info already on the correct GPU (no forwarding needed)
    forward_send_buffer, forward_send_counts, forward_send_mapping_tensors, local_data_info = _prepare_intra_node_distribution(
        inter_recv_buffer, distribution_plan, parsed_data_info, intra_node_group_size, my_intra_rank
    )
    
    # Intra-node forwarding phase no longer exchanges mapping metadata; uses replayed forward_recv_counts as receive counts
    
    forward_recv_buffer = _execute_intra_node_distribution(
        forward_send_buffer, forward_send_counts, forward_recv_counts, group.get_inner_group()
    )
    
    _assemble_final_output_cross_node(
        output_flat, forward_recv_buffer, local_data_info, inter_recv_buffer,
        recvcounts_elems, intra_node_group_size,
        inter_node_group_size
    )

    # Write intra-node forwarded received data (forward_recv_buffer) back to final output in deterministic order
    _assemble_forwarded_segments_without_metadata(
        output_flat, forward_recv_buffer, forward_receive_plan,
        recvcounts_elems, intra_node_group_size, inter_node_group_size
    )

    return output_tensor



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
    2D alltoallv:
      Step 1  (local pack) : Based on global comm matrix, bucket input into P local “aggregation targets” (by dest rank's local index)
      Step 2  (intra a2av) : alltoallv on inner_group, aggregating data for each outer group to this node's aggregation rank
      Step 3  (repack)     : On aggregation rank, re-bucket by destination node to form contiguous segments for inter-node transfer
      Step 4  (inter a2av) : alltoallv on outer_group, sending data cross-node directly to final dest rank's aggregation endpoint;
                              after receiving, perform local restore writing into output_tensor in “global source rank order”

    Notes:
      - P = intra_node_group_size (GPUs per node, e.g. P=4 on Perlmutter)
      - Q = inter_node_group_size (number of nodes)
      - outer_group members are all ranks with the same local index (i.e. rank % P is the same)
      - This implementation performs only two communications; the rest are device-local copies
      - Assumes _all_to_allv(sendcounts, recvcounts) concatenates in ascending peer rank order
    """
    
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

    # ---- Basic topology parameters ----
    P, Q = group.get_world_size()                  # P=intra, Q=inter
    world_size = P * Q

    # Get global rank (prefer from group, fall back to dist.get_rank())
    my_rank = group.get_global_rank() if hasattr(group, "get_global_rank") else dist.get_rank()

    assert len(sendcounts) == world_size and len(recvcounts) == world_size, \
        "sendcounts/recvcounts must be world_size long"

    # ---- Communication matrix (np.ndarray) & convenience indexing ----
    if comm_matrix_rows is None:
        # If not provided, construct own row from local sendcounts, other rows set to 0 (works but Step2/Step4 peer counts may be imprecise)
        cm = np.zeros((world_size, world_size), dtype=np.int64)
        cm[my_rank, :] = np.asarray(sendcounts, dtype=np.int64)
    else:
        cm = np.asarray(comm_matrix_rows, dtype=np.int64)
        assert cm.shape == (world_size, world_size)

    def node_of(r: int) -> int:
        return r // P
    def lidx_of(r: int) -> int:
        return r % P

    my_node = node_of(my_rank)
    my_lidx = lidx_of(my_rank)

    # ---- Validate input/output lengths ----
    total_send = int(np.sum(sendcounts))
    total_recv = int(np.sum(recvcounts))
    assert input_flat.numel() == total_send, "input_flat size must equal sum(sendcounts)"
    assert output_flat.numel() == total_recv, "output_flat size must equal sum(recvcounts)"

    # ---- Precompute: starting offset of each destination block in input_flat for my row ----
    dest_prefix = np.zeros(world_size + 1, dtype=np.int64)
    for d in range(world_size):
        dest_prefix[d + 1] = dest_prefix[d] + int(sendcounts[d])

    # ================================================================
    # Step 1: Array reorganization (bucket by destination's local index, preparing for intra-node a2av)
    # bucket k (0..P-1) collects blocks destined for all nodes with local index == k
    # ================================================================
    # Count total bytes per bucket
    intra_sendcounts = np.zeros(P, dtype=np.int64)
    # Also record per-bucket sizes subdivided by destination node, providing granularity for Step 3 repacking
    # intra_bucket_sizes[k][n] = total amount I send to “dest node n with local index == k”
    intra_bucket_sizes = [[0 for _ in range(Q)] for _ in range(P)]

    for d in range(world_size):
        cnt = int(cm[my_rank, d])  # = sendcounts[d]
        if cnt == 0:
            continue
        k = lidx_of(d)          # This block's corresponding outer group aggregation local index
        n = node_of(d)          # Destination node number
        intra_sendcounts[k] += cnt
        intra_bucket_sizes[k][n] += cnt

    # Construct Step1 packed buffer (concatenated by k=0..P-1; within each k, sorted by n=0..Q-1)
    step1_buf = torch.empty_like(input_flat)
    write_ptr_per_bucket = np.zeros(P, dtype=np.int64)  # Current write pointer within each bucket (relative to bucket start)
    bucket_base = np.zeros(P, dtype=np.int64)
    # Starting offset of each bucket in step1_buf
    acc = 0
    for k in range(P):
        bucket_base[k] = acc
        acc += int(intra_sendcounts[k])

    # Prepare read pointer for each destination block before copying from input_flat
    read_ptr_per_dest = dest_prefix.copy()

    # Actual copy: iterate over destinations d in ascending order, placing each block into its bucket (k)
    # within the destination node n's interval. For simplicity, layout is (k,n) order: k0:n0..nQ-1, k1:..., kP-1:...
    # Since each d belongs to a unique (k,n), we can simply advance write pointers sequentially
    for d in range(world_size):
        cnt = int(cm[my_rank, d])
        if cnt == 0:
            continue
        k = lidx_of(d)
        # Write position = bucket start + offset within bucket
        dst_off = int(bucket_base[k] + write_ptr_per_bucket[k])
        src_off = int(read_ptr_per_dest[d])
        step1_buf[dst_off: dst_off + cnt].copy_(input_flat[src_off: src_off + cnt])
        write_ptr_per_bucket[k] += cnt
        read_ptr_per_dest[d] += cnt

    # ---- Compute intra-node send/recv counts ----
    # Amount I send to intra-node peer with local index == k:
    #   intra_sendcounts[k] already computed
    # Amount I will receive from each intra-node source s (0..P-1) (as aggregation rank with local index my_lidx):
    #   intra_recvcounts[s] = sum_n cm[src=my_node*P + s, dest=n*P + my_lidx] over n
    intra_recvcounts = np.zeros(P, dtype=np.int64)
    for s in range(P):
        src_rank = my_node * P + s
        total = 0
        for n in range(Q):
            total += int(cm[src_rank, n * P + my_lidx])
        intra_recvcounts[s] = total

    # ================================================================
    # Step 2: Intra-node alltoallv
    #   Send counts: intra_sendcounts[k]
    #   Recv counts: intra_recvcounts[s]
    #   Send buffer: step1_buf; Recv buffer: step2_buf
    #   Received data concatenation order: s=0..P-1
    # ================================================================
    step2_total_recv = int(np.sum(intra_recvcounts))
    step2_buf = torch.empty(step2_total_recv, dtype=output_flat.dtype, device=output_flat.device)

    _all_to_allv(
        output_flat=step2_buf,
        input_tensor=step1_buf,
        sendcounts=intra_sendcounts.tolist(),
        recvcounts=intra_recvcounts.tolist(),
        group=group.get_inner_group(),
        async_op=False,
        use_pccl_cpp_backend=use_pccl_cpp_backend,
        algorithm=algorithm
    )

    # ================================================================
    # Step 3: Data reorganization (on aggregation rank, re-bucket by dest node n for inter-node a2av)
    #   step2_buf layout is concatenated in ascending order s=0..P-1;
    #   Since in Step1 each s was packed in n=0..Q-1 order, each s's subsegment in step2_buf can be subdivided into Q segments
    #   Here we merge these [s,n] subsegments into Q contiguous segments ordered by n
    # ================================================================
    # First recover the starting point of each s segment
    s_base = np.zeros(P + 1, dtype=np.int64)  # Cumulative s segments
    for s in range(P):
        s_base[s + 1] = s_base[s] + int(intra_recvcounts[s])

    # For current aggregation lidx=my_lidx, build inter_sendcounts[n] = sum_{s} cm[my_node*P + s, n*P + my_lidx]
    inter_sendcounts = np.zeros(Q, dtype=np.int64)
    # Also record [s,n] sizes for data movement
    sn_sizes = [[0 for _ in range(Q)] for _ in range(P)]
    for s in range(P):
        src_rank = my_node * P + s
        for n in range(Q):
            cnt = int(cm[src_rank, n * P + my_lidx])
            sn_sizes[s][n] = cnt
            inter_sendcounts[n] += cnt

    # Target buffer step3_buf: concatenated in order n=0..Q-1
    step3_total_send = int(np.sum(inter_sendcounts))
    step3_buf = torch.empty(step3_total_send, dtype=output_flat.dtype, device=output_flat.device)

    n_base = np.zeros(Q, dtype=np.int64)  # Starting point of each n
    acc = 0
    for n in range(Q):
        n_base[n] = acc
        acc += int(inter_sendcounts[n])

    # Copy [s,n] subsegments from step2_buf to corresponding n segments in step3_buf (for each n, append in order s=0..P-1)
    n_write_ptr = np.zeros(Q, dtype=np.int64)
    for s in range(P):
        # Within s segment, ordered by n=0..Q-1
        s_ptr = int(s_base[s])
        for n in range(Q):
            cnt = int(sn_sizes[s][n])
            if cnt == 0:
                continue
            dst_off = int(n_base[n] + n_write_ptr[n])
            step3_buf[dst_off: dst_off + cnt].copy_(step2_buf[s_ptr: s_ptr + cnt])
            s_ptr += cnt
            n_write_ptr[n] += cnt
        # Verify s_ptr == s_base[s+1]
        assert s_ptr == int(s_base[s + 1]), "Internal packing invariant broken in Step 3"

    # ================================================================
    # Step 4: Inter-node (outer group) alltoallv
    #   Send counts: inter_sendcounts[n]
    #   Recv counts: inter_recvcounts[m] = sum_{t=0..P-1} cm[m*P + t, my_rank]
    #   Send buffer: step3_buf; Recv buffer: step4_buf (bucketed by source node m)
    #   Finally restore locally to output_tensor in ascending global source rank order (m*P + t)
    # ================================================================
    inter_recvcounts = np.zeros(Q, dtype=np.int64)
    for m in range(Q):
        total = 0
        for t in range(P):
            total += int(cm[m * P + t, my_rank])
        inter_recvcounts[m] = total

    step4_total_recv = int(np.sum(inter_recvcounts))
    step4_buf = torch.empty(step4_total_recv, dtype=output_flat.dtype, device=output_flat.device)

    _all_to_allv(
        output_tensor=step4_buf,
        input_tensor=step3_buf,
        sendcounts=inter_sendcounts.tolist(),
        recvcounts=inter_recvcounts.tolist(),
        group=group.get_outer_group(),
        async_op=False,
        use_pccl_cpp_backend=use_pccl_cpp_backend,
        algorithm=algorithm
    )

    # ---- Final local restore: expand step4_buf (bucketed by source node m) into output_tensor ordered by global source rank 0..W-1 ----
    # Compute per-source prefix for output
    src_prefix = np.zeros(world_size + 1, dtype=np.int64)
    for s in range(world_size):
        src_prefix[s + 1] = src_prefix[s] + int(cm[s, my_rank])   # = recvcounts[s]

    # Starting point of each m segment in step4_buf
    m_base = np.zeros(Q + 1, dtype=np.int64)
    for m in range(Q):
        m_base[m + 1] = m_base[m] + int(inter_recvcounts[m])

    # From each m segment, extract blocks of size cm[m*P + t, my_rank] in order t=0..P-1, placing them at the global source order position in output
    out_write_ptr = src_prefix.copy()  # Write pointer for each source rank (effectively fixed)
    for m in range(Q):
        read_ptr = int(m_base[m])
        for t in range(P):
            s_rank = m * P + t
            cnt = int(cm[s_rank, my_rank])
            if cnt == 0:
                continue
            dst_off = int(src_prefix[s_rank])
            output_flat[dst_off: dst_off + cnt].copy_(step4_buf[read_ptr: read_ptr + cnt])
            read_ptr += cnt
        assert read_ptr == int(m_base[m + 1]), "Internal unpacking invariant broken in final restore"
