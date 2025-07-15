import torch
import torch.distributed as dist
from mpi4py import MPI
from typing import Optional, Union
from .request import Request
from .process_groups import ProcessGroups
from .nccl_comm import CommHandler, NCCLCommunicator

def spread_out_all_to_all_mpi(output_tensor: torch.Tensor,
                              input_tensor: torch.Tensor,
                              group: Optional[MPI.Comm] = None,
                              async_op: bool = False,
                              algorithm: str = "spread_out"):
    """
    Performs a spread-out all-to-all on CUDA tensors using MPI point-to-point operations.
    
    Each process starts with a 1D input_tensor of size (P * block_size) and sends
    block i to process i. The output_tensor will contain the blocks received from
    all processes.
    
    Parameters:
      output_tensor : torch.Tensor
          Pre-allocated tensor of same size as input_tensor on a CUDA device.
      input_tensor : torch.Tensor
          1D tensor representing local data to be distributed.
      group : Optional[MPI.Comm]
          MPI communicator; defaults to MPI.COMM_WORLD.
      async_op : bool
          Non-blocking operations are not supported in this implementation.
    """
    assert not async_op, "non-blocking primitives not supported"
    comm = MPI.COMM_WORLD if group is None else group
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Ensure input and output tensors have the same size
    assert input_tensor.numel() == output_tensor.numel(), "Input and output tensors must have same size"
    assert input_tensor.numel() % size == 0, "Input tensor size must be divisible by number of processes"
    
    total_elems = input_tensor.numel()
    block_size = total_elems // size
    
    # For now, only implement spread_out in Python (others use C++ backend)
    if algorithm != "spread_out":
        raise NotImplementedError(f"Algorithm {algorithm} not implemented in Python backend. Use C++ backend instead.")

    # Create temporary buffers for communication
    send_buf = torch.empty_like(input_tensor)
    recv_buf = torch.empty_like(input_tensor)
    
    # Copy input to send buffer
    send_buf.copy_(input_tensor)
    
    # Initialize output with local data
    output_tensor[rank * block_size:(rank + 1) * block_size].copy_(
        input_tensor[rank * block_size:(rank + 1) * block_size]
    )
    
    # Perform all-to-all exchanges
    for step in range(1, size):
        partner = (rank + step) % size
        
        # Calculate send and receive offsets
        send_offset = partner * block_size
        recv_offset = partner * block_size
        
        # Synchronize CUDA stream before MPI communication
        torch.cuda.current_stream().synchronize()
        
        # Exchange data with partner
        comm.Sendrecv(
            sendbuf=send_buf[send_offset:send_offset + block_size],
            dest=partner, sendtag=0,
            recvbuf=recv_buf[recv_offset:recv_offset + block_size],
            source=partner, recvtag=0
        )
        
        # Copy received data to output
        output_tensor[recv_offset:recv_offset + block_size].copy_(
            recv_buf[recv_offset:recv_offset + block_size]
        )

def _all_to_all(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Optional[Union[dist.ProcessGroup, MPI.Comm]] = None,
    async_op: bool = False,
    use_pccl_cpp_backend: bool = False,
    algorithm: str = "spread_out",
    is_intra_node: bool = False
) -> Optional[Request]:

    # Case 1: torch.distributed.ProcessGroup
    if group is None or isinstance(group, dist.ProcessGroup):

        if algorithm == "hypre" and is_intra_node and use_pccl_cpp_backend:
            
            from .nccl_comm import CommHandler
            comm_idx = CommHandler.create_communicator_from_process_group(group)
            nccl_comm = CommHandler.get_communicator_from_idx(comm_idx)
            
            nccl_comm_ptr = nccl_comm.get_comm_handle()
            if nccl_comm_ptr is None or nccl_comm_ptr == 0:
                raise RuntimeError("Failed to get valid NCCL communicator handle")
            
            # import pccl as pccl_cpp
            # pccl_cpp.all_to_all_nccl(output_tensor, input_tensor, nccl_comm_ptr)
            # return None
            nccl_comm.my_all_to_all(output_tensor, input_tensor)
            return None
        else:
            input_list = list(torch.chunk(input_tensor, dist.get_world_size(group)))
            output_list = list(torch.chunk(output_tensor, dist.get_world_size(group)))
            request = dist.all_to_all(output_list, input_list, group, async_op)
    
    # Case 2: mpi4py.MPI.Comm
    elif isinstance(group, MPI.Comm):
        
        # For hypre algorithm with MPI groups, use bruck
        if algorithm == "hypre":
            algorithm = "bruck"
        
        if use_pccl_cpp_backend:
            import pccl as pccl_cpp
            request = pccl_cpp.all_to_all_mpi(output_tensor, 
                                              input_tensor, 
                                              group,
                                              algorithm)
        else:
            request = spread_out_all_to_all_mpi(output_tensor, input_tensor, group, async_op, algorithm)
    else:
        raise TypeError(
            f"Unsupported group type: {type(group)}. "
            "Expected torch.distributed.ProcessGroup or mpi4py.MPI.Comm."
        )
    return request 

def all_to_all_2D(output_tensor: torch.Tensor,
                  input_tensor: torch.Tensor,
                  group: Optional[ProcessGroups] = None,
                  async_op: bool = False,
                  use_pccl_cpp_backend: bool = False,
                  algorithm: str = "spread_out"):

    assert not async_op, "Non blocking version not implemented"
    assert input_tensor.dim() == 1 and output_tensor.dim() == 1, "all_to_all_2D only admits 1D tensors"
    
    intra_node_group_size, inter_node_group_size = group.get_world_size()
    world_size = intra_node_group_size * inter_node_group_size
    
    # Ensure input and output tensors have the same size
    assert input_tensor.numel() == output_tensor.numel(), "Input and output tensors must have same size"
    assert input_tensor.numel() % world_size == 0, "Input tensor size must be divisible by world size"

    # Step 1: Permute input data for hierarchical communication
    # Reshape and transpose to group data by destination
    input_permuted = input_tensor.view(inter_node_group_size, intra_node_group_size, -1).transpose(0, 1).reshape(-1)
    
    # Step 2: Intra-node all-to-all
    output_intermediate = torch.empty_like(input_permuted)
    _all_to_all(output_intermediate, input_permuted, group.get_inner_group(), 
                async_op=False, use_pccl_cpp_backend=use_pccl_cpp_backend, algorithm=algorithm, is_intra_node=True)
    
    input_permuted = output_intermediate.view(intra_node_group_size, inter_node_group_size, -1).transpose(0, 1).reshape(-1)
    
    # Step 3: Inter-node all-to-all
    _all_to_all(output_tensor, input_permuted, group.get_outer_group(), 
                async_op=False, use_pccl_cpp_backend=use_pccl_cpp_backend, algorithm=algorithm, is_intra_node=False)
    
    # Step 4: Unpermute output data
    # output_unpermuted = output_tensor.view(intra_node_group_size, inter_node_group_size, -1).transpose(0, 1).reshape(-1)
    # output_tensor.copy_(output_unpermuted)