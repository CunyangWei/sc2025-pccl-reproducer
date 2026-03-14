#include <torch/extension.h>
#include <torch/torch.h>
#include <pybind11/pybind11.h> // PyBind11
#include <mpi4py/mpi4py.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include "reduce_scatter.h"
#include "all_gather.h"
#include "all_to_all.h"
#include "all_to_allv.h"
#include "permute.h"
#include "common.h"
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp> // PyTorch C++ header
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp> // NCCL backend header

namespace py = pybind11;

struct NcclComm {
    ncclComm_t comm;
    int        world_size;
    int        rank;
  
    NcclComm(const torch::Tensor& unique_id_tensor,
             int world_size_,
             int rank_): world_size(world_size_), rank(rank_) {
          // 1) Must be a CPU ByteTensor of length 128
          TORCH_CHECK(unique_id_tensor.device().is_cpu(), "unique_id must be on CPU");
          TORCH_CHECK(unique_id_tensor.scalar_type() == torch::kUInt8, "unique_id must be a torch.uint8 Tensor");
          TORCH_CHECK(unique_id_tensor.numel() == NCCL_UNIQUE_ID_BYTES, "unique_id must have length 128");
  
          // 2) Copy out the bytes
          ncclUniqueId id;
          auto* src = unique_id_tensor.data_ptr<uint8_t>();
          std::memcpy(id.internal, src, NCCL_UNIQUE_ID_BYTES);
  
          // 3) Init NCCL
          NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, rank));
    }
  
    ~NcclComm() {
      if (comm) ncclCommDestroy(comm);
    }
};
  
torch::Tensor get_nccl_unique_id() {
    // 1) ask NCCL for a new ID
    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));

    // 2) wrap it in a CPU uint8 ByteTensor
    auto t = torch::empty({NCCL_UNIQUE_ID_BYTES},
                            torch::dtype(torch::kUInt8)
                                .device(torch::kCPU));
    std::memcpy(t.data_ptr(), id.internal, NCCL_UNIQUE_ID_BYTES);
    return t;
}
  

void reduce_scatter_mpi(const torch::Tensor& output_tensor, 
    const torch::Tensor& input_tensor, 
    py::object py_comm,
    const std::string& algorithm = "recursive")
{
    TORCH_CHECK(output_tensor.is_contiguous(), "output tensor must be contiguous.");
    TORCH_CHECK(input_tensor.is_contiguous(), "input tensor must be contiguous.");

    // Ensure 1D tensors.
    TORCH_CHECK(output_tensor.dim() == 1, "output tensor must be 1D");
    TORCH_CHECK(input_tensor.dim() == 1, "input tensor must be 1D");

    // Get MPI rank/size.
    int rank, size;
    // Get reference to base communicator
    MPI_Comm comm = ((PyMPIIntracommObject*)(py_comm.ptr()))->__pyx_base.ob_mpi;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Check that input tensor size is divisible by world size.
    int64_t total_elems = input_tensor.numel();
    TORCH_CHECK(total_elems % size == 0,
    "Input tensor size must be divisible by number of processes");
    int64_t block_size = total_elems / size;

    // Ensure output tensor has exactly one block.
    TORCH_CHECK(output_tensor.numel() == block_size,
    "Output tensor must have block_size elements (input tensor numel()/world_size)");

    // Check type: must be float.
    TORCH_CHECK(input_tensor.scalar_type() == at::kFloat,
    "Input tensor must be of type float");
    TORCH_CHECK(output_tensor.scalar_type() == at::kFloat,
    "Output tensor must be of type float");

    // Get raw device pointers (assumes tensors reside on GPU).
    float* output_ptr = output_tensor.data_ptr<float>();
    float* input_ptr  = input_tensor.data_ptr<float>();
    
    
    // Call the corresponding GPU reduce-scatter algorithm.
    if (algorithm == "recursive") {
        // always use torch tensors. do NOT use malloc.
        // malloc's have high overheads and will slow your communication down
        // torch mallocs memory in advance and manages it internally.
        // therefore these calls are low overheads
        auto tmp_wrkspace_tensor_1 = torch::empty_like(input_tensor);
        auto tmp_wrkspace_tensor_2 = torch::empty_like(input_tensor);
        recursiveHalvingReduceScatterGPU(output_ptr, 
            input_ptr, 
            total_elems,
            tmp_wrkspace_tensor_1.data_ptr<float>(),
            tmp_wrkspace_tensor_2.data_ptr<float>(),
            comm);
    } else if (algorithm == "ring") {
        // always use torch tensors. do NOT use malloc.
        // malloc's have high overheads and will slow your communication down
        // torch mallocs memory in advance and manages it internally.
        // therefore these calls are low overheads
        auto tmp_wrkspace_tensor_1 = torch::empty_like(input_tensor);
        auto tmp_wrkspace_tensor_2 = torch::empty_like(output_tensor);
        auto tmp_wrkspace_tensor_3 = torch::empty_like(output_tensor);

        ringReduceScatterGPU(output_ptr, 
            input_ptr, 
            total_elems, 
            tmp_wrkspace_tensor_1.data_ptr<float>(),
            tmp_wrkspace_tensor_2.data_ptr<float>(),
            tmp_wrkspace_tensor_3.data_ptr<float>(),
            comm);
    } else {
    TORCH_CHECK(false, "Unknown algorithm specified for reduce_scatter_mpi: ", algorithm);
    }
}

void all_gather_mpi(const torch::Tensor& output_tensor, 
    const torch::Tensor& input_tensor, 
    py::object py_comm,
    const std::string& algorithm = "recursive")
{
    TORCH_CHECK(output_tensor.is_contiguous(), "output tensor must be contiguous.");
    TORCH_CHECK(input_tensor.is_contiguous(), "input tensor must be contiguous.");

    // Ensure 1D tensors.
    TORCH_CHECK(output_tensor.dim() == 1, "output tensor must be 1D");
    TORCH_CHECK(input_tensor.dim() == 1, "input tensor must be 1D");

    // Ensure input and output dtypes are the same
    TORCH_CHECK(input_tensor.dtype() == output_tensor.dtype(),
                "Input and output tensors must have the same dtype.");

    // Get MPI rank/size.
    int rank, size;
    // Get reference to base communicator
    MPI_Comm comm = ((PyMPIIntracommObject*)(py_comm.ptr()))->__pyx_base.ob_mpi;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int64_t block_size = input_tensor.numel();
    int64_t total_elems = block_size * size;

    // Ensure output tensor has exactly one block.
    TORCH_CHECK(output_tensor.numel() == total_elems,
    "Output tensor must have total_elem elements (input tensor numel() *world_size)");

    // Get raw device pointers (assumes tensors reside on GPU).
    void* output_ptr = output_tensor.data_ptr();
    const void* input_ptr  = input_tensor.data_ptr();
    
    // Get dtype size for generic handling
    int dtype_size = output_tensor.element_size();  
    
    // Call the corresponding GPU reduce-scatter algorithm.
    if (algorithm == "recursive") {
        // always use torch tensors. do NOT use malloc.
        // malloc's have high overheads and will slow your communication down
        // torch mallocs memory in advance and manages it internally.
        // therefore these calls are low overheads
        auto tmp_wrkspace_tensor_1 = torch::empty_like(output_tensor);
        //auto tmp_wrkspace_tensor_2 = torch::empty_like(input_tensor);
        recursiveDoublingAllGatherGPU(output_ptr, 
            input_ptr, 
            total_elems * dtype_size,
            tmp_wrkspace_tensor_1.data_ptr(),
            //tmp_wrkspace_tensor_2.data_ptr(),
            comm);
    } else {
    TORCH_CHECK(false, "Unknown algorithm specified for all_gather_mpi: ", algorithm);
    }
}

void all_to_all_mpi(const torch::Tensor& output_tensor, 
    const torch::Tensor& input_tensor, 
    py::object py_comm,
    const std::string& algorithm = "spread_out",
    int radix = -1)
{
    TORCH_CHECK(output_tensor.is_contiguous(), "output tensor must be contiguous.");
    TORCH_CHECK(input_tensor.is_contiguous(), "input tensor must be contiguous.");

    // Ensure 1D tensors.
    TORCH_CHECK(output_tensor.dim() == 1, "output tensor must be 1D");
    TORCH_CHECK(input_tensor.dim() == 1, "input tensor must be 1D");

    // Ensure input and output dtypes are the same
    TORCH_CHECK(input_tensor.dtype() == output_tensor.dtype(),
                "Input and output tensors must have the same dtype.");

    // Get MPI rank/size.
    int rank, size;
    // Get reference to base communicator
    MPI_Comm comm = ((PyMPIIntracommObject*)(py_comm.ptr()))->__pyx_base.ob_mpi;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int64_t total_elems = input_tensor.numel();
    TORCH_CHECK(total_elems % size == 0,
    "Input tensor size must be divisible by number of processes");

    // Ensure output tensor has same size as input
    TORCH_CHECK(output_tensor.numel() == total_elems,
    "Output tensor must have same size as input tensor");

    // Get raw device pointers (assumes tensors reside on GPU).
    void* output_ptr = output_tensor.data_ptr();
    const void* input_ptr = input_tensor.data_ptr();
    
    // Get dtype size for generic handling
    int dtype_size = output_tensor.element_size();  
    
    // Call the corresponding GPU all-to-all algorithm.
    if (algorithm == "spread_out") {
        // always use torch tensors. do NOT use malloc.
        // malloc's have high overheads and will slow your communication down
        // torch mallocs memory in advance and manages it internally.
        // therefore these calls are low overheads
        auto tmp_wrkspace_tensor_1 = torch::empty_like(input_tensor);
        auto tmp_wrkspace_tensor_2 = torch::empty_like(input_tensor);
        spreadOutAllToAllGPU(output_ptr, 
            input_ptr, 
            total_elems * dtype_size,
            tmp_wrkspace_tensor_1.data_ptr(),
            tmp_wrkspace_tensor_2.data_ptr(),
            comm);
    } else if (algorithm == "pairwise_exchange") {
        auto tmp_wrkspace_tensor_1 = torch::empty_like(input_tensor);
        auto tmp_wrkspace_tensor_2 = torch::empty_like(input_tensor);
        pairwiseExchangeAllToAllGPU(output_ptr, 
            input_ptr, 
            total_elems * dtype_size,
            tmp_wrkspace_tensor_1.data_ptr(),
            tmp_wrkspace_tensor_2.data_ptr(),
            comm);
    } else if (algorithm == "bruck") {
        auto tmp_wrkspace_tensor_1 = torch::empty_like(input_tensor);
        auto tmp_wrkspace_tensor_2 = torch::empty_like(input_tensor);
        bruckAllToAllGPU(output_ptr, 
            input_ptr, 
            total_elems * dtype_size,
            tmp_wrkspace_tensor_1.data_ptr(),
            tmp_wrkspace_tensor_2.data_ptr(),
            comm);
    } else if (algorithm == "radix_bruck") {
        // Radix-r Bruck algorithm with configurable radix
        if (radix == -1) {
            // Default radix is sqrt(size) for outer group
            radix = static_cast<int>(std::ceil(std::sqrt(size)));
            radix = std::max(radix, 2); // Ensure radix >= 2
        }
        TORCH_CHECK(radix >= 2, "Radix parameter must be >= 2");
        
        auto tmp_wrkspace_tensor_1 = torch::empty_like(input_tensor);
        auto tmp_wrkspace_tensor_2 = torch::empty_like(input_tensor);
        radixRBruckAllToAllGPU(output_ptr, 
            input_ptr, 
            total_elems * dtype_size,
            tmp_wrkspace_tensor_1.data_ptr(),
            tmp_wrkspace_tensor_2.data_ptr(),
            comm,
            radix);
    } else if (algorithm == "uniform_modified_radix_bruck") {
        // Uniform modified radix-r Bruck algorithm with configurable radix
        if (radix == -1) {
            // Default radix is sqrt(size) for outer group
            radix = static_cast<int>(std::ceil(std::sqrt(size)));
            radix = std::max(radix, 2); // Ensure radix >= 2
        }
        TORCH_CHECK(radix >= 2, "Radix parameter must be >= 2");
        
        auto tmp_wrkspace_tensor_1 = torch::empty_like(input_tensor);
        auto tmp_wrkspace_tensor_2 = torch::empty_like(input_tensor);
        uniformModifiedRadixRBruckAllToAllGPU(output_ptr, 
            input_ptr, 
            total_elems * dtype_size,
            tmp_wrkspace_tensor_1.data_ptr(),
            tmp_wrkspace_tensor_2.data_ptr(),
            comm,
            radix);
    } else if (algorithm == "nccl") {
        // For NCCL algorithm, we need to use PyTorch's NCCL backend
        // This is handled differently and should be called from Python side
        TORCH_CHECK(false, "NCCL algorithm should be handled on Python side with PyTorch distributed");
    } else {
    TORCH_CHECK(false, "Unknown algorithm specified for all_to_all_mpi: ", algorithm);
    }
}

void all_to_all_nccl(const torch::Tensor& output_tensor, 
    const torch::Tensor& input_tensor, 
    std::shared_ptr<NcclComm> nccl_comm,
    int rank,
    int size,
    const std::string& algorithm = "nccl")
{
    TORCH_CHECK(output_tensor.is_contiguous(), "output tensor must be contiguous.");
    TORCH_CHECK(input_tensor.is_contiguous(), "input tensor must be contiguous.");

    // Ensure 1D tensors
    TORCH_CHECK(output_tensor.dim() == 1, "output tensor must be 1D");
    TORCH_CHECK(input_tensor.dim() == 1, "input tensor must be 1D");

    // Ensure input and output dtypes are the same
    TORCH_CHECK(input_tensor.dtype() == output_tensor.dtype(),
    "Input and output tensors must have the same dtype.");

    // Ensure same size
    TORCH_CHECK(output_tensor.numel() == input_tensor.numel(),
    "Input and output tensors must have same size");

    // auto stream = at::cuda::getCurrentCUDAStream();

    void* output_ptr = output_tensor.data_ptr();
    const void* input_ptr = input_tensor.data_ptr();
    int dtype_size = output_tensor.element_size();
    int64_t total_elems = input_tensor.numel();

    ncclAllToAllGPU(output_ptr, 
        input_ptr, 
        total_elems * dtype_size,
        rank,
        size,
        nccl_comm->comm);
}

void all_to_all_nccl_p2p(const torch::Tensor& output_tensor, 
    const torch::Tensor& input_tensor, 
    std::shared_ptr<NcclComm> nccl_comm,
    int rank,
    int size,
    const std::string& algorithm = "spread_out",
    int radix = -1)
{
    TORCH_CHECK(output_tensor.is_contiguous(), "output tensor must be contiguous.");
    TORCH_CHECK(input_tensor.is_contiguous(), "input tensor must be contiguous.");

    // Ensure 1D tensors
    TORCH_CHECK(output_tensor.dim() == 1, "output tensor must be 1D");
    TORCH_CHECK(input_tensor.dim() == 1, "input tensor must be 1D");

    // Ensure input and output dtypes are the same
    TORCH_CHECK(input_tensor.dtype() == output_tensor.dtype(),
    "Input and output tensors must have the same dtype.");

    // Ensure same size
    TORCH_CHECK(output_tensor.numel() == input_tensor.numel(),
    "Input and output tensors must have same size");

    void* output_ptr = output_tensor.data_ptr();
    const void* input_ptr = input_tensor.data_ptr();
    int dtype_size = output_tensor.element_size();
    int64_t total_elems = input_tensor.numel();

    // Allocate temporary buffers for NCCL P2P algorithms
    torch::Tensor send_buf = torch::empty_like(input_tensor);
    torch::Tensor recv_buf = torch::empty_like(input_tensor);
    void* send_buf_ptr = send_buf.data_ptr();
    void* recv_buf_ptr = recv_buf.data_ptr();

    if (algorithm == "spread_out") {
        ncclSpreadOutAllToAllGPU(output_ptr, input_ptr, total_elems * dtype_size, 
                                 send_buf_ptr, recv_buf_ptr, rank, size, nccl_comm->comm);
    } else if (algorithm == "pairwise_exchange") {
        ncclPairwiseExchangeAllToAllGPU(output_ptr, input_ptr, total_elems * dtype_size, 
                                        send_buf_ptr, recv_buf_ptr, rank, size, nccl_comm->comm);
    } else if (algorithm == "bruck") {
        ncclBruckAllToAllGPU(output_ptr, input_ptr, total_elems * dtype_size, 
                             send_buf_ptr, recv_buf_ptr, rank, size, nccl_comm->comm);
    } else if (algorithm == "radix_bruck") {
        // Calculate default radix as sqrt(size) if not specified
        int actual_radix = radix;
        if (radix == -1) {
            actual_radix = std::max(2, static_cast<int>(std::ceil(std::sqrt(size))));
        }
        ncclRadixRBruckAllToAllGPU(output_ptr, input_ptr, total_elems * dtype_size, 
                                   send_buf_ptr, recv_buf_ptr, rank, size, nccl_comm->comm, actual_radix);
    } else if (algorithm == "uniform_modified_radix_bruck") {
        // Calculate default radix as sqrt(size) if not specified
        int actual_radix = radix;
        if (radix == -1) {
            actual_radix = std::max(2, static_cast<int>(std::ceil(std::sqrt(size))));
        }
        ncclUniformModifiedRadixRBruckAllToAllGPU(output_ptr, input_ptr, total_elems * dtype_size, 
                                                  send_buf_ptr, recv_buf_ptr, rank, size, nccl_comm->comm, actual_radix);
    } else {
        TORCH_CHECK(false, "Unknown NCCL P2P algorithm specified for all_to_all_nccl_p2p: ", algorithm);
    }
}

void pack_all_to_allv_stage2(
    const torch::Tensor& output_intermediate,
    torch::Tensor send_buffer,
    const torch::Tensor& comm_matrix,
    const torch::Tensor& recv_base_offsets,
    const torch::Tensor& recv_node_offsets,
    const torch::Tensor& stage2_send_displs,
    const torch::Tensor& inter_sendcounts,
    int intra_node_group_size,
    int inter_node_group_size,
    int my_node_idx,
    int my_intra_rank) {

    if (send_buffer.numel() == 0) {
        return;
    }

    TORCH_CHECK(output_intermediate.is_cuda(), "output_intermediate must be a CUDA tensor");
    TORCH_CHECK(send_buffer.is_cuda(), "send_buffer must be a CUDA tensor");
    TORCH_CHECK(comm_matrix.device().is_cpu() && comm_matrix.dtype() == torch::kInt64,
                "comm_matrix must be an int64 CPU tensor");
    TORCH_CHECK(recv_base_offsets.device().is_cpu() && recv_base_offsets.dtype() == torch::kInt64,
                "recv_base_offsets must be an int64 CPU tensor");
    TORCH_CHECK(recv_node_offsets.device().is_cpu() && recv_node_offsets.dtype() == torch::kInt64,
                "recv_node_offsets must be an int64 CPU tensor");
    TORCH_CHECK(stage2_send_displs.device().is_cpu() && stage2_send_displs.dtype() == torch::kInt64,
                "stage2_send_displs must be an int64 CPU tensor");
    TORCH_CHECK(inter_sendcounts.device().is_cpu() && inter_sendcounts.dtype() == torch::kInt64,
                "inter_sendcounts must be an int64 CPU tensor");

    TORCH_CHECK(comm_matrix.dim() == 2, "comm_matrix must be 2D");
    TORCH_CHECK(recv_node_offsets.dim() == 2, "recv_node_offsets must be 2D");

    const int64_t world_size = comm_matrix.size(0);
    TORCH_CHECK(world_size == comm_matrix.size(1),
                "comm_matrix must be square with size equal to world size");
    TORCH_CHECK(world_size == static_cast<int64_t>(intra_node_group_size) * inter_node_group_size,
                "comm_matrix size mismatch with provided intra/inter node sizes");
    TORCH_CHECK(recv_base_offsets.numel() == intra_node_group_size + 1,
                "recv_base_offsets length mismatch");
    TORCH_CHECK(recv_node_offsets.size(0) == intra_node_group_size &&
                recv_node_offsets.size(1) == inter_node_group_size + 1,
                "recv_node_offsets shape mismatch");
    TORCH_CHECK(stage2_send_displs.numel() == inter_node_group_size,
                "stage2_send_displs length mismatch");
    TORCH_CHECK(inter_sendcounts.numel() == inter_node_group_size,
                "inter_sendcounts length mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();
    const auto elem_size = output_intermediate.element_size();
    char* send_ptr_base = static_cast<char*>(send_buffer.data_ptr());
    const char* src_ptr_base = static_cast<const char*>(output_intermediate.data_ptr());

    const int64_t* comm_ptr = comm_matrix.data_ptr<int64_t>();
    const int64_t* base_offsets_ptr = recv_base_offsets.data_ptr<int64_t>();
    const int64_t* node_offsets_ptr = recv_node_offsets.data_ptr<int64_t>();
    const int64_t* send_displs_ptr = stage2_send_displs.data_ptr<int64_t>();
    const int64_t* sendcounts_ptr = inter_sendcounts.data_ptr<int64_t>();

    const int node_offset_stride = inter_node_group_size + 1;

    std::vector<int64_t> write_positions(inter_node_group_size);
    for (int dest_node = 0; dest_node < inter_node_group_size; ++dest_node) {
        write_positions[dest_node] = send_displs_ptr[dest_node];
    }

    for (int dest_node = 0; dest_node < inter_node_group_size; ++dest_node) {
        int64_t write_offset = write_positions[dest_node];
        const int dest_rank = dest_node * intra_node_group_size + my_intra_rank;

        for (int src_local_rank = 0; src_local_rank < intra_node_group_size; ++src_local_rank) {
            const int src_rank = my_node_idx * intra_node_group_size + src_local_rank;
            const int64_t count = comm_ptr[src_rank * world_size + dest_rank];
            if (count <= 0) {
                continue;
            }

            const int64_t read_offset = base_offsets_ptr[src_local_rank] +
                                        node_offsets_ptr[src_local_rank * node_offset_stride + dest_node];

            if (src_local_rank == my_intra_rank) {
                continue;
            }

            char* dst = send_ptr_base + write_offset * elem_size;
            const char* src = src_ptr_base + read_offset * elem_size;
            CUDA_CHECK(cudaMemcpyAsync(dst, src, static_cast<size_t>(count) * elem_size,
                                       cudaMemcpyDeviceToDevice, stream));
            write_offset += count;
        }

        TORCH_CHECK(write_offset == send_displs_ptr[dest_node] + sendcounts_ptr[dest_node],
                    "send buffer fill mismatch for dest_node ", dest_node);
    }
}

void all_to_allv_mpi(const torch::Tensor& output_tensor, 
                    const torch::Tensor& input_tensor,
                    const std::vector<int>& sendcounts,
                    const std::vector<int>& recvcounts,
                    const std::vector<int>& send_displs,
                    const std::vector<int>& recv_displs,
                    py::object py_comm,
                    const std::string& algorithm = "pairwise_sendrecv")
{
    TORCH_CHECK(output_tensor.is_contiguous(), "output tensor must be contiguous.");
    TORCH_CHECK(input_tensor.is_contiguous(), "input tensor must be contiguous.");

    // Ensure 1D tensors.
    TORCH_CHECK(output_tensor.dim() == 1, "output tensor must be 1D");
    TORCH_CHECK(input_tensor.dim() == 1, "input tensor must be 1D");

    // Ensure input and output dtypes are the same
    TORCH_CHECK(input_tensor.dtype() == output_tensor.dtype(),
                "Input and output tensors must have the same dtype.");

    // Get MPI rank/size.
    int rank, size;
    // Get reference to base communicator
    MPI_Comm comm = ((PyMPIIntracommObject*)(py_comm.ptr()))->__pyx_base.ob_mpi;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Validate count arrays
    TORCH_CHECK(sendcounts.size() == size, "sendcounts must have size equal to number of processes");
    TORCH_CHECK(recvcounts.size() == size, "recvcounts must have size equal to number of processes");
    TORCH_CHECK(send_displs.size() == size, "send_displs must have size equal to number of processes");
    TORCH_CHECK(recv_displs.size() == size, "recv_displs must have size equal to number of processes");

    // Get raw device pointers (assumes tensors reside on GPU).
    void* output_ptr = output_tensor.data_ptr();
    const void* input_ptr = input_tensor.data_ptr();
    
    // Get dtype size for generic handling
    int dtype_size = output_tensor.element_size();  

    // Convert sendcounts and recvcounts to byte counts
    std::vector<int> send_bytes(size), recv_bytes(size);
    std::vector<int> send_byte_displs(size), recv_byte_displs(size);
    
    // for (int i = 0; i < size; i++) {
    //     send_bytes[i] = sendcounts[i] * dtype_size;
    //     recv_bytes[i] = recvcounts[i] * dtype_size;
    //     send_byte_displs[i] = send_displs[i] * dtype_size;
    //     recv_byte_displs[i] = recv_displs[i] * dtype_size;
    //     if (send_bytes[i] < 0 || recv_bytes[i] < 0 || send_byte_displs[i] < 0 || recv_byte_displs[i] < 0) {
    //         TORCH_CHECK(false, "Send or recv byte count or displacement is negative for rank ", i);
    //     }
    // }
    // long long input_size_bytes = input_tensor.numel() * dtype_size;
    // long long output_size_bytes = output_tensor.numel() * dtype_size;
    
    // // Call the corresponding GPU all-to-allv algorithm.
    // if (algorithm == "pairwise_sendrecv") {
    //     AllToAllvGPU_pairwise_sendrecv(output_ptr, input_ptr, send_bytes.data(), recv_bytes.data(), 
    //                                   send_byte_displs.data(), recv_byte_displs.data(), rank, size, comm);   
    // } else if (algorithm == "pairwise_exchange") {
    //     AllToAllvGPU_pairwise_exchange(output_ptr, input_ptr, send_bytes.data(), recv_bytes.data(), 
    //                                   send_byte_displs.data(), recv_byte_displs.data(), rank, size, comm);   
    // } else if (algorithm == "pairwise_scattered") {
    //     AllToAllvGPU_pairwise_scattered(output_ptr, input_ptr, send_bytes.data(), recv_bytes.data(), 
    //                                   send_byte_displs.data(), recv_byte_displs.data(), rank, size, comm);   
    // } else if (algorithm == "pairwise_openmpi") {
    //     AllToAllvGPU_openmpi_pairwise(output_ptr, input_ptr, send_bytes.data(), recv_bytes.data(), 
    //                                   send_byte_displs.data(), recv_byte_displs.data(), rank, size, comm);   
    // } else if (algorithm == "openmpi_basic_linear") {
    //     AllToAllvGPU_openmpi_basic_linear(output_ptr, input_ptr, send_bytes.data(), recv_bytes.data(), 
    //                                   send_byte_displs.data(), recv_byte_displs.data(), rank, size, comm);   
    // } else if (algorithm == "openmpi_basic_inplace") {
    //     AllToAllvGPU_openmpi_basic_inplace(output_ptr, input_ptr, send_bytes.data(), recv_bytes.data(), 
    //                                   send_byte_displs.data(), recv_byte_displs.data(), rank, size, comm);   
    // } else if (algorithm == "openmpi_inter") {
    //     AllToAllvGPU_openmpi_inter(output_ptr, input_ptr, send_bytes.data(), recv_bytes.data(), 
    //                                   send_byte_displs.data(), recv_byte_displs.data(), rank, size, comm);   
    // } else if (algorithm == "openmpi_persistent") {
    //     AllToAllvGPU_openmpi_persistent(output_ptr, input_ptr, send_bytes.data(), recv_bytes.data(), 
    //                                   send_byte_displs.data(), recv_byte_displs.data(), rank, size, comm);   
    // } else 
    {
        // default to pairwise_sendrecv
        // AllToAllvGPU_pairwise_sendrecv(output_ptr, input_ptr, send_bytes.data(), recv_bytes.data(), 
        //                               send_byte_displs.data(), recv_byte_displs.data(), rank, size, comm);   

        AllToAllvGPU_pairwise_sendrecv_datatype(output_ptr, input_ptr, sendcounts.data(), recvcounts.data(), 
                                      send_displs.data(), recv_displs.data(), dtype_size, rank, size, comm);   
    }

}

void all_to_allv_nccl_p2p(const torch::Tensor& output_tensor, 
                          const torch::Tensor& input_tensor,
                          const std::vector<int>& sendcounts,
                          const std::vector<int>& recvcounts,
                          const std::vector<int>& send_displs,
                          const std::vector<int>& recv_displs,
                          py::object nccl_comm_obj,
                          int rank,
                          int size)
{
    TORCH_CHECK(output_tensor.is_contiguous(), "output tensor must be contiguous.");
    TORCH_CHECK(input_tensor.is_contiguous(), "input tensor must be contiguous.");

    // Ensure 1D tensors.
    TORCH_CHECK(output_tensor.dim() == 1, "output tensor must be 1D");
    TORCH_CHECK(input_tensor.dim() == 1, "input tensor must be 1D");

    // Ensure input and output dtypes are the same
    TORCH_CHECK(input_tensor.dtype() == output_tensor.dtype(),
                "Input and output tensors must have the same dtype.");

    // Validate count arrays
    TORCH_CHECK(sendcounts.size() == size, "sendcounts must have size equal to number of processes");
    TORCH_CHECK(recvcounts.size() == size, "recvcounts must have size equal to number of processes");
    TORCH_CHECK(send_displs.size() == size, "send_displs must have size equal to number of processes");
    TORCH_CHECK(recv_displs.size() == size, "recv_displs must have size equal to number of processes");

    // Get dtype size for generic handling
    int dtype_size = output_tensor.element_size();
    
    // Convert sendcounts and recvcounts to byte counts
    std::vector<int> send_bytes(size), recv_bytes(size);
    std::vector<int> send_byte_displs(size), recv_byte_displs(size);
    
    for (int i = 0; i < size; i++) {
        send_bytes[i] = sendcounts[i] * dtype_size;
        recv_bytes[i] = recvcounts[i] * dtype_size;
        send_byte_displs[i] = send_displs[i] * dtype_size;
        recv_byte_displs[i] = recv_displs[i] * dtype_size;
    }

    // Get NCCL communicator from Python object
    std::shared_ptr<NcclComm> nccl_comm = nccl_comm_obj.cast<std::shared_ptr<NcclComm>>();
    TORCH_CHECK(nccl_comm, "Failed to get NCCL communicator from Python object");

    // Get raw device pointers and tensor sizes
    void* output_ptr = output_tensor.data_ptr();
    const void* input_ptr = input_tensor.data_ptr();
    long long input_size_bytes = input_tensor.numel() * dtype_size;
    long long output_size_bytes = output_tensor.numel() * dtype_size;
    
    // Validate bounds for all ranks
    for (int i = 0; i < size; i++) {
        if (send_byte_displs[i] + send_bytes[i] > input_size_bytes) {
            TORCH_CHECK(false, "Send displacement + size exceeds input tensor bounds for rank ", i,
                       ": ", send_byte_displs[i], " + ", send_bytes[i], " > ", input_size_bytes);
        }
        if (recv_byte_displs[i] + recv_bytes[i] > output_size_bytes) {
            TORCH_CHECK(false, "Recv displacement + size exceeds output tensor bounds for rank ", i,
                       ": ", recv_byte_displs[i], " + ", recv_bytes[i], " > ", output_size_bytes);
        }
    }
    
    // Call the NCCL P2P all-to-allv algorithm
    ncclAllToAllvGPU(output_ptr, 
    // ncclSpreadOutAllToAllvGPU(output_ptr, 
                              input_ptr, 
                              send_bytes.data(),
                              recv_bytes.data(),
                              send_byte_displs.data(),
                              recv_byte_displs.data(),
                              rank,
                              size,
                              nccl_comm->comm);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_scatter_mpi", reduce_scatter_mpi);
    m.def("all_gather_mpi", all_gather_mpi);
    m.def("all_to_all_mpi", all_to_all_mpi);
    m.def("all_to_all_nccl", all_to_all_nccl);
    m.def("all_to_all_nccl_p2p", all_to_all_nccl_p2p);
    m.def("all_to_allv_nccl_p2p", all_to_allv_nccl_p2p);
    m.def("all_to_allv_mpi", all_to_allv_mpi);
    m.def("pack_all_to_allv_stage2", pack_all_to_allv_stage2);
    m.def("permute_data", &permute_data_wrapper, "Optimized GPU data permutation");
    m.def("get_nccl_unique_id", &get_nccl_unique_id,
      "Return a NCCL unique_id as a CPU ByteTensor");
    py::class_<NcclComm, std::shared_ptr<NcclComm>>(m, "NcclComm")
      .def(py::init<const torch::Tensor&, int, int>());
}
