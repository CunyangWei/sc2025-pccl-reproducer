#include <torch/extension.h>
#include <torch/torch.h>
#include <pybind11/pybind11.h> // PyBind11
#include <mpi4py/mpi4py.h>
#include "reduce_scatter.h"
#include "all_gather.h"
#include "all_to_all.h"
#include "common.h"
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp> // PyTorch C++ 头文件
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp> // NCCL 后端的头文件

namespace py = pybind11;

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
    const std::string& algorithm = "spread_out")
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
    } else if (algorithm == "ring") {
        auto tmp_wrkspace_tensor_1 = torch::empty_like(input_tensor);
        auto tmp_wrkspace_tensor_2 = torch::empty_like(input_tensor);
        ringAllToAllGPU(output_ptr, 
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
    uintptr_t nccl_comm_ptr)
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

    TORCH_CHECK(nccl_comm_ptr != 0, "NCCL communicator pointer is null");
    
    ncclComm_t nccl_comm = reinterpret_cast<ncclComm_t>(nccl_comm_ptr);
    
    TORCH_CHECK(nccl_comm != nullptr, "NCCL communicator is null after conversion");

    auto stream = at::cuda::getCurrentCUDAStream();

    void* output_ptr = output_tensor.data_ptr();
    const void* input_ptr = input_tensor.data_ptr();
    int dtype_size = output_tensor.element_size();
    int64_t total_elems = input_tensor.numel();

    ncclAllToAllGPU(output_ptr, 
        input_ptr, 
        total_elems * dtype_size,
        nccl_comm,
        stream);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_scatter_mpi", reduce_scatter_mpi);
    m.def("all_gather_mpi", all_gather_mpi);
    m.def("all_to_all_mpi", all_to_all_mpi);
    m.def("all_to_all_nccl", all_to_all_nccl);
}
