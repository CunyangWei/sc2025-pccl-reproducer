#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "common.h"
#include "permute.h"

void permute_data_wrapper(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor offsets)
{
    TORCH_CHECK(output.is_cuda(), "output tensor must reside on CUDA device");
    TORCH_CHECK(input.is_cuda(), "input tensor must reside on CUDA device");
    TORCH_CHECK(offsets.is_cuda(), "offset tensor must reside on CUDA device");
    TORCH_CHECK(output.is_contiguous(), "output tensor must be contiguous");
    TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous");
    TORCH_CHECK(offsets.dim() == 2 && offsets.size(1) == 3, "offsets must have shape [N, 3]");
    TORCH_CHECK(offsets.scalar_type() == at::kLong, "offsets tensor must be int64");

    if (offsets.size(0) == 0) {
        return;
    }

    TORCH_CHECK(input.scalar_type() == output.scalar_type(),
                "input and output tensors must share dtype");

    auto offsets_cpu = offsets.to(torch::kCPU, offsets.scalar_type(), /*non_blocking=*/false, /*copy=*/true);
    const auto* offsets_ptr = offsets_cpu.data_ptr<int64_t>();

    auto stream = at::cuda::getCurrentCUDAStream();
    const auto elem_size = static_cast<size_t>(output.element_size());
    const char* input_ptr = static_cast<const char*>(input.data_ptr());
    char* output_ptr = static_cast<char*>(output.data_ptr());

    const int64_t num_blocks = offsets_cpu.size(0);
    for (int64_t i = 0; i < num_blocks; ++i) {
        const int64_t read_offset = offsets_ptr[i * 3 + 0];
        const int64_t write_offset = offsets_ptr[i * 3 + 1];
        const int64_t count = offsets_ptr[i * 3 + 2];

        if (count <= 0) {
            continue;
        }

        const char* src = input_ptr + read_offset * elem_size;
        char* dst = output_ptr + write_offset * elem_size;
        CUDA_CHECK(cudaMemcpyAsync(dst, src, static_cast<size_t>(count) * elem_size,
                                   cudaMemcpyDeviceToDevice, stream));
    }
}
