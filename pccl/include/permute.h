#pragma once

#include <torch/extension.h>

void permute_data_wrapper(
    torch::Tensor output,
    const torch::Tensor input,
    const torch::Tensor offsets);
