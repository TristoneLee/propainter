// kernels/residual_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void add_residual_launcher(
    void* out, const void* x, const void* y, const float* bias,
    long long n, long long C, cudaStream_t stream);
extern "C" void to_half_launcher(void* out, const float* x, long long n,
                                 cudaStream_t stream);

// out = x (fp16) + y (fp16) [+ bias (fp32) per last-dim channel], same shape
static torch::Tensor add_residual(torch::Tensor x, torch::Tensor y,
                                  c10::optional<torch::Tensor> bias) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == torch::kFloat16, "bad x");
    TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.dtype() == torch::kFloat16, "bad y");
    TORCH_CHECK(x.numel() == y.numel(), "numel mismatch");
    TORCH_CHECK(x.numel() % 4 == 0, "numel must be divisible by 4");
    int64_t C = x.size(-1);
    const float* bptr = nullptr;
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() &&
                    bias->dtype() == torch::kFloat32 && bias->numel() == C, "bad bias");
        bptr = bias->data_ptr<float>();
    }
    auto out = torch::empty(x.sizes(), x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_residual_launcher(out.data_ptr(), x.data_ptr(), y.data_ptr(),
                          bptr, x.numel(), C, stream);
    return out;
}

static torch::Tensor to_half(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == torch::kFloat32, "bad x");
    auto out = torch::empty(x.sizes(), x.options().dtype(torch::kFloat16));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    to_half_launcher(out.data_ptr(), x.data_ptr<float>(), x.numel(), stream);
    return out;
}

static void register_residual(pybind11::module& m) {
    m.def("add_residual", &add_residual, py::arg("x"), py::arg("y"),
          py::arg("bias") = py::none());
    m.def("to_half", &to_half, py::arg("x"));
}

REGISTER_BINDING(residual, register_residual);
