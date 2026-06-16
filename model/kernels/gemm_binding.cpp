// kernels/gemm_binding.cpp - PyTorch bindings for cuBLASLt linear ops (fp16 in).
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" int lt_linear_hx(cudaStream_t stream, void* out, const void* in,
                            const void* weight, const void* bias, const void* residual,
                            long long N, long long K, long long M, int out_half);

static void check_h(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dtype() == torch::kFloat16, name, " must be float16");
}

static void check_f(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

static torch::Tensor linear_common(torch::Tensor input, torch::Tensor weight,
                                   torch::Tensor bias, const torch::Tensor* residual,
                                   bool out_half) {
    check_h(input, "input");
    check_h(weight, "weight");
    if (out_half) check_h(bias, "bias");
    else check_f(bias, "bias");
    long long K = weight.size(1);
    long long M = weight.size(0);
    TORCH_CHECK(input.size(-1) == K, "input last dim must equal weight.size(1)");
    TORCH_CHECK(bias.numel() == M, "bias size mismatch");
    long long N = input.numel() / K;

    auto sizes = input.sizes().vec();
    sizes.back() = M;
    auto out = torch::empty(sizes, input.options().dtype(out_half ? torch::kFloat16
                                                                  : torch::kFloat32));
    const void* res_ptr = nullptr;
    if (residual != nullptr) {
        check_f(*residual, "residual");
        TORCH_CHECK(!out_half, "residual requires fp32 output");
        TORCH_CHECK(residual->numel() == out.numel(), "residual numel mismatch");
        res_ptr = residual->data_ptr();
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    int st = lt_linear_hx(stream, out.data_ptr(), input.data_ptr(), weight.data_ptr(),
                          bias.data_ptr(), res_ptr, N, K, M, out_half ? 1 : 0);
    TORCH_CHECK(st == 0, "cublasLt matmul failed with status ", st);
    return out;
}

// fp16 in, fp16 out, fp16 bias
static torch::Tensor linear(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    return linear_common(input, weight, bias, nullptr, true);
}

static void register_gemm(pybind11::module& m) {
    m.def("linear", &linear, py::arg("input"), py::arg("weight"), py::arg("bias"));
}

REGISTER_BINDING(gemm, register_gemm);
