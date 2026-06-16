// kernels/layernorm_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <tuple>
#include "../binding_registry.h"

extern "C" void layernorm_pad_launcher(
    void* out, const void* x, const float* gamma, const float* beta,
    long long bt, long long H_in, long long H_out, long long W_in, long long W_out,
    long long C, float eps, cudaStream_t stream);

// x: 5D [b, t, h, w, c] fp16 -> fp16 output. pad_h==0 && pad_w==0 -> same 5D shape.
// otherwise -> 4D [b*t, h+pad_h, w+pad_w, c] with zero-filled pad tokens.
static torch::Tensor layernorm(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                               int64_t pad_w, int64_t pad_h) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == torch::kFloat16, "bad x");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(), "bad weight");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous(), "bad bias");
    TORCH_CHECK(x.dim() == 5, "x must be 5D [b,t,h,w,c]");
    int64_t b = x.size(0), t = x.size(1), h = x.size(2), w = x.size(3), c = x.size(4);
    TORCH_CHECK(c % 4 == 0, "C must be divisible by 4");
    TORCH_CHECK(weight.numel() == c && bias.numel() == c, "norm param size mismatch");

    auto opts = x.options().dtype(torch::kFloat16);
    torch::Tensor out;
    if (pad_h > 0 || pad_w > 0) {
        out = torch::empty({b * t, h + pad_h, w + pad_w, c}, opts);
    } else {
        out = torch::empty({b, t, h, w, c}, opts);
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    layernorm_pad_launcher(out.data_ptr(), x.data_ptr(),
                           weight.data_ptr<float>(), bias.data_ptr<float>(),
                           b * t, h, h + pad_h, w, w + pad_w, c, 1e-5f, stream);
    return out;
}

extern "C" void residual_norm_launcher(
    void* x_new, void* norm_out, const void* x, const void* branch,
    const float* bias2, const float* gamma, const float* beta,
    long long bt, long long H, long long H_out, long long Hb, long long W,
    long long Wb, long long W_out, long long C, float eps, cudaStream_t stream);

// Fused: x_new = x + branch (+ bias2); norm_out = LayerNorm(x_new).
// x: [b,t,h,w,c] fp16. branch: fp16, [bt,h,Wb,c] (Wb >= w) or same 5D shape.
// x_new keeps the unpadded 5D shape; norm_out is [bt,h+pad_h,w+pad_w,c] (4D) when
// either pad is set, else the same 5D shape as x.
static std::tuple<torch::Tensor, torch::Tensor> residual_norm(
    torch::Tensor x, torch::Tensor branch, torch::Tensor weight, torch::Tensor bias,
    int64_t pad_w, c10::optional<torch::Tensor> bias2, int64_t pad_h) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == torch::kFloat16, "bad x");
    TORCH_CHECK(branch.is_cuda() && branch.is_contiguous() &&
                branch.dtype() == torch::kFloat16, "bad branch");
    TORCH_CHECK(x.dim() == 5, "x must be 5D");
    int64_t b = x.size(0), t = x.size(1), h = x.size(2), w = x.size(3), c = x.size(4);
    int64_t Wb, Hb;
    if (branch.dim() == 4) {
        Hb = branch.size(1);
        Wb = branch.size(2);
        TORCH_CHECK(branch.size(0) == b * t && Hb >= h && Wb >= w &&
                    branch.size(3) == c, "branch shape mismatch");
    } else {
        TORCH_CHECK(branch.sizes() == x.sizes(), "branch shape mismatch");
        Hb = h; Wb = w;
    }
    auto x_new = torch::empty(x.sizes(), x.options());
    auto hopts = x.options().dtype(torch::kFloat16);
    torch::Tensor norm_out;
    if (pad_h > 0 || pad_w > 0) norm_out = torch::empty({b * t, h + pad_h, w + pad_w, c}, hopts);
    else norm_out = torch::empty(x.sizes(), hopts);

    const float* b2 = nullptr;
    if (bias2.has_value()) {
        TORCH_CHECK(bias2->is_cuda() && bias2->is_contiguous() &&
                    bias2->dtype() == torch::kFloat32 && bias2->numel() == c, "bad bias2");
        b2 = bias2->data_ptr<float>();
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    residual_norm_launcher(x_new.data_ptr(), norm_out.data_ptr(),
                           x.data_ptr(), branch.data_ptr(), b2,
                           weight.data_ptr<float>(), bias.data_ptr<float>(),
                           b * t, h, h + pad_h, Hb, w, Wb, w + pad_w, c, 1e-5f, stream);
    return std::make_tuple(x_new, norm_out);
}

static void register_layernorm(pybind11::module& m) {
    m.def("layernorm", &layernorm, py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("pad_w") = 0, py::arg("pad_h") = 0);
    m.def("residual_norm", &residual_norm, py::arg("x"), py::arg("branch"),
          py::arg("weight"), py::arg("bias"), py::arg("pad_w") = 0,
          py::arg("bias2") = py::none(), py::arg("pad_h") = 0);
}

REGISTER_BINDING(layernorm, register_layernorm);
