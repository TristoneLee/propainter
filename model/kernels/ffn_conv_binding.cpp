// kernels/ffn_conv_binding.cpp - conv-formulated FusionFeedForward bindings.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" int conv_fold_bwd(cudaStream_t stream, void* dx, const void* dy, const void* K,
                             long long bt, int fh, int fw, int Cm, int OH, int OW, int CO,
                             int kh, int kw, int sh, int sw, int ph, int pw);
extern "C" int conv_unfold_fwd(cudaStream_t stream, void* out, const void* x, const void* K,
                               long long bt, int OH, int OW, int CO, int fh, int fw, int Cm,
                               int kh, int kw, int sh, int sw, int ph, int pw);
extern "C" void gelu_div_bias_launcher(
    void* out, const void* raw, const float* bias1, long long bt,
    int fh, int fw, int OH, int OW, int kh, int kw, int sh, int sw,
    int ph, int pw, int CO, cudaStream_t stream);
extern "C" void w1_filter_launcher(void* out, const float* w, int M, int kh, int kw,
                                   int CO, cudaStream_t stream);
extern "C" void w2_filter_launcher(void* out, const float* w, int M, int kh, int kw,
                                   int CO, cudaStream_t stream);

// y: [b, t, fh, fw, Cm] fp16 (norm2 output); K1: [Cm, kh, kw, CO] fp16.
// Returns the raw folded map [bt, OH, OW, CO] fp16 (bias/div/gelu NOT applied).
static torch::Tensor ffn_fold_conv(torch::Tensor y, torch::Tensor K1,
                                   int64_t OH, int64_t OW,
                                   int64_t kh, int64_t kw, int64_t sh, int64_t sw,
                                   int64_t ph, int64_t pw) {
    TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.dtype() == torch::kFloat16, "bad y");
    TORCH_CHECK(K1.is_cuda() && K1.is_contiguous() && K1.dtype() == torch::kFloat16, "bad K1");
    TORCH_CHECK(y.dim() == 5, "y must be 5D");
    int64_t b = y.size(0), t = y.size(1), fh = y.size(2), fw = y.size(3), Cm = y.size(4);
    int64_t CO = K1.size(3);
    TORCH_CHECK(K1.size(0) == Cm && K1.size(1) == kh && K1.size(2) == kw, "K1 shape");
    TORCH_CHECK((OH + 2 * ph - kh) / sh + 1 == fh, "geometry mismatch (h)");
    TORCH_CHECK((OW + 2 * pw - kw) / sw + 1 == fw, "geometry mismatch (w)");

    auto out = torch::empty({b * t, OH, OW, CO}, y.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    int st = conv_fold_bwd(stream, out.data_ptr(), y.data_ptr(), K1.data_ptr(),
                           b * t, (int)fh, (int)fw, (int)Cm, (int)OH, (int)OW, (int)CO,
                           (int)kh, (int)kw, (int)sh, (int)sw, (int)ph, (int)pw);
    TORCH_CHECK(st == 0, "cudnn backward-data failed: ", st);
    return out;
}

// fd2 = gelu((raw + fold(bias1)) / count), same shape as raw.
static torch::Tensor ffn_gelu_div(torch::Tensor raw, torch::Tensor bias1, int64_t fh,
                                  int64_t fw, int64_t kh, int64_t kw, int64_t sh,
                                  int64_t sw, int64_t ph, int64_t pw) {
    TORCH_CHECK(raw.is_cuda() && raw.is_contiguous() && raw.dtype() == torch::kFloat16, "bad raw");
    TORCH_CHECK(bias1.is_cuda() && bias1.is_contiguous() &&
                bias1.dtype() == torch::kFloat32, "bad bias1");
    TORCH_CHECK(raw.dim() == 4, "raw must be [bt,OH,OW,CO]");
    int64_t bt = raw.size(0), OH = raw.size(1), OW = raw.size(2), CO = raw.size(3);
    TORCH_CHECK(bias1.numel() == CO * kh * kw, "bias1 size");

    auto out = torch::empty(raw.sizes(), raw.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gelu_div_bias_launcher(out.data_ptr(), raw.data_ptr(), bias1.data_ptr<float>(), bt,
                           (int)fh, (int)fw, (int)OH, (int)OW, (int)kh, (int)kw,
                           (int)sh, (int)sw, (int)ph, (int)pw, (int)CO, stream);
    return out;
}

// fd2: [bt, OH, OW, CO] fp16; K2: [Cm, kh, kw, CO] fp16 -> [b, bt/b, fh, fw, Cm]
// fp16 (fc2 bias NOT applied; folded into the residual add).
static torch::Tensor ffn_unfold_conv(torch::Tensor fd2, torch::Tensor K2, int64_t b,
                                     int64_t kh, int64_t kw, int64_t sh, int64_t sw,
                                     int64_t ph, int64_t pw) {
    TORCH_CHECK(fd2.is_cuda() && fd2.is_contiguous() && fd2.dtype() == torch::kFloat16, "bad fd2");
    TORCH_CHECK(K2.is_cuda() && K2.is_contiguous() && K2.dtype() == torch::kFloat16, "bad K2");
    int64_t bt = fd2.size(0), OH = fd2.size(1), OW = fd2.size(2), CO = fd2.size(3);
    int64_t Cm = K2.size(0);
    TORCH_CHECK(K2.size(1) == kh && K2.size(2) == kw && K2.size(3) == CO, "K2 shape");
    int64_t fh = (OH + 2 * ph - kh) / sh + 1;
    int64_t fw = (OW + 2 * pw - kw) / sw + 1;

    auto out = torch::empty({b, bt / b, fh, fw, Cm}, fd2.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    int st = conv_unfold_fwd(stream, out.data_ptr(), fd2.data_ptr(), K2.data_ptr(),
                             bt, (int)OH, (int)OW, (int)CO, (int)fh, (int)fw, (int)Cm,
                             (int)kh, (int)kw, (int)sh, (int)sw, (int)ph, (int)pw);
    TORCH_CHECK(st == 0, "cudnn conv forward failed: ", st);
    return out;
}

// fc1.weight [CO*kh*kw, M] fp32 -> NHWC filter [M, kh, kw, CO] fp16
static torch::Tensor w1_filter(torch::Tensor w, int64_t kh, int64_t kw) {
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.dtype() == torch::kFloat32, "bad w");
    int64_t Cin = w.size(0), M = w.size(1);
    int64_t CO = Cin / (kh * kw);
    auto out = torch::empty({M, kh, kw, CO}, w.options().dtype(torch::kFloat16));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    w1_filter_launcher(out.data_ptr(), w.data_ptr<float>(), (int)M, (int)kh, (int)kw,
                       (int)CO, stream);
    return out;
}

// fc2.weight [M, CO*kh*kw] fp32 -> NHWC filter [M, kh, kw, CO] fp16
static torch::Tensor w2_filter(torch::Tensor w, int64_t kh, int64_t kw) {
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.dtype() == torch::kFloat32, "bad w");
    int64_t M = w.size(0), Cin = w.size(1);
    int64_t CO = Cin / (kh * kw);
    auto out = torch::empty({M, kh, kw, CO}, w.options().dtype(torch::kFloat16));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    w2_filter_launcher(out.data_ptr(), w.data_ptr<float>(), (int)M, (int)kh, (int)kw,
                       (int)CO, stream);
    return out;
}

static void register_ffn_conv(pybind11::module& m) {
    m.def("ffn_fold_conv", &ffn_fold_conv, py::arg("y"), py::arg("K1"), py::arg("OH"),
          py::arg("OW"), py::arg("kh"), py::arg("kw"), py::arg("sh"), py::arg("sw"),
          py::arg("ph"), py::arg("pw"));
    m.def("ffn_gelu_div", &ffn_gelu_div, py::arg("raw"), py::arg("bias1"), py::arg("fh"),
          py::arg("fw"), py::arg("kh"), py::arg("kw"), py::arg("sh"), py::arg("sw"),
          py::arg("ph"), py::arg("pw"));
    m.def("ffn_unfold_conv", &ffn_unfold_conv, py::arg("fd2"), py::arg("K2"), py::arg("b"),
          py::arg("kh"), py::arg("kw"), py::arg("sh"), py::arg("sw"),
          py::arg("ph"), py::arg("pw"));
    m.def("w1_filter", &w1_filter, py::arg("w"), py::arg("kh"), py::arg("kw"));
    m.def("w2_filter", &w2_filter, py::arg("w"), py::arg("kh"), py::arg("kw"));
}

REGISTER_BINDING(ffn_conv, register_ffn_conv);
