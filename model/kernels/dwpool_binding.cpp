// kernels/dwpool_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void dwpool_launcher(
    void* out, const void* x, const float* wgt, const float* bias,
    long long bt, int H, int W, int C, int kh, int kw, int OH, int OW,
    cudaStream_t stream);

// x: [bt, H, W, C] NHWC fp16, weight: [C, 1, kh, kw] fp32 (depthwise, stride == kernel)
static torch::Tensor dwconv_pool(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == torch::kFloat16, "bad x");
    TORCH_CHECK(x.dim() == 4, "x must be [bt,H,W,C]");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(), "bad weight");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous(), "bad bias");
    int64_t bt = x.size(0), H = x.size(1), W = x.size(2), C = x.size(3);
    TORCH_CHECK(weight.dim() == 4 && weight.size(0) == C && weight.size(1) == 1, "weight must be [C,1,kh,kw]");
    int kh = (int)weight.size(2), kw = (int)weight.size(3);
    int OH = (int)((H - kh) / kh + 1), OW = (int)((W - kw) / kw + 1);

    auto out = torch::empty({bt, OH, OW, C}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    dwpool_launcher(out.data_ptr(), x.data_ptr(), weight.data_ptr<float>(),
                    bias.data_ptr<float>(), bt, (int)H, (int)W, (int)C, kh, kw, OH, OW, stream);
    return out;
}

static void register_dwpool(pybind11::module& m) {
    m.def("dwconv_pool", &dwconv_pool, py::arg("x"), py::arg("weight"), py::arg("bias"));
}

REGISTER_BINDING(dwpool, register_dwpool);
