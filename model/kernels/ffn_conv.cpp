// kernels/ffn_conv.cpp - cuDNN convolution launchers for the FusionFeedForward.
//
// Math identity used here:
//   fold(fc1(y))            == conv_backward_data(y, K1)   (K1 = permuted fc1.weight)
//   fc2(gelu(unfold(z)))    == conv_forward(gelu(z), K2)   (K2 = permuted fc2.weight)
// because unfold is a gather (elementwise GELU commutes with it) and fold/unfold
// with the fc GEMMs are exactly transposed/strided convolutions. This removes the
// [N, hidden] intermediates entirely.
//
// cudnn.h is found via the include dir injected by build_ext.py (which locates
// torch's bundled nvidia-cudnn package). For the in-repo utils/compile.sh path,
// CPATH is set to point at the same dir.
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace {

cudnnHandle_t g_dnn = nullptr;
void* g_ws = nullptr;
size_t g_ws_size = 0;
std::once_flag g_once;
std::mutex g_mu;

void init_once() { cudnnCreate(&g_dnn); }

void ensure_ws(size_t need) {
    if (need > g_ws_size) {
        if (g_ws) cudaFree(g_ws);
        cudaMalloc(&g_ws, need);
        g_ws_size = need;
    }
}

struct Key {
    long long a, b, c, d, e, f;
    bool operator<(const Key& o) const {
        return std::tie(a, b, c, d, e, f) < std::tie(o.a, o.b, o.c, o.d, o.e, o.f);
    }
};

struct FwdPlan {
    cudnnTensorDescriptor_t xd, yd;
    cudnnFilterDescriptor_t wd;
    cudnnConvolutionDescriptor_t cd;
    cudnnConvolutionFwdAlgo_t algo;
    size_t ws;
};
struct BwdPlan {
    cudnnTensorDescriptor_t dyd, dxd;
    cudnnFilterDescriptor_t wd;
    cudnnConvolutionDescriptor_t cd;
    cudnnConvolutionBwdDataAlgo_t algo;
    size_t ws;
};

std::map<Key, FwdPlan> g_fwd;
std::map<Key, BwdPlan> g_bwd;

cudnnConvolutionDescriptor_t make_conv(int ph, int pw, int sh, int sw) {
    cudnnConvolutionDescriptor_t cd;
    cudnnCreateConvolutionDescriptor(&cd);
    // fp32 accumulation: HALF compute makes cudnn pick slower algos here and
    // costs accuracy headroom (the fc1 reduction chain is ~2000 long).
    cudnnSetConvolution2dDescriptor(cd, ph, pw, sh, sw, 1, 1,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionMathType(cd, CUDNN_TENSOR_OP_MATH);
    return cd;
}

}  // namespace

// dx[bt, OH, OW, CO] (NHWC fp16) = conv_backward_data(dy[bt, fh, fw, Cm], K[Cm, kh, kw, CO])
// == fold of (dy @ K-flat): the fc1 + F.fold path.
extern "C" int conv_fold_bwd(cudaStream_t stream, void* dx, const void* dy, const void* K,
                             long long bt, int fh, int fw, int Cm, int OH, int OW, int CO,
                             int kh, int kw, int sh, int sw, int ph, int pw) {
    std::call_once(g_once, init_once);
    std::lock_guard<std::mutex> lk(g_mu);
    cudnnSetStream(g_dnn, stream);

    Key key{bt, ((long long)fh << 32) | fw, ((long long)OH << 32) | OW,
            ((long long)Cm << 32) | CO, ((long long)kh << 16) | kw,
            ((long long)sh << 16) | ((long long)sw << 8) | ph};
    auto it = g_bwd.find(key);
    if (it == g_bwd.end()) {
        BwdPlan p{};
        cudnnCreateTensorDescriptor(&p.dyd);
        cudnnSetTensor4dDescriptor(p.dyd, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF,
                                   (int)bt, Cm, fh, fw);
        cudnnCreateTensorDescriptor(&p.dxd);
        cudnnSetTensor4dDescriptor(p.dxd, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF,
                                   (int)bt, CO, OH, OW);
        cudnnCreateFilterDescriptor(&p.wd);
        cudnnSetFilter4dDescriptor(p.wd, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC,
                                   Cm, CO, kh, kw);
        p.cd = make_conv(ph, pw, sh, sw);

        cudnnConvolutionBwdDataAlgoPerf_t perf[8];
        int n = 0;
        cudnnGetConvolutionBackwardDataAlgorithm_v7(g_dnn, p.wd, p.dyd, p.cd, p.dxd,
                                                    8, &n, perf);
        p.algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        p.ws = 0;
        for (int i = 0; i < n; ++i) {
            if (perf[i].status == CUDNN_STATUS_SUCCESS &&
                perf[i].memory <= 512ull * 1024 * 1024) {
                p.algo = perf[i].algo;
                p.ws = perf[i].memory;
                break;
            }
        }
        it = g_bwd.emplace(key, p).first;
    }
    BwdPlan& p = it->second;
    ensure_ws(p.ws);
    float alpha = 1.f, beta = 0.f;
    cudnnStatus_t st = cudnnConvolutionBackwardData(
        g_dnn, &alpha, p.wd, K, p.dyd, dy, p.cd, p.algo, g_ws, p.ws,
        &beta, p.dxd, dx);
    return (int)st;
}

// out[bt, fh, fw, Cm] (NHWC fp16) = conv_forward(x[bt, OH, OW, CO], K[Cm, kh, kw, CO])
// == the F.unfold + fc2 path.
extern "C" int conv_unfold_fwd(cudaStream_t stream, void* out, const void* x, const void* K,
                               long long bt, int OH, int OW, int CO, int fh, int fw, int Cm,
                               int kh, int kw, int sh, int sw, int ph, int pw) {
    std::call_once(g_once, init_once);
    std::lock_guard<std::mutex> lk(g_mu);
    cudnnSetStream(g_dnn, stream);

    Key key{bt + (1ll << 60), ((long long)fh << 32) | fw, ((long long)OH << 32) | OW,
            ((long long)Cm << 32) | CO, ((long long)kh << 16) | kw,
            ((long long)sh << 16) | ((long long)sw << 8) | ph};
    auto it = g_fwd.find(key);
    if (it == g_fwd.end()) {
        FwdPlan p{};
        cudnnCreateTensorDescriptor(&p.xd);
        cudnnSetTensor4dDescriptor(p.xd, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF,
                                   (int)bt, CO, OH, OW);
        cudnnCreateTensorDescriptor(&p.yd);
        cudnnSetTensor4dDescriptor(p.yd, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF,
                                   (int)bt, Cm, fh, fw);
        cudnnCreateFilterDescriptor(&p.wd);
        cudnnSetFilter4dDescriptor(p.wd, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC,
                                   Cm, CO, kh, kw);
        p.cd = make_conv(ph, pw, sh, sw);

        cudnnConvolutionFwdAlgoPerf_t perf[8];
        int n = 0;
        cudnnGetConvolutionForwardAlgorithm_v7(g_dnn, p.xd, p.wd, p.cd, p.yd, 8, &n, perf);
        p.algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        p.ws = 0;
        for (int i = 0; i < n; ++i) {
            if (perf[i].status == CUDNN_STATUS_SUCCESS &&
                perf[i].memory <= 512ull * 1024 * 1024) {
                p.algo = perf[i].algo;
                p.ws = perf[i].memory;
                break;
            }
        }
        it = g_fwd.emplace(key, p).first;
    }
    FwdPlan& p = it->second;
    ensure_ws(p.ws);
    float alpha = 1.f, beta = 0.f;
    cudnnStatus_t st = cudnnConvolutionForward(
        g_dnn, &alpha, p.xd, x, p.wd, K, p.cd, p.algo, g_ws, p.ws,
        &beta, p.yd, out);
    return (int)st;
}
