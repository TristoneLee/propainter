// kernels/ffn_fold.cu - elementwise/permute helpers for the conv-formulated
// FusionFeedForward (the convolutions themselves live in ffn_conv.cpp).
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float gelu_exact(float v) {
    return 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
}

// fd2 = gelu((raw + B) / cnt), NHWC [bt, OH, OW, CO].
// B[co, y, x] = sum of fc1.bias over the patches covering pixel (y, x): the
// fold of the (per-patch constant) bias term, computed inline. cnt is the
// patch count (the F.fold(ones) normalizer), also computed analytically.
__global__ void gelu_div_bias_kernel(
    const __half* __restrict__ raw, const float* __restrict__ bias1,
    __half* __restrict__ out,
    int fh, int fw, int OH, int OW, int kh, int kw, int sh, int sw,
    int ph, int pw, int CO, long long total)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int co = (int)(idx % CO);
    long long r = idx / CO;
    int x = (int)(r % OW); r /= OW;
    int y = (int)(r % OH);

    int a = y + ph - kh + 1;
    int gh_lo = a > 0 ? (a + sh - 1) / sh : 0;
    int gh_hi = min(fh - 1, (y + ph) / sh);
    int b = x + pw - kw + 1;
    int gw_lo = b > 0 ? (b + sw - 1) / sw : 0;
    int gw_hi = min(fw - 1, (x + pw) / sw);

    float B = 0.f;
    int cnt = 0;
    const int kk_n = kh * kw;
    for (int gh = gh_lo; gh <= gh_hi; ++gh) {
        int ki = y + ph - gh * sh;
        for (int gw = gw_lo; gw <= gw_hi; ++gw) {
            int kj = x + pw - gw * sw;
            B += __ldg(&bias1[co * kk_n + ki * kw + kj]);
            ++cnt;
        }
    }
    float v = (__half2float(raw[idx]) + B) / (float)max(cnt, 1);
    out[idx] = __float2half_rn(gelu_exact(v));
}

extern "C" void gelu_div_bias_launcher(
    void* out, const void* raw, const float* bias1, long long bt,
    int fh, int fw, int OH, int OW, int kh, int kw, int sh, int sw,
    int ph, int pw, int CO, cudaStream_t stream)
{
    long long total = bt * OH * OW * CO;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    gelu_div_bias_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const __half*)raw, bias1, (__half*)out, fh, fw, OH, OW, kh, kw, sh, sw,
        ph, pw, CO, total);
}

// K1[m, ki, kj, co] = W1[co*kk + ki*kw + kj, m]  (fc1.weight [Cin=CO*kk, M] -> NHWC filter)
__global__ void w1_filter_kernel(const float* __restrict__ w, __half* __restrict__ out,
                                 int M, int kk_n, int kw, int CO, long long total)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int co = (int)(idx % CO);
    long long r = idx / CO;
    int kkv = (int)(r % kk_n);
    int m = (int)(r / kk_n);
    out[idx] = __float2half_rn(w[(long long)(co * kk_n + kkv) * M + m]);
}

// K2[m, ki, kj, co] = W2[m, co*kk + ki*kw + kj]  (fc2.weight [M, CO*kk] -> NHWC filter)
__global__ void w2_filter_kernel(const float* __restrict__ w, __half* __restrict__ out,
                                 int Cin, int kk_n, int kw, int CO, long long total)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int co = (int)(idx % CO);
    long long r = idx / CO;
    int kkv = (int)(r % kk_n);
    int m = (int)(r / kk_n);
    out[idx] = __float2half_rn(w[(long long)m * Cin + co * kk_n + kkv]);
}

extern "C" void w1_filter_launcher(void* out, const float* w, int M, int kh, int kw,
                                   int CO, cudaStream_t stream)
{
    long long total = (long long)M * kh * kw * CO;
    w1_filter_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
        w, (__half*)out, M, kh * kw, kw, CO, total);
}

extern "C" void w2_filter_launcher(void* out, const float* w, int M, int kh, int kw,
                                   int CO, cudaStream_t stream)
{
    long long total = (long long)M * kh * kw * CO;
    w2_filter_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
        w, (__half*)out, CO * kh * kw, kh * kw, kw, CO, total);
}
