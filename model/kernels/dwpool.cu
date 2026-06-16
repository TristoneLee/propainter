// kernels/dwpool.cu - depthwise conv with stride == kernel (the attention pool layer).
// Input NHWC fp16 [bt, H, W, C], weight/bias fp32 [C,1,kh,kw]/[C], output fp16.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void dwpool_kernel(
    const __half* __restrict__ x, const float* __restrict__ wgt,
    const float* __restrict__ bias, __half* __restrict__ out,
    int H, int W, int C, int kh, int kw, int OH, int OW, long long total)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = (int)(i % C);
    long long r = i / C;
    int ox = (int)(r % OW); r /= OW;
    int oy = (int)(r % OH);
    long long bt = r / OH;

    const __half* xp = x + ((bt * H + (long long)oy * kh) * W + (long long)ox * kw) * C + c;
    const float* wp = wgt + (long long)c * kh * kw;
    float acc = bias[c];
    for (int u = 0; u < kh; ++u)
        for (int v = 0; v < kw; ++v)
            acc += __half2float(xp[((long long)u * W + v) * C]) * wp[u * kw + v];
    out[i] = __float2half_rn(acc);
}

extern "C" void dwpool_launcher(
    void* out, const void* x, const float* wgt, const float* bias,
    long long bt, int H, int W, int C, int kh, int kw, int OH, int OW,
    cudaStream_t stream)
{
    long long total = bt * OH * OW * C;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    dwpool_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const __half*)x, wgt, bias, (__half*)out, H, W, C, kh, kw, OH, OW, total);
}
