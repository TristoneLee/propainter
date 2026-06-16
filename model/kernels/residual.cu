// kernels/residual.cu - residual-stream helpers (fp32 stream, fp16 branch outputs).
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// out = x (fp16) + y (fp16) [+ bias (fp32, per channel)], elementwise.
// One group = 4 channels = 2 __half2; bias is indexed per group (chv = C/4).
__global__ void add_residual_kernel(
    const __half2* __restrict__ x, const __half2* __restrict__ y,
    const float4* __restrict__ bias, __half2* __restrict__ out,
    int chv, long long total)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float2 a0 = __half22float2(x[idx * 2]);
    float2 a1 = __half22float2(x[idx * 2 + 1]);
    float2 b0 = __half22float2(y[idx * 2]);
    float2 b1 = __half22float2(y[idx * 2 + 1]);
    float4 v = make_float4(a0.x + b0.x, a0.y + b0.y, a1.x + b1.x, a1.y + b1.y);
    if (bias != nullptr) {
        float4 bb = bias[idx % chv];
        v.x += bb.x; v.y += bb.y; v.z += bb.z; v.w += bb.w;
    }
    out[idx * 2] = __floats2half2_rn(v.x, v.y);
    out[idx * 2 + 1] = __floats2half2_rn(v.z, v.w);
}

extern "C" void add_residual_launcher(
    void* out, const void* x, const void* y, const float* bias,
    long long n, long long C, cudaStream_t stream)
{
    long long total = n / 4;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    add_residual_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const __half2*)x, (const __half2*)y, (const float4*)bias, (__half2*)out,
        (int)(C / 4), total);
}

// ---- fp32 -> fp16 conversion (weight/bias preparation) ----
__global__ void to_half_kernel(const float* __restrict__ x, __half* __restrict__ out,
                               long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(x[i]);
}

extern "C" void to_half_launcher(void* out, const float* x, long long n,
                                 cudaStream_t stream)
{
    int threads = 256;
    long long blocks = (n + threads - 1) / threads;
    to_half_kernel<<<(unsigned)blocks, threads, 0, stream>>>(x, (__half*)out, n);
}
