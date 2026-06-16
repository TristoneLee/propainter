// kernels/layernorm.cu - fused LayerNorm (fp32 in -> fp16 out) with optional
// right-padding of the W dim (pad tokens written as zeros), matching
// F.pad(norm(x)) in the reference model.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float ln_block_reduce_sum(float v, float* red) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int nwarp = (blockDim.x + 31) >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = (threadIdx.x < nwarp) ? red[threadIdx.x] : 0.f;
    if (warp == 0) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    return red[0];
}

// fp16 residual stream: x is read as 2 __half2 per 4-channel group (matches the
// fp16 norm-output layout). LayerNorm statistics are still accumulated in fp32.
__global__ void layernorm_pad_kernel(
    const __half2* __restrict__ x, const float4* __restrict__ gamma,
    const float4* __restrict__ beta, __half2* __restrict__ out,
    int H_in, int H_out, int W_in, int W_out, int nvec, float eps)
{
    long long tok = blockIdx.x;
    int w = (int)(tok % W_out);
    long long row = tok / W_out;            // flattened (bt, h_out)
    int h = (int)(row % H_out);
    long long bt = row / H_out;
    __half2* op = out + tok * (long long)nvec * 2;   // 2 half2 per 4-ch group

    if (w >= W_in || h >= H_in) {           // padded token -> zeros
        for (int i = threadIdx.x; i < nvec * 2; i += blockDim.x)
            op[i] = __half2half2(__float2half_rn(0.f));
        return;
    }
    const __half2* xp = x + ((bt * H_in + h) * W_in + w) * (long long)nvec * 2;

    float s = 0.f, ss = 0.f;
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
        float2 v0 = __half22float2(xp[i * 2]);
        float2 v1 = __half22float2(xp[i * 2 + 1]);
        s += v0.x + v0.y + v1.x + v1.y;
        ss += v0.x * v0.x + v0.y * v0.y + v1.x * v1.x + v1.y * v1.y;
    }
    __shared__ float red[32];
    float total = ln_block_reduce_sum(s, red);
    __syncthreads();
    float total2 = ln_block_reduce_sum(ss, red);

    float C = (float)(nvec * 4);
    float mean = total / C;
    float var = total2 / C - mean * mean;
    float inv = rsqrtf(var + eps);

    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
        float2 v0 = __half22float2(xp[i * 2]);
        float2 v1 = __half22float2(xp[i * 2 + 1]);
        float4 g = gamma[i], b = beta[i];
        float ox = (v0.x - mean) * inv * g.x + b.x;
        float oy = (v0.y - mean) * inv * g.y + b.y;
        float oz = (v1.x - mean) * inv * g.z + b.z;
        float ow = (v1.y - mean) * inv * g.w + b.w;
        op[i * 2] = __floats2half2_rn(ox, oy);
        op[i * 2 + 1] = __floats2half2_rn(oz, ow);
    }
}

extern "C" void layernorm_pad_launcher(
    void* out, const void* x, const float* gamma, const float* beta,
    long long bt, long long H_in, long long H_out, long long W_in, long long W_out,
    long long C, float eps, cudaStream_t stream)
{
    int nvec = (int)(C / 4);
    long long blocks = bt * H_out * W_out;
    layernorm_pad_kernel<<<(unsigned)blocks, 128, 0, stream>>>(
        (const __half2*)x, (const float4*)gamma, (const float4*)beta, (__half2*)out,
        (int)H_in, (int)H_out, (int)W_in, (int)W_out, nvec, eps);
}

// ---- fused: x_new = x + branch (fp16); norm_out = LayerNorm(x_new) (fp16) ----
// branch is read at width Wb (>= W, e.g. the padded proj output); norm output
// is written at width W_out (>= W, pad tokens zero-filled). Saves one full
// fp32 read+write of the residual stream per fusion site.
__global__ void residual_norm_kernel(
    const __half2* __restrict__ x, const __half2* __restrict__ branch,
    const float4* __restrict__ bias2, const float4* __restrict__ gamma,
    const float4* __restrict__ beta,
    __half2* __restrict__ x_new, __half2* __restrict__ norm_out,
    int H, int H_out, int Hb, int W, int Wb, int W_out, int nvec, float eps)
{
    long long tok = blockIdx.x;                  // over bt * H_out * W_out
    int w = (int)(tok % W_out);
    long long row = tok / W_out;                 // (bt, h_out)
    int h = (int)(row % H_out);
    long long bt = row / H_out;
    __half2* op = norm_out + tok * (long long)nvec * 2;

    if (w >= W || h >= H) {                       // pad token: zero norm output only
        for (int i = threadIdx.x; i < nvec * 2; i += blockDim.x)
            op[i] = __half2half2(__float2half_rn(0.f));
        return;
    }
    long long xtok = (bt * H + h) * W + w;        // x_new: unpadded grid
    const __half2* xp = x + xtok * (long long)nvec * 2;
    const __half2* bp = branch + ((bt * Hb + h) * Wb + w) * (long long)nvec * 2;
    __half2* np = x_new + xtok * (long long)nvec * 2;

    float4 buf[4];
    int nb = 0;
    float s = 0.f, ss = 0.f;
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
        float2 a0 = __half22float2(xp[i * 2]);
        float2 a1 = __half22float2(xp[i * 2 + 1]);
        float2 b0 = __half22float2(bp[i * 2]);
        float2 b1 = __half22float2(bp[i * 2 + 1]);
        float4 v = make_float4(a0.x + b0.x, a0.y + b0.y, a1.x + b1.x, a1.y + b1.y);
        if (bias2 != nullptr) {
            float4 bb = bias2[i];
            v.x += bb.x; v.y += bb.y; v.z += bb.z; v.w += bb.w;
        }
        np[i * 2] = __floats2half2_rn(v.x, v.y);
        np[i * 2 + 1] = __floats2half2_rn(v.z, v.w);
        buf[nb++] = v;
        s += v.x + v.y + v.z + v.w;
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    __shared__ float red[32];
    float total = ln_block_reduce_sum(s, red);
    __syncthreads();
    float total2 = ln_block_reduce_sum(ss, red);

    float C = (float)(nvec * 4);
    float mean = total / C;
    float var = total2 / C - mean * mean;
    float inv = rsqrtf(var + eps);

    nb = 0;
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
        float4 v = buf[nb++], g = gamma[i], b = beta[i];
        float ox = (v.x - mean) * inv * g.x + b.x;
        float oy = (v.y - mean) * inv * g.y + b.y;
        float oz = (v.z - mean) * inv * g.z + b.z;
        float ow = (v.w - mean) * inv * g.w + b.w;
        op[i * 2] = __floats2half2_rn(ox, oy);
        op[i * 2 + 1] = __floats2half2_rn(oz, ow);
    }
}

extern "C" void residual_norm_launcher(
    void* x_new, void* norm_out, const void* x, const void* branch,
    const float* bias2, const float* gamma, const float* beta,
    long long bt, long long H, long long H_out, long long Hb, long long W,
    long long Wb, long long W_out, long long C, float eps, cudaStream_t stream)
{
    int nvec = (int)(C / 4);
    long long blocks = bt * H_out * W_out;
    residual_norm_kernel<<<(unsigned)blocks, 128, 0, stream>>>(
        (const __half2*)x, (const __half2*)branch, (const float4*)bias2,
        (const float4*)gamma, (const float4*)beta, (__half2*)x_new, (__half2*)norm_out,
        (int)H, (int)H_out, (int)Hb, (int)W, (int)Wb, (int)W_out, nvec, eps);
}
