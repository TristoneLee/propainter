// kernels/attn.cu - sparse window attention kernels (fp16 activations, fp32 math).
//
// Layouts: q/k/v are [t, H, W, C] fp16 (NHWC over the padded token grid), heads
// are contiguous slices of C (head h owns channels [h*ch, (h+1)*ch)).
//
// - window_mask_kernel: per-window mask flags (maxpool over window, sum over l_t)
// - attn_unmasked_kernel: fused in-window attention for unmasked windows
//   (per frame, kv = the window tokens only); register-tiled, fp32 compute
// - gather_q / gather_kv / scatter_out: build masked-branch batched-GEMM operands
//   directly from the token grid (roll = wrapped offset lookup, pool tokens appended)
// - softmax_rows: row softmax for the masked-branch fp16 score matrix
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------- mask flags
__global__ void window_mask_kernel(
    const float* __restrict__ mask, float* __restrict__ flags,
    int l_t, int H, int W_in, int wh, int ww, int n_wh, int n_ww)
{
    int wid = blockIdx.x * blockDim.x + threadIdx.x;
    if (wid >= n_wh * n_ww) return;
    int wi = wid / n_ww, wj = wid % n_ww;
    float s = 0.f;
    for (int f = 0; f < l_t; ++f) {
        float mx = 0.f;
        for (int pi = 0; pi < wh; ++pi) {
            int hh = wi * wh + pi;
            if (hh >= H) continue;
            for (int pj = 0; pj < ww; ++pj) {
                int wp = wj * ww + pj;
                if (wp >= W_in) continue;
                mx = fmaxf(mx, mask[((long long)f * H + hh) * W_in + wp]);
            }
        }
        s += mx;
    }
    flags[wid] = s;
}

extern "C" void window_mask_launcher(
    float* flags, const float* mask, int l_t, int H, int W_in,
    int wh, int ww, int n_wh, int n_ww, cudaStream_t stream)
{
    int n = n_wh * n_ww;
    window_mask_kernel<<<(n + 63) / 64, 64, 0, stream>>>(
        mask, flags, l_t, H, W_in, wh, ww, n_wh, n_ww);
}

// ------------------------------------------------------- unmasked attention
// grid: (n_unmask, t, n_head); one block computes a full [NT,NT] attention.
// Register-tiled: S stage 2x4 tiles, PV stage 2x2 tiles. Q/K/V staged in smem
// as fp16 (halves smem -> 2 blocks/SM), scores in fp32.
__global__ void attn_unmasked_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ v, __half* __restrict__ out,
    const int* __restrict__ win_ids,
    int H, int W, int C, int n_head, int wh, int ww, int n_ww, float scale)
{
    const int NT = wh * ww;
    const int ch = C / n_head;
    const int wid = win_ids[blockIdx.x];
    const int f = blockIdx.y;
    const int hd = blockIdx.z;
    const int wi = wid / n_ww, wj = wid % n_ww;
    const int ldk = (NT + 4) & ~3;        // padded rows for K^T (8B-aligned half4)

    extern __shared__ char smraw[];
    __half* Qs = (__half*)smraw;                     // [NT][ch]   (broadcast reads)
    __half* KTs = Qs + NT * ch;                      // [ch][ldk]  (transposed)
    __half* Vs = KTs + ch * ldk;                     // [NT][ch]
    float* Ss = (float*)(Vs + NT * ch);              // [NT][NT+1]

    const long long frame = (long long)f * H * W * C;

    for (int idx = threadIdx.x; idx < ch * ldk; idx += blockDim.x)
        KTs[idx] = __float2half_rn(0.f);
    __syncthreads();
    for (int idx = threadIdx.x; idx < NT * ch; idx += blockDim.x) {
        int tok = idx / ch, c = idx % ch;
        int pi = tok / ww, pj = tok % ww;
        long long g = frame + ((long long)(wi * wh + pi) * W + (wj * ww + pj)) * C + hd * ch + c;
        Qs[tok * ch + c] = __float2half_rn(__half2float(q[g]) * scale);
        KTs[c * ldk + tok] = k[g];
        Vs[tok * ch + c] = v[g];
    }
    __syncthreads();

    // S[i][j] = sum_c Q[i][c] * K[j][c]; one thread computes a 2x4 tile.
    {
        const int tj_n = (NT + 3) / 4, ti_n = (NT + 1) / 2;
        for (int tile = threadIdx.x; tile < ti_n * tj_n; tile += blockDim.x) {
            int i0 = (tile / tj_n) * 2, j0 = (tile % tj_n) * 4;
            bool has_i1 = (i0 + 1) < NT;
            float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f;
            float a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f;
            const __half* q0 = Qs + i0 * ch;
            const __half* q1 = Qs + (has_i1 ? i0 + 1 : i0) * ch;
#pragma unroll 4
            for (int c = 0; c < ch; ++c) {
                uint2 kraw = *(const uint2*)&KTs[c * ldk + j0];
                const __half2* kh = (const __half2*)&kraw;
                float2 k01 = __half22float2(kh[0]);
                float2 k23 = __half22float2(kh[1]);
                float x0 = __half2float(q0[c]), x1 = __half2float(q1[c]);
                a00 += x0 * k01.x; a01 += x0 * k01.y; a02 += x0 * k23.x; a03 += x0 * k23.y;
                a10 += x1 * k01.x; a11 += x1 * k01.y; a12 += x1 * k23.x; a13 += x1 * k23.y;
            }
            float* r0 = Ss + i0 * (NT + 1) + j0;
            if (j0 + 0 < NT) r0[0] = a00;
            if (j0 + 1 < NT) r0[1] = a01;
            if (j0 + 2 < NT) r0[2] = a02;
            if (j0 + 3 < NT) r0[3] = a03;
            if (has_i1) {
                float* r1 = r0 + (NT + 1);
                if (j0 + 0 < NT) r1[0] = a10;
                if (j0 + 1 < NT) r1[1] = a11;
                if (j0 + 2 < NT) r1[2] = a12;
                if (j0 + 3 < NT) r1[3] = a13;
            }
        }
    }
    __syncthreads();

    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int nwarp = blockDim.x >> 5;
    for (int i = warp; i < NT; i += nwarp) {
        float* row = Ss + i * (NT + 1);
        float mx = -1e30f;
        for (int j = lane; j < NT; j += 32) mx = fmaxf(mx, row[j]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        float s = 0.f;
        for (int j = lane; j < NT; j += 32) {
            float e = __expf(row[j] - mx);
            row[j] = e;
            s += e;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        float inv = 1.f / s;
        for (int j = lane; j < NT; j += 32) row[j] *= inv;
    }
    __syncthreads();

    // O[i][c] = sum_j P[i][j] * V[j][c]; one thread computes a 2x2 tile.
    {
        const int tc_n = ch / 2, ti_n = (NT + 1) / 2;
        for (int tile = threadIdx.x; tile < ti_n * tc_n; tile += blockDim.x) {
            int i0 = (tile / tc_n) * 2, c0 = (tile % tc_n) * 2;
            bool has_i1 = (i0 + 1) < NT;
            const float* p0 = Ss + i0 * (NT + 1);
            const float* p1 = p0 + (has_i1 ? (NT + 1) : 0);
            float a00 = 0.f, a01 = 0.f, a10 = 0.f, a11 = 0.f;
            for (int j = 0; j < NT; ++j) {
                float2 vv = __half22float2(*(const __half2*)&Vs[j * ch + c0]);
                float w0 = p0[j], w1 = p1[j];
                a00 += w0 * vv.x; a01 += w0 * vv.y;
                a10 += w1 * vv.x; a11 += w1 * vv.y;
            }
            int pi = i0 / ww, pj = i0 % ww;
            long long g0 = frame + ((long long)(wi * wh + pi) * W + (wj * ww + pj)) * C + hd * ch + c0;
            *(__half2*)&out[g0] = __floats2half2_rn(a00, a01);
            if (has_i1) {
                int qi = (i0 + 1) / ww, qj = (i0 + 1) % ww;
                long long g1 = frame + ((long long)(wi * wh + qi) * W + (wj * ww + qj)) * C + hd * ch + c0;
                *(__half2*)&out[g1] = __floats2half2_rn(a10, a11);
            }
        }
    }
}

extern "C" void attn_unmasked_launcher(
    void* out, const void* q, const void* k, const void* v,
    const int* win_ids, int n_unmask, int t, int H, int W, int C,
    int n_head, int wh, int ww, int n_ww, float scale, cudaStream_t stream)
{
    if (n_unmask <= 0) return;
    const int NT = wh * ww;
    const int ch = C / n_head;
    const int ldk = (NT + 4) & ~3;
    size_t smem = (size_t)(NT * ch * 2 + ch * ldk) * sizeof(__half)
                + (size_t)(NT * (NT + 1)) * sizeof(float);
    static size_t configured = 0;
    if (smem > 48 * 1024 && smem > configured) {
        cudaFuncSetAttribute(attn_unmasked_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = smem;
    }
    dim3 grid(n_unmask, t, n_head);
    attn_unmasked_kernel<<<grid, 512, smem, stream>>>(
        (const __half*)q, (const __half*)k, (const __half*)v, (__half*)out,
        win_ids, H, W, C, n_head, wh, ww, n_ww, scale);
}

// -------------------------------------------------- masked branch gather ops
// 8 fp16 values per thread (uint4). qg: [m, n_head, t*NT, ch], pre-scaled.
__global__ void gather_q_kernel(
    const uint4* __restrict__ q, uint4* __restrict__ qg,
    const int* __restrict__ win_ids,
    int n_head, int t, int H, int W, int chv, int NT,
    int wh, int ww, int n_ww, float scale, long long total)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c8 = (int)(idx % chv);
    long long r = idx / chv;
    int pos = (int)(r % NT); r /= NT;
    int f = (int)(r % t); r /= t;
    int hd = (int)(r % n_head);
    int w_i = (int)(r / n_head);

    int wid = win_ids[w_i];
    int wi = wid / n_ww, wj = wid % n_ww;
    int pi = pos / ww, pj = pos % ww;
    long long src = ((long long)f * H + (wi * wh + pi)) * W + (wj * ww + pj);

    union { uint4 u; __half2 h[4]; } val;
    val.u = q[src * (n_head * chv) + hd * chv + c8];
    __half2 s2 = __float2half2_rn(scale);
#pragma unroll
    for (int i = 0; i < 4; ++i) val.h[i] = __hmul2(val.h[i], s2);
    qg[idx] = val.u;
}

// kg/vg: [m, n_head, T_sel*L, ch] where L = NT + n_roll + n_pool.
// Per selected frame fs: [window tokens(NT) | rolled tokens(n_roll) | pool tokens(n_pool)]
__global__ void gather_kv_kernel(
    const uint4* __restrict__ k, const uint4* __restrict__ v,
    const uint4* __restrict__ pk, const uint4* __restrict__ pv,
    uint4* __restrict__ kg, uint4* __restrict__ vg,
    const int* __restrict__ win_ids, const int2* __restrict__ rtab,
    int n_head, int T_sel, int t_off, int t_dil, int H, int W, int chv,
    int NT, int n_roll, int n_pool,
    int wh, int ww, int n_ww, long long total)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int L = NT + n_roll + n_pool;
    int c8 = (int)(idx % chv);
    long long r = idx / chv;
    int pos = (int)(r % L); r /= L;
    int fs = (int)(r % T_sel); r /= T_sel;
    int hd = (int)(r % n_head);
    int w_i = (int)(r / n_head);

    int wid = win_ids[w_i];
    int wi = wid / n_ww, wj = wid % n_ww;
    int f = t_off + fs * t_dil;

    const uint4 *ksrc, *vsrc;
    long long src;
    int C8 = n_head * chv;
    if (pos < NT + n_roll) {
        int hh, wp;
        if (pos < NT) {
            hh = wi * wh + pos / ww;
            wp = wj * ww + pos % ww;
        } else {
            int2 d = rtab[pos - NT];
            hh = wi * wh + d.x;
            wp = wj * ww + d.y;
            hh = (hh % H + H) % H;     // torch.roll wrap-around
            wp = (wp % W + W) % W;
        }
        src = ((long long)f * H + hh) * W + wp;
        ksrc = k; vsrc = v;
    } else {
        int pp = pos - NT - n_roll;    // index into [ph*pw] pool grid
        src = (long long)f * n_pool + pp;
        ksrc = pk; vsrc = pv;
    }
    kg[idx] = ksrc[src * C8 + hd * chv + c8];
    vg[idx] = vsrc[src * C8 + hd * chv + c8];
}

// O: [m, n_head, t*NT, ch] fp16 -> out[t, H, W, C] fp16
__global__ void scatter_out_kernel(
    const uint4* __restrict__ O, uint4* __restrict__ out,
    const int* __restrict__ win_ids,
    int n_head, int t, int H, int W, int chv, int NT,
    int wh, int ww, int n_ww, long long total)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c8 = (int)(idx % chv);
    long long r = idx / chv;
    int pos = (int)(r % NT); r /= NT;
    int f = (int)(r % t); r /= t;
    int hd = (int)(r % n_head);
    int w_i = (int)(r / n_head);

    int wid = win_ids[w_i];
    int wi = wid / n_ww, wj = wid % n_ww;
    int pi = pos / ww, pj = pos % ww;
    long long dst = ((long long)f * H + (wi * wh + pi)) * W + (wj * ww + pj);
    out[dst * (n_head * chv) + hd * chv + c8] = O[idx];
}

// ------------------------------------------------------------- row softmax
__device__ __forceinline__ float blk_reduce(float v, float* red, bool is_max) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int nwarp = (blockDim.x + 31) >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, v, o);
        v = is_max ? fmaxf(v, other) : v + other;
    }
    if (lane == 0) red[warp] = v;
    __syncthreads();
    v = (threadIdx.x < nwarp) ? red[threadIdx.x] : (is_max ? -1e30f : 0.f);
    if (warp == 0) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            float other = __shfl_xor_sync(0xffffffffu, v, o);
            v = is_max ? fmaxf(v, other) : v + other;
        }
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    return red[0];
}

// Vectorized (half2) row softmax; row stride ld >= L. Requires L % 2 == 0.
template <int MAXV>
__global__ void softmax_rows_kernel(__half* __restrict__ S, int L, long long ld)
{
    long long row = blockIdx.x;
    __half2* p = (__half2*)(S + row * ld);
    const int Lv = L / 2;
    __shared__ float red[32];

    float2 loc[MAXV];
    int n = 0;
    float mx = -1e30f;
    for (int i = threadIdx.x; i < Lv; i += blockDim.x) {
        float2 f = __half22float2(p[i]);
        loc[n++] = f;
        mx = fmaxf(mx, fmaxf(f.x, f.y));
    }
    mx = blk_reduce(mx, red, true);
    __syncthreads();
    float s = 0.f;
    for (int j = 0; j < n; ++j) {
        float2 f = loc[j];
        f.x = __expf(f.x - mx);
        f.y = __expf(f.y - mx);
        s += f.x + f.y;
        loc[j] = f;
    }
    s = blk_reduce(s, red, false);
    float inv = 1.f / s;
    n = 0;
    for (int i = threadIdx.x; i < Lv; i += blockDim.x) {
        float2 f = loc[n++];
        p[i] = __floats2half2_rn(f.x * inv, f.y * inv);
    }
}

__global__ void softmax_rows_kernel_nocache(__half* __restrict__ S, int L, long long ld)
{
    long long row = blockIdx.x;
    __half* p = S + row * ld;
    __shared__ float red[32];

    float mx = -1e30f;
    for (int i = threadIdx.x; i < L; i += blockDim.x)
        mx = fmaxf(mx, __half2float(p[i]));
    mx = blk_reduce(mx, red, true);
    __syncthreads();
    float s = 0.f;
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        float e = __expf(__half2float(p[i]) - mx);
        p[i] = __float2half_rn(e);
        s += e;
    }
    s = blk_reduce(s, red, false);
    float inv = 1.f / s;
    for (int i = threadIdx.x; i < L; i += blockDim.x)
        p[i] = __float2half_rn(__half2float(p[i]) * inv);
}

// ---------------------------------------------------------------- launchers
extern "C" void gather_q_launcher(
    void* qg, const void* q, const int* win_ids, int m, int n_head, int t,
    int H, int W, int C, int NT, int wh, int ww, int n_ww, float scale,
    cudaStream_t stream)
{
    int chv = C / n_head / 8;
    long long total = (long long)m * n_head * t * NT * chv;
    long long blocks = (total + 255) / 256;
    gather_q_kernel<<<(unsigned)blocks, 256, 0, stream>>>(
        (const uint4*)q, (uint4*)qg, win_ids, n_head, t, H, W, chv, NT,
        wh, ww, n_ww, scale, total);
}

extern "C" void gather_kv_launcher(
    void* kg, void* vg, const void* k, const void* v,
    const void* pk, const void* pv, const int* win_ids, const int2* rtab,
    int m, int n_head, int T_sel, int t_off, int t_dil, int H, int W, int C,
    int NT, int n_roll, int n_pool, int wh, int ww, int n_ww,
    cudaStream_t stream)
{
    int chv = C / n_head / 8;
    long long total = (long long)m * n_head * T_sel * (NT + n_roll + n_pool) * chv;
    long long blocks = (total + 255) / 256;
    gather_kv_kernel<<<(unsigned)blocks, 256, 0, stream>>>(
        (const uint4*)k, (const uint4*)v, (const uint4*)pk, (const uint4*)pv,
        (uint4*)kg, (uint4*)vg, win_ids, rtab, n_head, T_sel, t_off, t_dil,
        H, W, chv, NT, n_roll, n_pool, wh, ww, n_ww, total);
}

extern "C" void scatter_out_launcher(
    void* out, const void* O, const int* win_ids, int m, int n_head, int t,
    int H, int W, int C, int NT, int wh, int ww, int n_ww, cudaStream_t stream)
{
    int chv = C / n_head / 8;
    long long total = (long long)m * n_head * t * NT * chv;
    long long blocks = (total + 255) / 256;
    scatter_out_kernel<<<(unsigned)blocks, 256, 0, stream>>>(
        (const uint4*)O, (uint4*)out, win_ids, n_head, t, H, W, chv, NT,
        wh, ww, n_ww, total);
}

extern "C" void softmax_rows_launcher(
    void* S, long long rows, int L, long long ld, cudaStream_t stream)
{
    const int threads = 256;
    if (L % 2 == 0 && L <= threads * 2 * 12) {
        softmax_rows_kernel<12><<<(unsigned)rows, threads, 0, stream>>>((__half*)S, L, ld);
    } else {
        softmax_rows_kernel_nocache<<<(unsigned)rows, threads, 0, stream>>>((__half*)S, L, ld);
    }
}
