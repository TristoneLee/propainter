// kernels/attn_binding.cpp - bindings + orchestration for sparse window attention.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>
#include "../binding_registry.h"

extern "C" void window_mask_launcher(
    float* flags, const float* mask, int l_t, int H, int W_in,
    int wh, int ww, int n_wh, int n_ww, cudaStream_t stream);
extern "C" void attn_unmasked_launcher(
    void* out, const void* q, const void* k, const void* v,
    const int* win_ids, int n_unmask, int t, int H, int W, int C,
    int n_head, int wh, int ww, int n_ww, float scale, cudaStream_t stream);
extern "C" void gather_q_launcher(
    void* qg, const void* q, const int* win_ids, int m, int n_head, int t,
    int H, int W, int C, int NT, int wh, int ww, int n_ww, float scale,
    cudaStream_t stream);
extern "C" void gather_kv_launcher(
    void* kg, void* vg, const void* k, const void* v,
    const void* pk, const void* pv, const int* win_ids, const int2* rtab,
    int m, int n_head, int T_sel, int t_off, int t_dil, int H, int W, int C,
    int NT, int n_roll, int n_pool, int wh, int ww, int n_ww,
    cudaStream_t stream);
extern "C" void scatter_out_launcher(
    void* out, const void* O, const int* win_ids, int m, int n_head, int t,
    int H, int W, int C, int NT, int wh, int ww, int n_ww, cudaStream_t stream);
extern "C" void softmax_rows_launcher(void* S, long long rows, int L, long long ld,
                                      cudaStream_t stream);
extern "C" int bgemm_scores_h16(cudaStream_t stream, void* S, const void* q, const void* kv,
                                long long batch, long long Lq, long long Lk, long long ch,
                                long long ldS);
extern "C" int bgemm_pv_h16(cudaStream_t stream, void* O, const void* P, const void* V,
                            long long batch, long long Lq, long long Lk, long long ch,
                            long long ldP);

namespace {

void check_h(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.dtype() == torch::kFloat16, "bad ", name);
}

// Rolled-token offset table, replicating the reference valid_ind_rolled order:
// stack(tl, tr, bl, br) masks, keep nonzero entries in flat order. Entry ->
// (dh, dw) such that source pixel = ((wi*wh + dh) mod H, (wj*ww + dw) mod W).
std::vector<int2> build_rtab(int wh, int ww, int eh, int ew) {
    std::vector<int2> tab;
    for (int r = 0; r < 4; ++r) {
        int sh = (r < 2) ? eh : -eh;
        int sw = (r % 2 == 0) ? ew : -ew;
        for (int pi = 0; pi < wh; ++pi) {
            for (int pj = 0; pj < ww; ++pj) {
                bool zero;
                if (r == 0)      zero = (pi < wh - eh) && (pj < ww - ew);
                else if (r == 1) zero = (pi < wh - eh) && (pj >= ew);
                else if (r == 2) zero = (pi >= eh) && (pj < ww - ew);
                else             zero = (pi >= eh) && (pj >= ew);
                if (!zero) tab.push_back(make_int2(pi + sh, pj + sw));
            }
        }
    }
    return tab;
}

}  // namespace

// mask: [b, l_t, h, w, 1] float; windows over the (h_pad, w_pad) grid.
// Returns (mask_ind, unmask_ind) as int32 CUDA tensors.
static std::tuple<torch::Tensor, torch::Tensor> window_mask_partition(
    torch::Tensor mask, int64_t wh, int64_t ww, int64_t n_wh, int64_t n_ww) {
    TORCH_CHECK(mask.is_cuda() && mask.is_contiguous() && mask.dtype() == torch::kFloat32,
                "bad mask");
    TORCH_CHECK(mask.dim() == 5 && mask.size(0) == 1, "mask must be [1,l_t,h,w,1]");
    int l_t = (int)mask.size(1), H = (int)mask.size(2), W_in = (int)mask.size(3);
    int n_win = (int)(n_wh * n_ww);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    auto flags = torch::empty({n_win}, mask.options());
    window_mask_launcher(flags.data_ptr<float>(), mask.data_ptr<float>(),
                         l_t, H, W_in, (int)wh, (int)ww, (int)n_wh, (int)n_ww, stream);

    std::vector<float> h_flags(n_win);
    cudaMemcpyAsync(h_flags.data(), flags.data_ptr<float>(), n_win * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<int> m_ind, u_ind;
    for (int i = 0; i < n_win; ++i) {
        if (h_flags[i] != 0.f) m_ind.push_back(i);
        else u_ind.push_back(i);
    }
    auto opts = mask.options().dtype(torch::kInt32);
    auto mt = torch::empty({(long)m_ind.size()}, opts);
    auto ut = torch::empty({(long)u_ind.size()}, opts);
    if (!m_ind.empty())
        cudaMemcpy(mt.data_ptr<int>(), m_ind.data(), m_ind.size() * sizeof(int),
                   cudaMemcpyHostToDevice);
    if (!u_ind.empty())
        cudaMemcpy(ut.data_ptr<int>(), u_ind.data(), u_ind.size() * sizeof(int),
                   cudaMemcpyHostToDevice);
    return std::make_tuple(mt, ut);
}

// q/k/v: [t, H, W, C] fp16. Allocates the output buffer [t, H, W, C] fp16 and
// computes the unmasked windows (in-window, per-frame attention).
static torch::Tensor attn_unmasked(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                   torch::Tensor unmask_ind, int64_t n_head,
                                   int64_t wh, int64_t ww) {
    check_h(q, "q"); check_h(k, "k"); check_h(v, "v");
    TORCH_CHECK(q.dim() == 4, "q must be [t,H,W,C]");
    TORCH_CHECK(unmask_ind.dtype() == torch::kInt32 && unmask_ind.is_cuda(), "bad unmask_ind");
    int t = (int)q.size(0), H = (int)q.size(1), W = (int)q.size(2), C = (int)q.size(3);
    int n_ww = W / (int)ww;
    int ch = C / (int)n_head;
    float scale = 1.f / sqrtf((float)ch);

    auto out = torch::empty({t, H, W, C}, q.options());
    int u = (int)unmask_ind.size(0);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    if (u > 0) {
        attn_unmasked_launcher(out.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
                               unmask_ind.data_ptr<int>(), u, t, H, W, C,
                               (int)n_head, (int)wh, (int)ww, n_ww, scale, stream);
    }
    return out;
}

// Masked windows: q over all t frames, kv over T_ind-selected frames with
// window + rolled + pooled tokens. Writes results into `out` (from attn_unmasked).
static void attn_masked(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                        torch::Tensor pk, torch::Tensor pv,
                        torch::Tensor mask_ind, torch::Tensor out,
                        int64_t n_head, int64_t t_off, int64_t t_dil,
                        int64_t wh, int64_t ww, int64_t eh, int64_t ew,
                        c10::optional<torch::Tensor> rtab_in = c10::nullopt) {
    check_h(q, "q"); check_h(k, "k"); check_h(v, "v");
    check_h(pk, "pk"); check_h(pv, "pv"); check_h(out, "out");
    TORCH_CHECK(mask_ind.dtype() == torch::kInt32 && mask_ind.is_cuda(), "bad mask_ind");
    int m_total = (int)mask_ind.size(0);
    if (m_total == 0) return;

    int t = (int)q.size(0), H = (int)q.size(1), W = (int)q.size(2), C = (int)q.size(3);
    int ph = (int)pk.size(1), pw = (int)pk.size(2);
    int NT = (int)(wh * ww);
    int n_ww = W / (int)ww;
    int nh = (int)n_head;
    int ch = C / nh;
    float scale = 1.f / sqrtf((float)ch);
    int T_sel = (t - (int)t_off + (int)t_dil - 1) / (int)t_dil;

    auto opts = q.options();
    // Roll-table: constant for a given (wh,ww,eh,ew). When the caller supplies a
    // precomputed device table (graph-safe path), use it directly so the hot
    // path has no host build + synchronous H2D copy (both break CUDA-graph
    // capture). Otherwise fall back to building it here (legacy eager path).
    torch::Tensor rtab_t;
    int n_roll;
    if (rtab_in.has_value()) {
        rtab_t = rtab_in.value();
        TORCH_CHECK(rtab_t.is_cuda() && rtab_t.is_contiguous() &&
                    rtab_t.dtype() == torch::kInt32 && rtab_t.dim() == 2 &&
                    rtab_t.size(1) == 2, "bad rtab");
        n_roll = (int)rtab_t.size(0);
    } else {
        auto rtab_host = build_rtab((int)wh, (int)ww, (int)eh, (int)ew);
        n_roll = (int)rtab_host.size();
        rtab_t = torch::empty({n_roll, 2}, opts.dtype(torch::kInt32));
        cudaMemcpy(rtab_t.data_ptr<int>(), rtab_host.data(), n_roll * sizeof(int2),
                   cudaMemcpyHostToDevice);
    }
    int n_pool = ph * pw;
    int L = NT + n_roll + n_pool;
    long long Lq = (long long)t * NT;
    long long Lk = (long long)T_sel * L;
    long long Lk_pad = (Lk + 7) & ~7LL;   // 8-aligned ld for tensor-core friendliness

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    const int CHUNK = 32;
    for (int off = 0; off < m_total; off += CHUNK) {
        int m = std::min(CHUNK, m_total - off);
        const int* ids = mask_ind.data_ptr<int>() + off;

        auto qg = torch::empty({m, nh, Lq, ch}, opts);
        auto kg = torch::empty({m, nh, Lk, ch}, opts);
        auto vg = torch::empty({m, nh, Lk, ch}, opts);
        auto S = torch::empty({(long long)m * nh, Lq, Lk_pad}, opts);
        auto O = torch::empty({m, nh, Lq, ch}, opts);

        gather_q_launcher(qg.data_ptr(), q.data_ptr(), ids, m, nh, t,
                          H, W, C, NT, (int)wh, (int)ww, n_ww, scale, stream);
        gather_kv_launcher(kg.data_ptr(), vg.data_ptr(), k.data_ptr(), v.data_ptr(),
                           pk.data_ptr(), pv.data_ptr(), ids,
                           (const int2*)rtab_t.data_ptr<int>(), m, nh, T_sel,
                           (int)t_off, (int)t_dil, H, W, C, NT, n_roll, n_pool,
                           (int)wh, (int)ww, n_ww, stream);

        int st = bgemm_scores_h16(stream, S.data_ptr(), qg.data_ptr(), kg.data_ptr(),
                                  (long long)m * nh, Lq, Lk, ch, Lk_pad);
        TORCH_CHECK(st == 0, "scores bgemm failed: ", st);
        softmax_rows_launcher(S.data_ptr(), (long long)m * nh * Lq, (int)Lk, Lk_pad, stream);
        st = bgemm_pv_h16(stream, O.data_ptr(), S.data_ptr(), vg.data_ptr(),
                          (long long)m * nh, Lq, Lk, ch, Lk_pad);
        TORCH_CHECK(st == 0, "pv bgemm failed: ", st);

        scatter_out_launcher(out.data_ptr(), O.data_ptr(), ids, m, nh, t,
                             H, W, C, NT, (int)wh, (int)ww, n_ww, stream);
    }
}

// Precompute the device roll-table once (it is constant for a given window
// config); pass the result as attn_masked's `rtab` to keep that call
// CUDA-graph capturable. Runs the host build + synchronous copy here, OUTSIDE
// any captured region.
static torch::Tensor build_rtab_device(int64_t wh, int64_t ww, int64_t eh, int64_t ew) {
    auto tab = build_rtab((int)wh, (int)ww, (int)eh, (int)ew);
    int n_roll = (int)tab.size();
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto t = torch::empty({n_roll, 2}, opts);
    if (n_roll > 0)
        cudaMemcpy(t.data_ptr<int>(), tab.data(), n_roll * sizeof(int2),
                   cudaMemcpyHostToDevice);
    return t;
}

static void register_attn(pybind11::module& m) {
    m.def("window_mask_partition", &window_mask_partition,
          py::arg("mask"), py::arg("wh"), py::arg("ww"), py::arg("n_wh"), py::arg("n_ww"));
    m.def("attn_unmasked", &attn_unmasked, py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("unmask_ind"), py::arg("n_head"), py::arg("wh"), py::arg("ww"));
    m.def("attn_masked", &attn_masked, py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("pk"), py::arg("pv"), py::arg("mask_ind"), py::arg("out"),
          py::arg("n_head"), py::arg("t_off"), py::arg("t_dil"),
          py::arg("wh"), py::arg("ww"), py::arg("eh"), py::arg("ew"),
          py::arg("rtab") = py::none());
    m.def("build_rtab", &build_rtab_device,
          py::arg("wh"), py::arg("ww"), py::arg("eh"), py::arg("ew"));
}

REGISTER_BINDING(attn, register_attn);
