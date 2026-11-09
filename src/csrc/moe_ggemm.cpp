// Sync-free grouped GEMM for MoE experts (innovation 1, MoE side): hipBLASLt ext GroupedGemm with
// DEVICE user arguments, so per-expert token counts and row offsets are written on the GPU and the
// host never reads them (no tokens_per_expert.tolist()).
// Row-major Y_e[m_e, N] = X_e[m_e, K] @ W_e[K, N] (W stored [E, K, N] row-major, bf16, fp32 accum).
// hipBLASLt is column-major: Y_e^T[N, m_e] = W_e^T[N, K] * X_e^T[K, m_e]  ->  m'=N, n'=m_e, k=K,
// A = W_e (col-major N x K, lda = N), B = X_e (col-major K x m_e, ldb = K), D = C = Y_e (ldd = N).
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct GG {
    hipblasLtHandle_t handle;
    hipblaslt_ext::GroupedGemm* gemm;
    void* ws;
    void* dua;          // device UserArguments[E]
    int E;
    float alpha, beta;
};

extern "C" {
int gg_userargs_size() { return (int)sizeof(hipblaslt_ext::UserArguments); }

// rep_m: representative tokens per expert used for the heuristic (kernel must handle any m <= cap)
// per-expert host bounds (grid size per group); rows of expert e must be <= bound[e]
void* gg_create_v(int E, long N, long K, const long* bound, void* W, void* X, void* Y, long ws_bytes);
void* gg_create(int E, long N, long K, long rep_m, void* W, void* X, void* Y, long ws_bytes) {
    std::vector<long> b(E, rep_m);
    return gg_create_v(E, N, K, b.data(), W, X, Y, ws_bytes);
}
void* gg_create_v(int E, long N, long K, const long* bound, void* W, void* X, void* Y, long ws_bytes) {
    long rep_m = 0; for (int e = 0; e < E; e++) rep_m = bound[e] > rep_m ? bound[e] : rep_m;
    GG* g = new GG();
    g->E = E; g->alpha = 1.f; g->beta = 0.f;
    if (hipblasLtCreate(&g->handle) != HIPBLAS_STATUS_SUCCESS) { fprintf(stderr, "gg: create handle\n"); return nullptr; }
    g->gemm = new hipblaslt_ext::GroupedGemm(g->handle, HIPBLAS_OP_N, HIPBLAS_OP_N, HIP_R_16BF, HIP_R_16BF,
                                             HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F);
    std::vector<int64_t> m(E, N), n(E), k(E, K), b(E, 1);
    for (int e = 0; e < E; e++) n[e] = bound[e] > 0 ? bound[e] : 1;
    std::vector<hipblaslt_ext::GemmEpilogue> epi(E);
    std::vector<hipblaslt_ext::GemmInputs> in(E);
    char* w = (char*)W; char* x = (char*)X; char* y = (char*)Y;
    for (int e = 0; e < E; e++) {
        in[e].a = w + (size_t)e * K * N * 2;
        in[e].b = x;            // dummy (real pointers are written into the device user args per call)
        in[e].c = y;
        in[e].d = in[e].c;
        in[e].alpha = &g->alpha; in[e].beta = &g->beta;
    }
    if (g->gemm->setProblem(m, n, k, b, epi, in) != HIPBLAS_STATUS_SUCCESS) { fprintf(stderr, "gg: setProblem\n"); return nullptr; }
    hipblaslt_ext::GemmPreference pref; pref.setMaxWorkspaceBytes(ws_bytes);
    std::vector<hipblasLtMatmulHeuristicResult_t> res;
    if (g->gemm->algoGetHeuristic(8, pref, res) != HIPBLAS_STATUS_SUCCESS || res.empty()) { fprintf(stderr, "gg: no algo\n"); return nullptr; }
    size_t wsz = 0; int pick = -1;
    for (size_t i = 0; i < res.size(); i++) { size_t s = 0; if (g->gemm->isAlgoSupported(res[i].algo, s) == HIPBLAS_STATUS_SUCCESS && s <= (size_t)ws_bytes) { pick = (int)i; wsz = s; break; } }
    if (pick < 0) { fprintf(stderr, "gg: no supported algo\n"); return nullptr; }
    hipMalloc(&g->ws, ws_bytes > 0 ? ws_bytes : 1);
    if (g->gemm->initialize(res[pick].algo, g->ws, true) != HIPBLAS_STATUS_SUCCESS) { fprintf(stderr, "gg: initialize\n"); return nullptr; }
    size_t uas = sizeof(hipblaslt_ext::UserArguments) * E;
    std::vector<char> hua(uas);
    g->gemm->getDefaultValueForDeviceUserArguments(hua.data());
    hipMalloc(&g->dua, uas);
    hipMemcpy(g->dua, hua.data(), uas, hipMemcpyHostToDevice);
    if (getenv("GG_VERBOSE")) fprintf(stderr, "gg: E=%d N=%ld K=%ld max_bound=%ld algo#%d ws=%zu ua=%zu B\n", E, N, K, rep_m, pick, wsz, sizeof(hipblaslt_ext::UserArguments));
    return g;
}
// generic: column-major problem per group g: D(m x n_g) = op(A)(m x k_g) * op(B)(k_g x n_g), packed lds.
// opA/opB: 0 = N, 1 = T. n_bound/k_bound: host sizes (grid uses m, n_bound; algo heuristic uses k_bound).
void* gg_create_g(int G, int opA, int opB, long m, const long* n_bound, const long* k_bound,
                  void* A, void* B, void* D, long ws_bytes) {
    GG* g = new GG();
    g->E = G; g->alpha = 1.f; g->beta = 0.f;
    if (hipblasLtCreate(&g->handle) != HIPBLAS_STATUS_SUCCESS) return nullptr;
    g->gemm = new hipblaslt_ext::GroupedGemm(g->handle, opA ? HIPBLAS_OP_T : HIPBLAS_OP_N, opB ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                                             HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F);
    std::vector<int64_t> mm(G, m), n(G), k(G), b(G, 1);
    for (int e = 0; e < G; e++) { n[e] = n_bound[e] > 0 ? n_bound[e] : 1; k[e] = k_bound[e] > 0 ? k_bound[e] : 1; }
    std::vector<hipblaslt_ext::GemmEpilogue> epi(G);
    std::vector<hipblaslt_ext::GemmInputs> in(G);
    for (int e = 0; e < G; e++) { in[e].a = A; in[e].b = B; in[e].c = D; in[e].d = D; in[e].alpha = &g->alpha; in[e].beta = &g->beta; }
    if (g->gemm->setProblem(mm, n, k, b, epi, in) != HIPBLAS_STATUS_SUCCESS) { fprintf(stderr, "gg_g: setProblem\n"); return nullptr; }
    hipblaslt_ext::GemmPreference pref; pref.setMaxWorkspaceBytes(ws_bytes);
    std::vector<hipblasLtMatmulHeuristicResult_t> res;
    if (g->gemm->algoGetHeuristic(8, pref, res) != HIPBLAS_STATUS_SUCCESS || res.empty()) { fprintf(stderr, "gg_g: no algo\n"); return nullptr; }
    int pick = -1; size_t wsz = 0;
    for (size_t i = 0; i < res.size(); i++) { size_t s = 0; if (g->gemm->isAlgoSupported(res[i].algo, s) == HIPBLAS_STATUS_SUCCESS && s <= (size_t)ws_bytes) { pick = (int)i; wsz = s; break; } }
    if (pick < 0) { fprintf(stderr, "gg_g: no supported algo\n"); return nullptr; }
    hipMalloc(&g->ws, ws_bytes > 0 ? ws_bytes : 1);
    if (g->gemm->initialize(res[pick].algo, g->ws, true) != HIPBLAS_STATUS_SUCCESS) { fprintf(stderr, "gg_g: initialize\n"); return nullptr; }
    size_t uas = sizeof(hipblaslt_ext::UserArguments) * G;
    std::vector<char> hua(uas);
    g->gemm->getDefaultValueForDeviceUserArguments(hua.data());
    hipMalloc(&g->dua, uas);
    hipMemcpy(g->dua, hua.data(), uas, hipMemcpyHostToDevice);
    if (getenv("GG_VERBOSE")) fprintf(stderr, "gg_g: G=%d op=%d%d m=%ld algo#%d ws=%zu\n", G, opA, opB, m, pick, wsz);
    return g;
}
// dump the default user args of group 0 (for inspecting strides)
void gg_dump(void* h) {
    GG* g = (GG*)h; hipblaslt_ext::UserArguments u; hipMemcpy(&u, g->dua, sizeof(u), hipMemcpyDeviceToHost);
    fprintf(stderr, "ua m=%u n=%u batch=%u k=%u sD=%u/%u sC=%u/%u sA=%u/%u sB=%u/%u\n", u.m, u.n, u.batch, u.k,
            u.strideD1, u.strideD2, u.strideC1, u.strideC2, u.strideA1, u.strideA2, u.strideB1, u.strideB2);
}
void* gg_userargs(void* h) { return ((GG*)h)->dua; }
int gg_run(void* h, void* stream) { GG* g = (GG*)h; return (int)g->gemm->run(g->dua, (hipStream_t)stream); }
void gg_destroy(void* h) { GG* g = (GG*)h; hipFree(g->ws); hipFree(g->dua); delete g->gemm; hipblasLtDestroy(g->handle); delete g; }
}
