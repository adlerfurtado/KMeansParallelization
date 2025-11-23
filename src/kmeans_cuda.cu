// kmeans_cuda.cu

#include "common.h"
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <cmath> // Adicionado para a macro INFINITY

// Macro simples para checar erros CUDA
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " - " << cudaGetErrorString(err) << std::endl;           \
            return std::vector<Centroid>();                                       \
        }                                                                         \
    } while (0)

// atomicAddDouble helper (Necessário para a GTX 1650/CC 7.5, mas implementado de forma genérica)
__device__ double atomicAddDouble(double* address, double val) {
// A GTX 1650 (Turing/CC 7.5) suporta atomicAdd(double*, double) nativo.
#if __CUDA_ARCH__ >= 700
    return atomicAdd(address, val);
#else
    // Fallback para atomicCAS para GPUs mais antigas (pré-Volta)
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        double sum = val + __longlong_as_double(assumed);
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(sum));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

// ===============================================
// KERNELS COM ATOMIC DOUBLE (Ponto a ponto, Preciso, mas Lento)
// ===============================================

__global__ void assign_and_accumulate_double(
    int n, int k,
    const double* __restrict__ px, const double* __restrict__ py,
    const double* __restrict__ cent_x, const double* __restrict__ cent_y,
    int* clusters,
    double* sum_x, double* sum_y, int* counts,
    int* changes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // CORRIGIDO: Usando INFINITY
    double best_dist = INFINITY;
    int best_cluster = 0;

    double x = px[i];
    double y = py[i];

    // Encontra o centróide mais próximo
    for (int c = 0; c < k; ++c) {
        double dx = x - cent_x[c];
        double dy = y - cent_y[c];
        double d2 = dx*dx + dy*dy;
        if (d2 < best_dist) {
            best_dist = d2;
            best_cluster = c;
        }
    }

    int old = clusters[i];
    if (old != best_cluster) {
        atomicAdd(changes, 1);
        clusters[i] = best_cluster;
    }

    // Acumula nas somas do centróide escolhido
    atomicAddDouble(&sum_x[best_cluster], x);
    atomicAddDouble(&sum_y[best_cluster], y);
    atomicAdd(&counts[best_cluster], 1);
}

// Kernel para recalcular centróides na GPU (um thread por centróide)
__global__ void recompute_centroids_double(int k, double* cent_x, double* cent_y,
                                         const double* sum_x, const double* sum_y,
                                         const int* counts)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;

    int cnt = counts[c];
    if (cnt > 0) {
        cent_x[c] = sum_x[c] / double(cnt);
        cent_y[c] = sum_y[c] / double(cnt);
    }
}

// ===============================================
// KERNELS COM ATOMIC FLOAT (Ponto a ponto, Rápido, mas Impreciso)
// ===============================================

__global__ void assign_and_accumulate_float(
    int n, int k,
    const double* __restrict__ px, const double* __restrict__ py,
    const double* __restrict__ cent_x, const double* __restrict__ cent_y,
    int* clusters,
    float* sum_x, float* sum_y, int* counts, 
    int* changes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // CORRIGIDO: Usando INFINITY
    double best_dist = INFINITY;
    int best_cluster = 0;

    double x = px[i];
    double y = py[i];

    // Encontra o centróide mais próximo
    for (int c = 0; c < k; ++c) {
        double dx = x - cent_x[c];
        double dy = y - cent_y[c];
        double d2 = dx*dx + dy*dy;
        if (d2 < best_dist) {
            best_dist = d2;
            best_cluster = c;
        }
    }

    int old = clusters[i];
    if (old != best_cluster) {
        atomicAdd(changes, 1);
        clusters[i] = best_cluster;
    }

    // Acumula nas somas do centróide escolhido usando atomicAdd(float*)
    atomicAdd(&sum_x[best_cluster], (float)x);
    atomicAdd(&sum_y[best_cluster], (float)y);
    atomicAdd(&counts[best_cluster], 1);
}

// Kernel para recalcular centróides a partir de somas float
__global__ void recompute_centroids_from_float(int k, double* cent_x, double* cent_y,
                                               const float* sum_xf, const float* sum_yf,
                                               const int* counts)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;
    int cnt = counts[c];
    if (cnt > 0) {
        cent_x[c] = double(sum_xf[c]) / double(cnt);
        cent_y[c] = double(sum_yf[c]) / double(cnt);
    }
}

// =========================================================
// KERNELS COM BLOCK REDUCTION (Otimizado, usa DOUBLE na Shared Memory)
// =========================================================

__global__ void assign_and_accumulate_block_double(int n, int k,
                                                 const double* __restrict__ px, const double* __restrict__ py,
                                                 const double* __restrict__ cent_x, const double* __restrict__ cent_y,
                                                 int* clusters,
                                                 double* global_sum_x, double* global_sum_y, int* global_counts, 
                                                 int* changes)
{
    extern __shared__ char smem[]; // dynamic shared memory
    // CORRIGIDO: Shared memory agora aloca double para manter a precisão
    double* s_sumx = (double*)smem; 
    double* s_sumy = s_sumx + k;
    int* s_counts = (int*)(s_sumy + k);

    // Inicializa arrays compartilhados
    for (int idx = threadIdx.x; idx < k; idx += blockDim.x) {
        s_sumx[idx] = 0.0;
        s_sumy[idx] = 0.0;
        s_counts[idx] = 0;
    }
    __syncthreads();

    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        double x = px[i];
        double y = py[i];
        
        // CORRIGIDO: Usando INFINITY
        double best_dist = INFINITY;
        int best_cluster = 0;
        
        for (int c = 0; c < k; ++c) {
            double dx = x - cent_x[c];
            double dy = y - cent_y[c];
            double d2 = dx*dx + dy*dy;
            if (d2 < best_dist) { best_dist = d2; best_cluster = c; }
        }

        int old = clusters[i];
        if (old != best_cluster) {
            atomicAdd(changes, 1);
            clusters[i] = best_cluster;
        }

        // Acumula em shared (double) bins usando atomicAddDouble
        atomicAddDouble(&s_sumx[best_cluster], x);
        atomicAddDouble(&s_sumy[best_cluster], y);
        atomicAdd(&s_counts[best_cluster], 1);
    }
    __syncthreads();

    // Flush shared accumulators para global (usando atomicAddDouble)
    for (int idx = threadIdx.x; idx < k; idx += blockDim.x) {
        if (s_counts[idx] != 0) {
            atomicAddDouble(&global_sum_x[idx], s_sumx[idx]);
            atomicAddDouble(&global_sum_y[idx], s_sumy[idx]);
            atomicAdd(&global_counts[idx], s_counts[idx]);
        }
    }
}


// ===============================================
// FUNÇÕES HOST
// ===============================================

// Host-side implementation: Versão double atomic (kmeans_cuda - não usado no run_both original)
std::vector<Centroid> kmeans_cuda(std::vector<Point>& points, std::vector<Centroid>& initial_centroids, int max_iters)
{
    int n = (int)points.size();
    int k = (int)initial_centroids.size();
    if (n == 0 || k == 0) return std::vector<Centroid>();

    std::vector<double> h_px(n), h_py(n);
    std::vector<int> h_clusters(n, -1);
    for (int i = 0; i < n; ++i) { h_px[i] = points[i].x; h_py[i] = points[i].y; h_clusters[i] = points[i].cluster; }
    std::vector<double> h_cent_x(k), h_cent_y(k);
    for (int c = 0; c < k; ++c) { h_cent_x[c] = initial_centroids[c].x; h_cent_y[c] = initial_centroids[c].y; }

    double *d_px = nullptr, *d_py = nullptr;
    double *d_cent_x = nullptr, *d_cent_y = nullptr;
    int *d_clusters = nullptr;
    double *d_sum_x = nullptr, *d_sum_y = nullptr; 
    int *d_counts = nullptr;
    int *d_changes = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_px, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc((void**)&d_py, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc((void**)&d_cent_x, sizeof(double) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_cent_y, sizeof(double) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_clusters, sizeof(int) * n));

    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cent_x, h_cent_x.data(), sizeof(double) * k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cent_y, h_cent_y.data(), sizeof(double) * k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clusters, h_clusters.data(), sizeof(int) * n, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&d_sum_x, sizeof(double) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_sum_y, sizeof(double) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_counts, sizeof(int) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_changes, sizeof(int)));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    int threads_cent = 256;
    int blocks_cent = (k + threads_cent - 1) / threads_cent;

    std::vector<double> h_cent_out_x(k), h_cent_out_y(k);
    std::vector<int> h_clusters_out(n);
    int h_changes = 0;
    float total_kernel_ms = 0.0f;
    int iter_count = 0;
    
    for (int iter = 0; iter < max_iters; ++iter) {
        CUDA_CHECK(cudaMemset(d_sum_x, 0, sizeof(double) * k));
        CUDA_CHECK(cudaMemset(d_sum_y, 0, sizeof(double) * k));
        CUDA_CHECK(cudaMemset(d_counts, 0, sizeof(int) * k));
        CUDA_CHECK(cudaMemset(d_changes, 0, sizeof(int)));
        
        cudaEvent_t kstart, kstop;
        CUDA_CHECK(cudaEventCreate(&kstart));
        CUDA_CHECK(cudaEventCreate(&kstop));
        CUDA_CHECK(cudaEventRecord(kstart));

        assign_and_accumulate_double<<<blocks, threads>>>(n, k,
                                                         d_px, d_py,
                                                         d_cent_x, d_cent_y,
                                                         d_clusters,
                                                         d_sum_x, d_sum_y, d_counts,
                                                         d_changes);
        CUDA_CHECK(cudaGetLastError());

        recompute_centroids_double<<<blocks_cent, threads_cent>>>(k, d_cent_x, d_cent_y, d_sum_x, d_sum_y, d_counts);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(kstop));
        CUDA_CHECK(cudaEventSynchronize(kstop));
        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kstart, kstop));
        total_kernel_ms += kernel_ms;
        iter_count++;
        CUDA_CHECK(cudaEventDestroy(kstart));
        CUDA_CHECK(cudaEventDestroy(kstop));

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_changes == 0) break;
    }

    CUDA_CHECK(cudaMemcpy(h_cent_out_x.data(), d_cent_x, sizeof(double) * k, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cent_out_y.data(), d_cent_y, sizeof(double) * k, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_clusters_out.data(), d_clusters, sizeof(int) * n, cudaMemcpyDeviceToHost));

    std::vector<Centroid> results(k);
    for (int c = 0; c < k; ++c) { results[c].x = h_cent_out_x[c]; results[c].y = h_cent_out_y[c]; }
    for (int i = 0; i < n; ++i) points[i].cluster = h_clusters_out[i];

    cudaFree(d_px); cudaFree(d_py); cudaFree(d_cent_x); cudaFree(d_cent_y); cudaFree(d_clusters);
    cudaFree(d_sum_x); cudaFree(d_sum_y); cudaFree(d_counts); cudaFree(d_changes);

    if (iter_count > 0) {
        std::cerr << "[CUDA timing - double] iterations=" << iter_count
                  << " total_kernel_ms=" << total_kernel_ms
                  << " avg_kernel_ms=" << (total_kernel_ms / iter_count) << std::endl;
    }
    return results;
}


// Host-side implementation: Versão float atomic (kmeans_cuda_float)
std::vector<Centroid> kmeans_cuda_float(std::vector<Point>& points, std::vector<Centroid>& initial_centroids, int max_iters)
{
    int n = (int)points.size();
    int k = (int)initial_centroids.size();
    if (n == 0 || k == 0) return std::vector<Centroid>();

    std::vector<double> h_px(n), h_py(n);
    std::vector<int> h_clusters(n, -1);
    for (int i = 0; i < n; ++i) { h_px[i] = points[i].x; h_py[i] = points[i].y; h_clusters[i] = points[i].cluster; }
    std::vector<double> h_cent_x(k), h_cent_y(k);
    for (int c = 0; c < k; ++c) { h_cent_x[c] = initial_centroids[c].x; h_cent_y[c] = initial_centroids[c].y; }

    double *d_px = nullptr, *d_py = nullptr;
    double *d_cent_x = nullptr, *d_cent_y = nullptr;
    int *d_clusters = nullptr;
    float *d_sum_xf = nullptr, *d_sum_yf = nullptr; 
    int *d_counts = nullptr;
    int *d_changes = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_px, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc((void**)&d_py, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc((void**)&d_cent_x, sizeof(double) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_cent_y, sizeof(double) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_clusters, sizeof(int) * n));

    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cent_x, h_cent_x.data(), sizeof(double) * k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cent_y, h_cent_y.data(), sizeof(double) * k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clusters, h_clusters.data(), sizeof(int) * n, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&d_sum_xf, sizeof(float) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_sum_yf, sizeof(float) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_counts, sizeof(int) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_changes, sizeof(int)));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    int threads_cent = 128;
    int blocks_cent = (k + threads_cent - 1) / threads_cent;

    std::vector<double> h_cent_out_x(k), h_cent_out_y(k);
    std::vector<int> h_clusters_out(n);

    for (int iter = 0; iter < max_iters; ++iter) {
        CUDA_CHECK(cudaMemset(d_sum_xf, 0, sizeof(float) * k));
        CUDA_CHECK(cudaMemset(d_sum_yf, 0, sizeof(float) * k));
        CUDA_CHECK(cudaMemset(d_counts, 0, sizeof(int) * k));
        CUDA_CHECK(cudaMemset(d_changes, 0, sizeof(int)));

        // Chamada CORRIGIDA: Usa assign_and_accumulate_float com ponteiros float
        assign_and_accumulate_float<<<blocks, threads>>>(n, k, d_px, d_py, d_cent_x, d_cent_y, d_clusters, d_sum_xf, d_sum_yf, d_counts, d_changes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        recompute_centroids_from_float<<<blocks_cent, threads_cent>>>(k, d_cent_x, d_cent_y, d_sum_xf, d_sum_yf, d_counts);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_changes = 0;
        CUDA_CHECK(cudaMemcpy(&h_changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_changes == 0) break;
    }

    CUDA_CHECK(cudaMemcpy(h_cent_out_x.data(), d_cent_x, sizeof(double) * k, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cent_out_y.data(), d_cent_y, sizeof(double) * k, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_clusters_out.data(), d_clusters, sizeof(int) * n, cudaMemcpyDeviceToHost));

    std::vector<Centroid> results(k);
    for (int c = 0; c < k; ++c) { results[c].x = h_cent_out_x[c]; results[c].y = h_cent_out_y[c]; }
    for (int i = 0; i < n; ++i) points[i].cluster = h_clusters_out[i];

    cudaFree(d_px); cudaFree(d_py); cudaFree(d_cent_x); cudaFree(d_cent_y); cudaFree(d_clusters);
    cudaFree(d_sum_xf); cudaFree(d_sum_yf); cudaFree(d_counts); cudaFree(d_changes);

    return results;
}

// Host-side implementation: Versão Block Reduction (kmeans_cuda_blockreduce)
std::vector<Centroid> kmeans_cuda_blockreduce(std::vector<Point>& points, std::vector<Centroid>& initial_centroids, int max_iters)
{
    int n = (int)points.size();
    int k = (int)initial_centroids.size();
    if (n == 0 || k == 0) return std::vector<Centroid>();

    std::vector<double> h_px(n), h_py(n);
    std::vector<int> h_clusters(n, -1);
    for (int i = 0; i < n; ++i) { h_px[i] = points[i].x; h_py[i] = points[i].y; h_clusters[i] = points[i].cluster; }
    std::vector<double> h_cent_x(k), h_cent_y(k);
    for (int c = 0; c < k; ++c) { h_cent_x[c] = initial_centroids[c].x; h_cent_y[c] = initial_centroids[c].y; }

    double *d_px = nullptr, *d_py = nullptr;
    double *d_cent_x = nullptr, *d_cent_y = nullptr;
    int *d_clusters = nullptr;
    double *d_sum_x = nullptr, *d_sum_y = nullptr; 
    int *d_counts = nullptr;
    int *d_changes = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_px, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc((void**)&d_py, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc((void**)&d_cent_x, sizeof(double) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_cent_y, sizeof(double) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_clusters, sizeof(int) * n));

    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cent_x, h_cent_x.data(), sizeof(double) * k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cent_y, h_cent_y.data(), sizeof(double) * k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clusters, h_clusters.data(), sizeof(int) * n, cudaMemcpyHostToDevice));

    // CORRIGIDO: Aloca buffers de redução DOUBLE
    CUDA_CHECK(cudaMalloc((void**)&d_sum_x, sizeof(double) * k)); 
    CUDA_CHECK(cudaMalloc((void**)&d_sum_y, sizeof(double) * k)); 
    CUDA_CHECK(cudaMalloc((void**)&d_counts, sizeof(int) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_changes, sizeof(int)));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    int threads_cent = 128;
    int blocks_cent = (k + threads_cent - 1) / threads_cent;

    // CORRIGIDO: Cálculo do tamanho da Shared Memory com DOUBLE
    size_t shared_bytes = (sizeof(double)*2 + sizeof(int)) * k;
    size_t max_shared = 48 * 1024; // 48KB típico
    bool use_shared = (shared_bytes <= max_shared);

    if (!use_shared) {
        std::cerr << "[KMeans CUDA BlockReduce] Shared Memory too small for K=" << k << " (" << shared_bytes << " bytes needed).\n";
        std::cerr << "Falling back to per-point atomic double (assign_and_accumulate_double).\n";
    }

    std::vector<double> h_cent_out_x(k), h_cent_out_y(k);
    std::vector<int> h_clusters_out(n);

    for (int iter = 0; iter < max_iters; ++iter) {
        // Zera buffers DOUBLE
        CUDA_CHECK(cudaMemset(d_sum_x, 0, sizeof(double) * k));
        CUDA_CHECK(cudaMemset(d_sum_y, 0, sizeof(double) * k));
        CUDA_CHECK(cudaMemset(d_counts, 0, sizeof(int) * k));
        CUDA_CHECK(cudaMemset(d_changes, 0, sizeof(int)));

        if (use_shared) {
            // Chamada CORRIGIDA: Usa assign_and_accumulate_block_double
            assign_and_accumulate_block_double<<<blocks, threads, shared_bytes>>>(n, k, d_px, d_py, d_cent_x, d_cent_y, d_clusters, d_sum_x, d_sum_y, d_counts, d_changes);
        } else {
            // Fallback para atomic double
            assign_and_accumulate_double<<<blocks, threads>>>(n, k, d_px, d_py, d_cent_x, d_cent_y, d_clusters, d_sum_x, d_sum_y, d_counts, d_changes);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Recalcula centróides (usa o kernel double)
        recompute_centroids_double<<<blocks_cent, threads_cent>>>(k, d_cent_x, d_cent_y, d_sum_x, d_sum_y, d_counts);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_changes = 0;
        CUDA_CHECK(cudaMemcpy(&h_changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_changes == 0) break;
    }

    CUDA_CHECK(cudaMemcpy(h_cent_out_x.data(), d_cent_x, sizeof(double) * k, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cent_out_y.data(), d_cent_y, sizeof(double) * k, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_clusters_out.data(), d_clusters, sizeof(int) * n, cudaMemcpyDeviceToHost));

    std::vector<Centroid> results(k);
    for (int c = 0; c < k; ++c) { results[c].x = h_cent_out_x[c]; results[c].y = h_cent_out_y[c]; }
    for (int i = 0; i < n; ++i) points[i].cluster = h_clusters_out[i];

    cudaFree(d_px); cudaFree(d_py); cudaFree(d_cent_x); cudaFree(d_cent_y); cudaFree(d_clusters);
    cudaFree(d_sum_x); cudaFree(d_sum_y); cudaFree(d_counts); cudaFree(d_changes);

    return results;
}