// Implementação CUDA do K-Means
// Fornece: vector<Centroid> kmeans_cuda(vector<Point>& points, vector<Centroid>& initial_centroids, int max_iters)
// Observações:
// - Usa atomicAdd para acumular somas de centróides e counts na GPU.
// - Assume suporte a atomicAdd para double (compute capability >= 6.0). Se necessário, pode-se mudar para float.

#include "common.h"
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>

// Macro simples para checar erros CUDA
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  \
                      << " - " << cudaGetErrorString(err) << std::endl;   \
            return std::vector<Centroid>();                                 \
        }                                                                   \
    } while (0)

// Forward declaration for helper used inside kernel
__device__ double atomicAddDouble(double* address, double val);

// Kernel: cada thread processa um ponto, escolhe o centróide mais próximo e acumula (atomic) nos vetores de soma/count
__global__ void assign_and_accumulate(int n, int k,
                                      const double* px, const double* py,
                                      const double* cent_x, const double* cent_y,
                                      int* clusters,
                                      double* sum_x, double* sum_y, int* counts,
                                      int* changes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double best_dist = INFINITY;
    int best_cluster = 0;

    double x = px[i];
    double y = py[i];

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

    // acumula nas somas do centróide escolhido
    // atomicAdd para double nem sempre está disponível na arquitetura do device; usamos helper abaixo
    extern __shared__ int __notused; // placeholder se precisar
    atomicAddDouble(&sum_x[best_cluster], x);
    atomicAddDouble(&sum_y[best_cluster], y);
    atomicAdd(&counts[best_cluster], 1);
}

// Helper: atomic add para double (compatível com arquiteturas antigas)
__device__ double atomicAddDouble(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
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

// Host-side implementation
std::vector<Centroid> kmeans_cuda(std::vector<Point>& points, std::vector<Centroid>& initial_centroids, int max_iters)
{
    int n = (int)points.size();
    int k = (int)initial_centroids.size();
    if (n == 0 || k == 0) return std::vector<Centroid>();

    // Host arrays
    std::vector<double> h_px(n), h_py(n);
    std::vector<int>    h_clusters(n);
    for (int i = 0; i < n; ++i) {
        h_px[i] = points[i].x;
        h_py[i] = points[i].y;
        h_clusters[i] = points[i].cluster; // possivelmente -1
    }

    std::vector<double> h_cent_x(k), h_cent_y(k);
    for (int c = 0; c < k; ++c) {
        h_cent_x[c] = initial_centroids[c].x;
        h_cent_y[c] = initial_centroids[c].y;
    }

    // Device pointers
    double *d_px = nullptr, *d_py = nullptr;
    double *d_cent_x = nullptr, *d_cent_y = nullptr;
    int *d_clusters = nullptr;
    double *d_sum_x = nullptr, *d_sum_y = nullptr;
    int *d_counts = nullptr;
    int *d_changes = nullptr;

    // Aloca e copia pontos e centróides iniciais
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

    // Aloca buffers de redução na GPU
    CUDA_CHECK(cudaMalloc((void**)&d_sum_x, sizeof(double) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_sum_y, sizeof(double) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_counts, sizeof(int) * k));
    CUDA_CHECK(cudaMalloc((void**)&d_changes, sizeof(int)));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    std::vector<double> h_sum_x(k), h_sum_y(k);
    std::vector<int>    h_counts(k);

    for (int iter = 0; iter < max_iters; ++iter) {
        // Zera buffers na GPU
        CUDA_CHECK(cudaMemset(d_sum_x, 0, sizeof(double) * k));
        CUDA_CHECK(cudaMemset(d_sum_y, 0, sizeof(double) * k));
        CUDA_CHECK(cudaMemset(d_counts, 0, sizeof(int) * k));
        CUDA_CHECK(cudaMemset(d_changes, 0, sizeof(int)));

        // Lança kernel
        assign_and_accumulate<<<blocks, threads>>>(n, k, d_px, d_py, d_cent_x, d_cent_y, d_clusters, d_sum_x, d_sum_y, d_counts, d_changes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copia reduções para host
        CUDA_CHECK(cudaMemcpy(h_sum_x.data(), d_sum_x, sizeof(double) * k, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sum_y.data(), d_sum_y, sizeof(double) * k, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, sizeof(int) * k, cudaMemcpyDeviceToHost));

        int h_changes = 0;
        CUDA_CHECK(cudaMemcpy(&h_changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost));

        // Recalcula centróides no host
        bool any_empty = false;
        for (int c = 0; c < k; ++c) {
            if (h_counts[c] > 0) {
                h_cent_x[c] = h_sum_x[c] / double(h_counts[c]);
                h_cent_y[c] = h_sum_y[c] / double(h_counts[c]);
            } else {
                any_empty = true; // mantém centróide antigo se vazio
            }
        }

        // Copia novos centróides para GPU
        CUDA_CHECK(cudaMemcpy(d_cent_x, h_cent_x.data(), sizeof(double) * k, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cent_y, h_cent_y.data(), sizeof(double) * k, cudaMemcpyHostToDevice));

        if (h_changes == 0) {
            break; // convergiu
        }
    }

    // Copia centróides finais de volta para host
    std::vector<Centroid> results(k);
    CUDA_CHECK(cudaMemcpy(h_cent_x.data(), d_cent_x, sizeof(double) * k, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cent_y.data(), d_cent_y, sizeof(double) * k, cudaMemcpyDeviceToHost));

    for (int c = 0; c < k; ++c) {
        results[c].x = h_cent_x[c];
        results[c].y = h_cent_y[c];
    }

    // Atualiza clusters dos pontos no host (opcional)
    CUDA_CHECK(cudaMemcpy(h_clusters.data(), d_clusters, sizeof(int) * n, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        points[i].cluster = h_clusters[i];
    }

    // Libera memória
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_cent_x);
    cudaFree(d_cent_y);
    cudaFree(d_clusters);
    cudaFree(d_sum_x);
    cudaFree(d_sum_y);
    cudaFree(d_counts);
    cudaFree(d_changes);

    return results;
}
