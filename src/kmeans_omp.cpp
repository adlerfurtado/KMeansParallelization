#include "common.h"
#include <iostream>
#include <cmath>
#include <omp.h>
#include <limits>

// K-Means paralelizado com OpenMP
vector<Centroid> kmeans_omp(vector<Point>& points, vector<Centroid>& initial_centroids, int max_iters) {
    int n = points.size();
    int k = initial_centroids.size();
    vector<Centroid> centroids = initial_centroids;

    for (int iter = 0; iter < max_iters; ++iter) {
        int total_changed = 0;

        vector<double> global_sum_x(k, 0.0);
        vector<double> global_sum_y(k, 0.0);
        vector<int> global_count(k, 0);

        #pragma omp parallel 
        {
            // Vetores locais de cada thread
            vector<double> local_sum_x(k, 0.0);
            vector<double> local_sum_y(k, 0.0);
            vector<int> local_count(k, 0);
            int local_changed = 0;

            // Etapa 1: Atribuição dos pontos ao centróide mais próximo
            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                int old_cluster = points[i].cluster;

                // encontra o cluster mais próximo
                double best_dist = INFINITY; // distance(points[i], centroids[0]);
                int best_cluster = 0;
                
                for (int c = 0; c < k; ++c) {
                    double d = distance(points[i], centroids[c]);
                    if (d < best_dist) {
                        best_dist = d;
                        best_cluster = c;
                    }
                }

                if (old_cluster != best_cluster) {
                    local_changed++;
                    points[i].cluster = best_cluster;
                }

                // Somas parciais para evitar conflitos.
                local_sum_x[best_cluster] += points[i].x;
                local_sum_y[best_cluster] += points[i].y;
                local_count[best_cluster]++;
            }

            // Fusão das reduções locais → globais
            for (int c = 0; c < k; ++c) {
                #pragma omp atomic
                global_sum_x[c] += local_sum_x[c];
                #pragma omp atomic
                global_sum_y[c] += local_sum_y[c];
                #pragma omp atomic
                global_count[c] += local_count[c];
            }
            #pragma omp atomic
            total_changed += local_changed;
            
        }

        // Etapa 2: Recomputar centróides 
        #pragma omp for schedule(static)
        for (int c = 0; c < k; ++c) {
            if (global_count[c] > 0) {
                centroids[c].x = global_sum_x[c] / global_count[c];
                centroids[c].y = global_sum_y[c] / global_count[c];
            }
        }

        // Etapa 3: Verificação de convergência
        if (total_changed == 0) {
            // cout << "Convergiu em " << iter << " iterações.\n";
            break;
        }

    }

    return centroids;
}



// K-Means paralelizado com OpenMP - versão otimizada
// vector<Centroid> kmeans_omp(vector<Point>& points, vector<Centroid>& initial_centroids, int max_iters) {
//     int n = static_cast<int>(points.size());
//     int k = static_cast<int>(initial_centroids.size());
//     vector<Centroid> centroids = initial_centroids;

//     if (k <= 0 || n == 0) return centroids;

//     // Número de threads será determinado pelo runtime / omp_set_num_threads chamado externamente
//     for (int iter = 0; iter < max_iters; ++iter) {
//         int num_threads = omp_get_max_threads();

//         // Buffers locais organizados por thread: [thread_id][centroid_index]
//         // Usamos vetor de vetores para alocação dinâmica segura
//         vector<vector<double>> local_sum_x(num_threads, vector<double>(k, 0.0));
//         vector<vector<double>> local_sum_y(num_threads, vector<double>(k, 0.0));
//         vector<vector<int>>    local_count(num_threads, vector<int>(k, 0));
//         vector<int>            local_changed(num_threads, 0);

//         // Etapa 1: atribuição dos pontos ao centróide mais próximo (paralela)
//         #pragma omp parallel
//         {
//             int tid = omp_get_thread_num();
//             // Cada thread escreve apenas em local_sum_*/local_count[tid]
//             #pragma omp for schedule(static)
//             for (int i = 0; i < n; ++i) {
//                 // Calcula melhor centróide
//                 double best_dist = distance(points[i], centroids[0]);
//                 int best_cluster = 0;
//                 for (int c = 1; c < k; ++c) {
//                     double d = distance(points[i], centroids[c]);
//                     if (d < best_dist) {
//                         best_dist = d;
//                         best_cluster = c;
//                     }
//                 }

//                 int old_cluster = points[i].cluster;
//                 if (old_cluster != best_cluster) {
//                     local_changed[tid] += 1;
//                     points[i].cluster = best_cluster;
//                 }

//                 // Acumula nas estruturas locais (sem sincronização)
//                 local_sum_x[tid][best_cluster] += points[i].x;
//                 local_sum_y[tid][best_cluster] += points[i].y;
//                 local_count[tid][best_cluster] += 1;
//             }
//         } // fim região parallel

//         // Agregar os resultados locais em globais (sequencial)
//         vector<double> global_sum_x(k, 0.0);
//         vector<double> global_sum_y(k, 0.0);
//         vector<int>    global_count(k, 0);
//         int total_changed = 0;

//         for (int t = 0; t < num_threads; ++t) {
//             total_changed += local_changed[t];
//             for (int c = 0; c < k; ++c) {
//                 global_sum_x[c] += local_sum_x[t][c];
//                 global_sum_y[c] += local_sum_y[t][c];
//                 global_count[c] += local_count[t][c];
//             }
//         }

//         // Etapa 2: Recomputar centróides (sequencial, k tipicamente pequeno)
//         for (int c = 0; c < k; ++c) {
//             if (global_count[c] > 0) {
//                 centroids[c].x = global_sum_x[c] / global_count[c];
//                 centroids[c].y = global_sum_y[c] / global_count[c];
//             }
//             // else: se nenhum ponto atribuído, mantém centróide anterior (ou poderia re-inicializar)
//         }

//         // Etapa 3: Verificação de convergência
//         if (total_changed == 0) {
//             break;
//         }
//     }

//     return centroids;
// }
