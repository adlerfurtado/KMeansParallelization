#include "common.h"
#include <iostream>
#include <cmath>

// K-Means paralelizado com OpenMP
vector<Centroid> kmeans_omp(vector<Point>& points, vector<Centroid>& initial_centroids, int max_iters) {
    int n = points.size();
    int k = initial_centroids.size();
    vector<Centroid> centroids = initial_centroids;
    // vector<Centroid> centroids = initialize_centroids(points, k);

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
                
                for (int c = 1; c < k; ++c) {
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
            #pragma omp critical  // critical: Especifica que o código só é executado em um thread por vez.
            {
                for (int c = 0; c < k; ++c) {
                    global_sum_x[c] += local_sum_x[c];
                    global_sum_y[c] += local_sum_y[c];
                    global_count[c] += local_count[c];
                }
                total_changed += local_changed;
            }
        }

        // Etapa 2: Recomputar centróides (sequencial, k geralmente pequeno)
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