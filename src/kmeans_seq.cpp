#include "common.h"
#include <iostream>
#include <cmath>

// K-Means sequencial
vector<Centroid> kmeans_seq(vector<Point>& points, vector<Centroid>& initial_centroids, int max_iters) {
    int n = points.size();
    int k = initial_centroids.size();
    vector<Centroid> centroids = initial_centroids;
    // vector<Centroid> centroids = initialize_centroids(points, k);

    for (int iter = 0; iter < max_iters; ++iter) {
        bool changed = false;

        // Etapa 1: atribuir cada ponto ao centróide mais próximo
        for (int i = 0; i < n; ++i) {
            double min_dist = INFINITY;
            int best_cluster = 0;

            for (int j = 0; j < k; ++j) {
                double d = distance(points[i], centroids[j]);
                if (d < min_dist) {
                    min_dist = d;
                    best_cluster = j;
                }
            }

            if (points[i].cluster != best_cluster) {
                changed = true;
                points[i].cluster = best_cluster;
            }
        }

        // Etapa 2: recalcular centróides
        vector<double> sum_x(k, 0.0), sum_y(k, 0.0);
        vector<int> count(k, 0);

        for (const auto& p : points) {
            int c = p.cluster;
            sum_x[c] += p.x;
            sum_y[c] += p.y;
            count[c]++;
        }

        for (int j = 0; j < k; ++j) {
            if (count[j] > 0) {
                centroids[j].x = sum_x[j] / count[j];
                centroids[j].y = sum_y[j] / count[j];
            }
        }

        // Se não houve mudança de cluster, convergiu
        if (!changed) {
            // cout << "Convergiu em " << iter << " iterações.\n";
            break;
        }
    }

    return centroids;
}
