#include "common.h"
#include <iostream>
#include <cmath>

// Função para calcular distância Euclidiana
double distance(const Point& a, const Centroid& b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// Função para inicializar centróides aleatoriamente
vector<Centroid> initialize_centroids(const vector<Point>& points, int k) {
    vector<Centroid> centroids;
    for (int i = 0; i < k; ++i) {
        int idx = rand() % points.size();
        centroids.push_back({points[idx].x, points[idx].y});
    }
    return centroids;
}

