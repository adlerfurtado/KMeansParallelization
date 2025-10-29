#ifndef __COMMON_H__
#define __COMMON_H__

#include <vector> 
#include <cmath>

using namespace std;

// Estrutura para representar um ponto (x, y)
struct Point {
    double x, y;
    int cluster;  // índice do centróide mais próximo
};

// Estrutura para representar um centróide
struct Centroid {
    double x, y;

    // Overload the less-than operator for sorting
    bool operator<(const Centroid& other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }

    // Equality with tolerance
    bool operator==(const Centroid& other) const {
        const double EPS = 1e-6;
        return std::fabs(x - other.x) < EPS &&
               std::fabs(y - other.y) < EPS;
    }
};

double distance(const Point& a, const Centroid& b);
vector<Centroid> initialize_centroids(const vector<Point>& points, int k);
vector<Centroid> kmeans_seq(vector<Point>& points, vector<Centroid>& centroids, int max_iters);
vector<Centroid> kmeans_omp(vector<Point>& points, vector<Centroid>& centroids, int max_iters);

#endif
