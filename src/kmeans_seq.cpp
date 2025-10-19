#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <fstream>
#include <string>

using namespace std;

// Estrutura para representar um ponto (x, y)
struct Point {
    double x, y;
    int cluster;  // índice do centróide mais próximo
};

// Estrutura para representar um centróide
struct Centroid {
    double x, y;
};

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

// Função principal do K-Means
void kmeans(vector<Point>& points, int k, int max_iters = 100) {
    int n = points.size();
    vector<Centroid> centroids = initialize_centroids(points, k);

    for (int iter = 0; iter < max_iters; ++iter) {
        bool changed = false;

        // Etapa 1: atribuir cada ponto ao centróide mais próximo
        for (int i = 0; i < n; ++i) {
            double min_dist = numeric_limits<double>::max();
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

        // Se não houve mudança de cluster, convergiu
        if (!changed) {
            cout << "Convergiu em " << iter << " iterações." << endl;
            break;
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
    }

    // Mostrar resultado final
    for (int j = 0; j < k; ++j) {
        cout << "Centróide " << j << ": (" << centroids[j].x << ", " << centroids[j].y << ")\n";
    }
}

// Função para ler pontos de um arquivo
vector<Point> read_points(const string& filename) {
    vector<Point> points;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Erro ao abrir o arquivo: " << filename << endl;
        exit(1);
    }

    double x, y;
    while (file >> x >> y) {
        points.push_back({x, y, -1});
    }

    if (points.empty()) {
        cerr << "Nenhum ponto encontrado no arquivo." << endl;
        exit(1);
    }

    return points;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Uso: ./kmeans_seq <arquivo_dados> <k>\n";
        cout << "Exemplo: ./kmeans_seq data/points.txt 3\n";
        return 1;
    }

    string filename = argv[1];
    int k = stoi(argv[2]);

    srand(time(NULL));

    vector<Point> points = read_points(filename);

    clock_t start = clock();
    kmeans(points, k);
    clock_t end = clock();

    double elapsed = double(end - start) / CLOCKS_PER_SEC;
    cout << "Tempo total: " << elapsed << " segundos." << endl;

    return 0;
}
