#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>

#include <omp.h>

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
void kmeans_omp(vector<Point>& points, int k, int max_iters = 10) {
    int n = points.size();
    vector<Centroid> centroids = initialize_centroids(points, k);

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
            #pragma omp critical
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
            cout << "Convergência alcançada na iteração " << iter + 1 << ".\n";
            break;
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

    double t0 = omp_get_wtime();
    kmeans_omp(points, k);
    double t1 = omp_get_wtime();

    cout << "Tempo total: " << (t1 - t0) << " segundos." << endl;
    cout << "Threads usadas: " << omp_get_max_threads() << endl;

    return 0;
}
