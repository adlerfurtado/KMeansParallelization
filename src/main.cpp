#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>

#include <omp.h>

#include "common.h"
#include "CycleTimer.h"

using namespace std;

// ------------------------------------------------------------
// Lê pontos do arquivo
// ------------------------------------------------------------
vector<Point> read_points(const string& filename, int colX=0, int colY=1) {
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

    file.close();
    return points;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char* argv[]) {

    bool run_benchmark = false;
    int num_threads = -1;

    if (argc < 3) {
        cout << "Uso: ./kmeans <arquivo_dados> <k> [--threads N]\n";
        cout << "Se --threads não for usado, roda benchmark automático.\n";
        return 1;
    }

    string filename = argv[1];
    int k = stoi(argv[2]);

    // ------------------------------------------------------------
    // Interpretar argumentos
    // ------------------------------------------------------------
    if (argc == 5) {
        string flag = argv[3];
        if (flag == "--threads") {
            num_threads = atoi(argv[4]);
        } else {
            cerr << "Argumento desconhecido: " << flag << endl;
            return 1;
        }
    } else {
        // Nenhum número de threads → entra no modo benchmark
        run_benchmark = true;
    }

    srand(time(NULL));

    cout << "Carregando arquivo...\n";
    vector<Point> points = read_points(filename);
    vector<Centroid> initial_centroids = initialize_centroids(points, k);

    vector<Centroid> results_seq;
    vector<Centroid> results_omp;

    double start_time, end_time;

    // =============================================================
    // RODAR VERSÃO SEQUENCIAL (sempre faz uma vez antes do benchmark)
    // =============================================================
    double min_serial = INFINITY;
    cout << "Executando versão sequencial (baseline)...\n";

    for (int i = 0; i < 3; i++) {
        vector<Point> pts_seq = points;

        start_time = CycleTimer::currentSeconds();
        results_seq = kmeans_seq(pts_seq, initial_centroids, 200);
        end_time = CycleTimer::currentSeconds();

        min_serial = min(min_serial, end_time - start_time);
    }

    cout << "Tempo sequencial mínimo: " << min_serial*1000 << " ms\n\n";

    // =============================================================
    // SE MODO NORMAL COM --threads N
    // =============================================================
    if (!run_benchmark) {

        omp_set_num_threads(num_threads);
        cout << "Executando versão OpenMP com " << num_threads << " threads\n";

        double min_omp = INFINITY;
        for (int i = 0; i < 3; i++) {
            vector<Point> pts_omp = points;

            start_time = CycleTimer::currentSeconds();
            results_omp = kmeans_omp(pts_omp, initial_centroids, 200);
            end_time = CycleTimer::currentSeconds();

            min_omp = min(min_omp, end_time - start_time);
        }

        double speedup = min_serial / min_omp;

        printf("Tempo total versão OpenMP: \t[%.3f] ms\n", min_omp * 1000);
        printf("\t\t\t\t(%.2fx speedup from OpenMP)\n", speedup);
        return 0;
    }

    // =============================================================
    // =============================================================
    //                MODO BENCHMARK AUTOMÁTICO
    // =============================================================
    // =============================================================

    cout << "==================== MODO BENCHMARK ====================\n";

    int max_hw_threads = omp_get_max_threads();

    vector<int> thread_list = {1, 2, 4, 8};
    if (max_hw_threads > 8) thread_list.push_back(max_hw_threads);
    else if (find(thread_list.begin(), thread_list.end(), max_hw_threads) == thread_list.end())
        thread_list.push_back(max_hw_threads);

    double best_speedup = 0.0;
    int best_t = 1;

    for (int t : thread_list) {
        if (t > max_hw_threads) continue;

        omp_set_num_threads(t);
        cout << "\n--- Testando com " << t << " threads ---\n";

        double min_omp = INFINITY;

        for (int i = 0; i < 3; i++) {
            vector<Point> pts_omp = points;

            start_time = CycleTimer::currentSeconds();
            kmeans_omp(pts_omp, initial_centroids, 200);
            end_time = CycleTimer::currentSeconds();

            min_omp = min(min_omp, end_time - start_time);
        }

        double speedup = min_serial / min_omp;

        printf("OpenMP %2d threads: %.3f ms  | speedup %.2fx\n",
               t, min_omp * 1000, speedup);

        if (speedup > best_speedup) {
            best_speedup = speedup;
            best_t = t;
        }
    }

    printf("\n>>> Melhor configuração: %d threads (%.2fx speedup)\n", best_t, best_speedup);

    return 0;
}
