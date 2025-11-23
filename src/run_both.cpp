#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <iomanip>

#include "common.h"
#include "CycleTimer.h"

#include <cuda_runtime.h>

using namespace std;

vector<Point> read_points(const string& filename) {
    vector<Point> points;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Erro ao abrir arquivo: " << filename << endl;
        exit(1);
    }
    double x, y;
    while (file >> x >> y) points.push_back({x,y,-1});
    if (points.empty()) {
        cerr << "Nenhum ponto encontrado.\n";
        exit(1);
    }
    return points;
}

void check_gpu() {
    int count = 0;
    cudaGetDeviceCount(&count);

    if (count == 0) {
        cerr << "\nERRO: Nenhuma GPU CUDA encontrada!\n";
        exit(1);
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    cout << "GPU Detectada: " << prop.name
         << "  (Compute Capability " << prop.major << "." << prop.minor << ")\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Uso: run_both <arquivo> <k> [threads_comma]\n";
        return 1;
    }

    check_gpu();

    string filename = argv[1];
    int k = stoi(argv[2]);

    vector<int> thread_list = {1,2,4,8};
    if (argc >= 4) {
        thread_list.clear();
        string s = argv[3];
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            if (comma == string::npos) comma = s.size();
            thread_list.push_back(stoi(s.substr(pos, comma-pos)));
            pos = comma + 1;
        }
    }

    cout << "Carregando pontos...\n";
    vector<Point> points = read_points(filename);

    vector<Centroid> initial = initialize_centroids(points, k);

    struct R { string name; int th; double t; vector<Centroid> c; bool ok; };
    vector<R> results;

    // ===========================
    // SEQUENCIAL
    // ===========================

    double best_seq = 1e30;
    vector<Centroid> seq_best;

    for (int i = 0; i < 3; i++) {
        vector<Point> copy = points;
        double t0 = CycleTimer::currentSeconds();
        auto r = kmeans_seq(copy, initial, 200);
        double t1 = CycleTimer::currentSeconds();
        if (t1-t0 < best_seq) {
            best_seq = t1-t0;
            seq_best = r;
        }
    }
    sort(seq_best.begin(), seq_best.end());
    results.push_back({"Sequencial",0,best_seq,seq_best,true});


    // ===========================
    // OPENMP
    // ===========================

    for (int th : thread_list) {
        double best = 1e30;
        vector<Centroid> bestc;

        omp_set_num_threads(th);

        for (int i = 0; i < 3; i++) {
            vector<Point> copy = points;
            double t0 = CycleTimer::currentSeconds();
            auto r = kmeans_omp(copy, initial, 200);
            double t1 = CycleTimer::currentSeconds();
            if (t1-t0 < best) {
                best = t1-t0;
                bestc = r;
            }
        }

        sort(bestc.begin(), bestc.end());
        bool ok = (bestc == seq_best);

        results.push_back({"OpenMP",th,best,bestc,ok});
    }


    // ===========================
    // CUDA
    // ===========================





        // --- CUDA (double-atomic original) - optional: run if desired
        // (skipped here since we'll run float and block-reduce variants instead)

        // --- CUDA (float-sum atomic) - quick test
        double best_cuda_float = INFINITY;
        vector<Centroid> best_cuda_float_cent;
        for (int i=0;i<3;i++) {
            vector<Point> pts = points;
            double t0 = CycleTimer::currentSeconds();
            vector<Centroid> r = kmeans_cuda_float(pts, initial, 200);
            double t1 = CycleTimer::currentSeconds();
            if (t1 - t0 < best_cuda_float) { best_cuda_float = t1 - t0; best_cuda_float_cent = r; }
        }
        sort(best_cuda_float_cent.begin(), best_cuda_float_cent.end());
        bool cuda_float_matches = (best_cuda_float_cent == seq_best);
        results.push_back({"CUDA_float", 0, best_cuda_float, best_cuda_float_cent, cuda_float_matches});

        // --- CUDA (block-reduction) - reduced atomics
        double best_cuda_block = INFINITY;
        vector<Centroid> best_cuda_block_cent;
        for (int i=0;i<3;i++) {
            vector<Point> pts = points;
            double t0 = CycleTimer::currentSeconds();
            vector<Centroid> r = kmeans_cuda_blockreduce(pts, initial, 200);
            double t1 = CycleTimer::currentSeconds();
            if (t1 - t0 < best_cuda_block) { best_cuda_block = t1 - t0; best_cuda_block_cent = r; }
        }
        sort(best_cuda_block_cent.begin(), best_cuda_block_cent.end());
        bool cuda_block_matches = (best_cuda_block_cent == seq_best);
        results.push_back({"CUDA_block", 0, best_cuda_block, best_cuda_block_cent, cuda_block_matches});

    // ===========================
    // PRINT
    // ===========================

    cout << "\n================== RESUMO ==================\n";
    cout << left << setw(12) << "Metodo"
         << setw(10) << "Threads"
         << setw(14) << "Tempo(ms)"
         << setw(10) << "Speedup"
         << "Corretude\n";

    double base = best_seq;

    for (auto &r : results) {
        string th = (r.name=="OpenMP") ? to_string(r.th) : "-";
        string ok = r.ok ? "OK" : "DIFF";
        cout << left << setw(12) << r.name
             << setw(10) << th
             << setw(14) << r.t * 1000
             << setw(10) << base / r.t
             << ok << "\n";
    }

    return 0;
}