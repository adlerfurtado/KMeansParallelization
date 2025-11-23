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

using namespace std;

vector<Point> read_points(const string& filename) {
    vector<Point> points;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Erro ao abrir o arquivo: " << filename << endl;
        exit(1);
    }
    double x,y;
    while (file >> x >> y) points.push_back({x,y,-1});
    if (points.empty()) { cerr << "Nenhum ponto encontrado." << endl; exit(1); }
    return points;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Uso: run_both <arquivo> <k> [threads_comma_separated]\n";
        cout << "Exemplo: run_both data/random_100.txt 3 1,2,4,8\n";
        return 1;
    }

    string filename = argv[1];
    int k = stoi(argv[2]);

    // default thread list
    vector<int> thread_list = {1,2,4,8};
    if (argc >= 4) {
        // parse comma separated list
        thread_list.clear();
        string s = argv[3];
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            if (comma == string::npos) comma = s.size();
            int t = stoi(s.substr(pos, comma - pos));
            thread_list.push_back(t);
            pos = comma + 1;
        }
    }

    srand(time(NULL));

    cout << "Carregando pontos de: " << filename << "\n";
    vector<Point> points = read_points(filename);
    vector<Centroid> initial_centroids = initialize_centroids(points, k);

    // --- Sequencial (baseline): executar 3x e pegar mínimo
    double best_seq = INFINITY;
    vector<Centroid> res_seq;
    for (int i=0;i<3;i++) {
        vector<Point> pts = points;
        double t0 = CycleTimer::currentSeconds();
        vector<Centroid> r = kmeans_seq(pts, initial_centroids, 200);
        double t1 = CycleTimer::currentSeconds();
        if (t1 - t0 < best_seq) {
            best_seq = t1 - t0;
            res_seq = r;
        }
    }
    sort(res_seq.begin(), res_seq.end());

    struct Result { string name; int threads; double time_s; vector<Centroid> cent; bool matches_seq; };
    vector<Result> results;

    // push sequential result
    results.push_back({"Sequencial", 0, best_seq, res_seq, true});

    // --- OpenMP: para cada t na lista, executar 3x e pegar mínimo
    int max_hw = omp_get_max_threads();
    for (int t : thread_list) {
        if (t <= 0) continue;
        if (t > max_hw) continue;

        omp_set_num_threads(t);
        double best = INFINITY;
        vector<Centroid> best_cent;
        for (int i=0;i<3;i++) {
            vector<Point> pts = points;
            double t0 = CycleTimer::currentSeconds();
            vector<Centroid> r = kmeans_omp(pts, initial_centroids, 200);
            double t1 = CycleTimer::currentSeconds();
            if (t1 - t0 < best) { best = t1 - t0; best_cent = r; }
        }
        sort(best_cent.begin(), best_cent.end());
        bool match = (best_cent.size() == res_seq.size() && std::equal(best_cent.begin(), best_cent.end(), res_seq.begin()));
        results.push_back({"OpenMP", t, best, best_cent, match});
    }

    // --- CUDA: executar 3x e pegar mínimo
    double best_cuda = INFINITY;
    vector<Centroid> best_cuda_cent;
    for (int i=0;i<3;i++) {
        vector<Point> pts = points;
        double t0 = CycleTimer::currentSeconds();
        vector<Centroid> r = kmeans_cuda(pts, initial_centroids, 200);
        double t1 = CycleTimer::currentSeconds();
        if (t1 - t0 < best_cuda) { best_cuda = t1 - t0; best_cuda_cent = r; }
    }
    sort(best_cuda_cent.begin(), best_cuda_cent.end());
    bool cuda_matches = (best_cuda_cent.size() == res_seq.size() && std::equal(best_cuda_cent.begin(), best_cuda_cent.end(), res_seq.begin()));
    results.push_back({"CUDA", 0, best_cuda, best_cuda_cent, cuda_matches});

    // --- Imprimir resumo organizado
    cout << "\n================== RESUMO ==================\n";
    cout << left << setw(12) << "Metodo" << setw(10) << "Threads" << setw(16) << "Tempo (ms)" << setw(12) << "Speedup" << "Corretude" << "\n";
    cout << string(60, '-') << "\n";

    double base = results[0].time_s;
    for (auto &r : results) {
        double speed = (r.time_s > 0) ? base / r.time_s : 0.0;
        string corr = r.matches_seq ? "OK" : "DIFFER";
        string thr = (r.name == "OpenMP") ? to_string(r.threads) : "-";
        cout << left << setw(12) << r.name << setw(10) << thr << setw(16) << (r.time_s*1000.0) << setw(12) << fixed << setprecision(2) << speed << corr << "\n";
    }

    // Se houver divergências, mostrar centróides para investigação
    bool any_div = false;
    for (auto &r : results) if (!r.matches_seq) any_div = true;
    if (any_div) {
        cout << "\n-- Detalhes (centróides) --\n";
        cout << "Sequencial:\n";
        for (auto &c : res_seq) cout << "(" << c.x << "," << c.y << ") ";
        cout << "\n";
        for (auto &r : results) {
            if (r.name == "Sequencial") continue;
            if (!r.matches_seq) {
                cout << r.name << " (threads=" << (r.threads>0?to_string(r.threads):string("-")) << "):\n";
                for (auto &c : r.cent) cout << "(" << c.x << "," << c.y << ") ";
                cout << "\n";
            }
        }
    }

    return 0;
}
