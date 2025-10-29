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

// Função para ler pontos de um arquivo
vector<Point> read_points(const string& filename, int colX=0, int colY=1) {
    vector<Point> points;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Erro ao abrir o arquivo: " << filename << endl;
        exit(1);
    }

    // double x, y;
    // while (file >> x >> y) {
    //     points.push_back({x, y, -1});
    // }

    // if (points.empty()) {
    //     cerr << "Nenhum ponto encontrado no arquivo." << endl;
    //     exit(1);
    // }

    string line;
    bool header = true;

    while (getline(file, line)) {
        if (header) { // Se quiser pular cabeçalho
            header = false;
            continue;
        }

        stringstream ss(line);
        string value;
        vector<string> row;

        // Divide linha por vírgulas
        while (getline(ss, value, ',')) {
            row.push_back(value);
        }

        if (row.size() > max(colX, colY)) {
            Point p;
            p.x = stod(row[colX]);
            p.y = stod(row[colY]);
            p.cluster = -1;
            points.push_back(p);
        }
    }

    file.close();
    return points;
}

int main(int argc, char* argv[]) {
    
    int  num_threads = -1;

    if (argc < 3) {
        cout << "Uso: ./kmeans_seq <arquivo_dados> <k> [num_threads]\n";
        cout << "Exemplo: ./kmeans_seq data/points.txt 3 [num_threads]\n";
        return 1;
    }

    if (argc == 4) {
        num_threads = atoi(argv[3]);
    } else {
        num_threads = omp_get_max_threads();
    }

    omp_set_num_threads(num_threads);

    string filename = argv[1];
    int k = stoi(argv[2]);

    srand(time(NULL));

    cout << "Carregando arquivo...\n";
    vector<Point> points = read_points(filename);
    vector<Centroid> initial_centroids = initialize_centroids(points, k);

    vector<Centroid> results_seq;
    vector<Centroid> results_omp;

    double start_time, end_time;

    // Executa versão sequencial 3 vezes e reporta o mínimo
    double min_serial = 1e30;
    cout << "Executando versão sequencial.\n";
    for (int i = 0; i < 3; ++i) {
        start_time = CycleTimer::currentSeconds();
        results_seq = kmeans_seq(points, initial_centroids, 100);
        end_time = CycleTimer::currentSeconds();
        
        min_serial = min(min_serial, end_time - start_time);
    }

    // Executa versão OpenMP 3 vezes e reporta o mínimo
    double min_omp = 1e30;
    cout << "Executando versão OpenMP com " << num_threads << " threads\n";
    for (int i = 0; i < 3; ++i) {
        start_time = CycleTimer::currentSeconds();
        results_omp = kmeans_seq(points, initial_centroids, 100);
        end_time = CycleTimer::currentSeconds();

        min_omp = min(min_omp, end_time - start_time);
    }

    // Testa corretude dos algoritmos
    sort(results_seq.begin(), results_seq.end());
    sort(results_omp.begin(), results_omp.end());

    if (results_seq.size() == results_omp.size() && 
        equal(results_seq.begin(), results_seq.end(), results_omp.begin())) {
        cout << "Algoritmos corretos.\n";
    } else {
        cout << "Resultados divergiram.\n";

        cout << "Centróides (Sequencial):\n";
        for (int j = 0; j < k; ++j) {
            cout << "(" << results_seq[j].x << ", " << results_seq[j].y << ")  ";
        }
        cout << endl;

        cout << "Centróides (OpenMP):\n";
        for (int j = 0; j < k; ++j) {
            cout << "(" << results_omp[j].x << ", " << results_omp[j].y << ")  ";
        }
        cout << endl;
    }

    double speedup = min_serial / min_omp;
    printf("Tempo total versão Sequencial: \t[%.3f] ms\n", min_serial * 1000);
    printf("Tempo total versão OpenMP: \t[%.3f] ms\n", min_omp * 1000);
    printf("\t\t\t\t(%.2fx speedup from OpenMP)\n", speedup);

    // para compilar: g++ -I../ -std=c++11 -fopenmp -O3 -g -o kmeans  main.cpp common.cpp
    return 0;
}
