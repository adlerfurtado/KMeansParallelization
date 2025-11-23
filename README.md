# K-Means-Parallelization
K-Means is an unsupervised machine learning algorithm that groups data into K clusters. It iteratively assigns points to the nearest centroid and recalculates centroids until convergence. Though simple, it becomes computationally expensive for large datasets, making it ideal for parallelization studies.

How to use:
1. Clone the project and navegate to the directory

cd KMeansParallelization

2. Build all versions
Compile all implementations (Sequential, OpenMP, and CUDA):

```markdown
# K-Means-Parallelization

This repository implements the K-Means clustering algorithm with multiple implementations to compare correctness and performance:

- Sequential (CPU)
- Parallel CPU using OpenMP
- GPU using CUDA

The project is intended for experimentation and benchmarking. Input data is a plain text file with one point per line: `x y` (two floats separated by whitespace).

Quick start
1. Clone the repository and change to the project directory:

```bash
cd KMeansParallelization
```

2. Build
Compile the default targets (CPU/OpenMP). If CUDA Toolkit (`nvcc`) is available the CUDA binary will also be built.

```bash
make
```

This creates these main executables under `bin/`:
- `bin/kmeans`        : main orchestrator (sequential + OpenMP benchmarking)
- `bin/kmeans_cuda`   : CUDA-enabled implementation (only if `nvcc` is present)
- `bin/run_both`      : runs sequential, OpenMP (multiple thread counts) and CUDA and prints a summary (requires `nvcc` to be present at build time)

3. Generate test data (optional)
You can generate random point files with the included Python script:

```bash
python3 generate_points.py 1000   
# creates data/random_1000.txt
```

4. Run

- Run sequential + OpenMP benchmark + CUDA:

```bash
make bin/run_both

bin/run_both data/random_10000.txt 3 
# with the archive (random_10000.txt) and number of clusters K (3)
```

- For the OpenMP version, the program will run with the following default threads: 1,2,4,8. In case you want to customize the threads list you can run by this command:

```bash
bin/run_both data/random_10000.txt 3 2,6,12 
#being the last parameter the number of threads you want to run with
```

- In case you want to save the output into a archive you can also run:

```bash
bin/run_both data/random_10000.txt 3 | tee results.txt
```

Makefile targets
- `make` : build default CPU/OpenMP binary (`bin/kmeans`) and other objects
- `make bin/kmeans_cuda` : build CUDA binary (requires nvcc)
- `make bin/run_both` : build the runner that compares OpenMP and CUDA (requires nvcc)
- `make clean` : remove `obj/` and `bin/`

Notes & troubleshooting
- CUDA: to build CUDA targets you need the NVIDIA CUDA Toolkit installed and `nvcc` available in `PATH`. On Ubuntu you can install a package or use NVIDIA's installer; compatibility depends on GPU driver and toolkit versions.
- If `nvcc` is not found the Makefile will skip CUDA targets and still build the CPU/OpenMP binaries.
- Input format: plain text file with `x y` coordinates per line. Files in `data/` follow this format.
- There are small helper programs & tests in `src/` — the `run_both` runner was added to simplify side‑by‑side comparisons.

If you want, I can add a JSON/CSV output option for `run_both` so results are easy to parse programmatically.

```
