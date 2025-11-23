# K-Means-Parallelization
K-Means is an unsupervised machine learning algorithm that groups data into K clusters. It iteratively assigns points to the nearest centroid and recalculates centroids until convergence. Though simple, it becomes computationally expensive for large datasets, making it ideal for parallelization studies.

How to use:
1. Clone the project and navegate to the directory

cd KMeansParallelization

2. Build all versions
Compile all implementations (Sequential, OpenMP, and CUDA):

make 

This will create these executables: 
- bin/kmeans_seq
- bin/kmeans_omp
- bin/kmeans_cuda

3. Run the Sequential version

make run-seq ARGS="data/points.txt 3"

4. Run the Cuda version 

make run-cuda ARGS="data/points.txt 3"
