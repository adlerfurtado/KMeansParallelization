# ====================================
# Compiladores
# ====================================
CXX = g++
NVCC = nvcc

# ====================================
# Flags de compilação
# ====================================
CXXFLAGS = -O2 -Wall
OMPFLAGS = -fopenmp
CUDAFLAGS = -O2

# ====================================
# Pastas
# ====================================
SRC_DIR = src
BIN_DIR = bin
DATA_DIR = data

# ====================================
# Arquivos-fonte
# ====================================
SEQ_SRC = $(SRC_DIR)/kmeans_seq.cpp
OMP_SRC = $(SRC_DIR)/kmeans_omp.cpp
CUDA_SRC = $(SRC_DIR)/kmeans_cuda.cu

# ====================================
# Executáveis
# ====================================
SEQ_EXE = $(BIN_DIR)/kmeans_seq
OMP_EXE = $(BIN_DIR)/kmeans_omp
CUDA_EXE = $(BIN_DIR)/kmeans_cuda

# ====================================
# Criação de pastas
# ====================================
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

# ====================================
# Compilação
# ====================================
all: $(SEQ_EXE) $(OMP_EXE) $(CUDA_EXE)

# Versão sequencial
$(SEQ_EXE): $(SEQ_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

# Versão paralela (OpenMP)
$(OMP_EXE): $(OMP_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

# Versão paralela (CUDA)
$(CUDA_EXE): $(CUDA_SRC) | $(BIN_DIR)
	$(NVCC) $(CUDAFLAGS) -o $@ $<

# ====================================
# Limpeza
# ====================================
clean:
	rm -rf $(BIN_DIR)

# ====================================
# Execução rápida (com parâmetros)
# ====================================
run-seq: $(SEQ_EXE)
	./$(SEQ_EXE) $(ARGS)

run-omp: $(OMP_EXE)
	./$(OMP_EXE) $(ARGS)

run-cuda: $(CUDA_EXE)
	./$(CUDA_EXE) $(ARGS)

# ====================================
# Instruções
# ====================================
# Compilar todas as versões  → make
# Rodar versão sequencial    → make run-seq ARGS="data/points.txt 3"
# Rodar versão OpenMP        → make run-omp ARGS="data/points.txt 3"
# Rodar versão CUDA          → make run-cuda ARGS="data/points.txt 3"
# Limpar binários            → make clean
