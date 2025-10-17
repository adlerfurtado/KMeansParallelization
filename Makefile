# Makefile
# Compiladores
CXX = g++
NVCC = nvcc

# Flags de compilação
CXXFLAGS = -O2 -Wall
OMPFLAGS = -fopenmp
CUDAFLAGS = -O2

# Pastas
SRC_DIR = src
BIN_DIR = bin

# Arquivos-fonte
SEQ_SRC = $(SRC_DIR)/kmeans_seq.cpp
OMP_SRC = $(SRC_DIR)/kmeans_omp.cpp
CUDA_SRC = $(SRC_DIR)/kmeans_cuda.cu

# Executáveis
SEQ_EXE = $(BIN_DIR)/kmeans_seq
OMP_EXE = $(BIN_DIR)/kmeans_omp
CUDA_EXE = $(BIN_DIR)/kmeans_cuda

# Cria a pasta bin se não existir
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# ===============================
# Regras de compilação
# ===============================

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

# ===============================
# Limpeza
# ===============================

clean:
	rm -rf $(BIN_DIR)

# ===============================
# Execução rápida
# ===============================

run-seq: $(SEQ_EXE)
	./$(SEQ_EXE)

run-omp: $(OMP_EXE)
	./$(OMP_EXE)

run-cuda: $(CUDA_EXE)
	./$(CUDA_EXE)

# ===============================

# Compilar todas as versões = make
# Rodar versão sequencial = make run-seq
# Rodar versão com OpenMP = make run-omp
# Rodar versão com CUDA = make run-cuda
# Limpar binários = make clean