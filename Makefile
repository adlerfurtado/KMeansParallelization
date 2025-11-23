# =====================================================
# COMPILADORES
# =====================================================

CXX = g++
CXXFLAGS = -O3 -Wall -fopenmp -std=c++17

NVCC := $(shell which nvcc 2>/dev/null)
NVCCFLAGS = -O3 -std=c++14 -Xcompiler -fopenmp

# =====================================================
# DIRETÓRIOS
# =====================================================

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# =====================================================
# FONTES
# =====================================================

CPP_MAIN = $(SRC_DIR)/main.cpp \
           $(SRC_DIR)/kmeans_seq.cpp \
           $(SRC_DIR)/kmeans_omp.cpp \
           $(SRC_DIR)/common.cpp

CPP_RUN = $(SRC_DIR)/run_both.cpp \
          $(SRC_DIR)/kmeans_seq.cpp \
          $(SRC_DIR)/kmeans_omp.cpp \
          $(SRC_DIR)/common.cpp

OBJ_MAIN = $(addprefix $(OBJ_DIR)/, $(notdir $(CPP_MAIN:.cpp=.o)))
OBJ_RUN  = $(addprefix $(OBJ_DIR)/, $(notdir $(CPP_RUN:.cpp=.o)))

CUDA_OBJ = $(OBJ_DIR)/kmeans_cuda.o

# =====================================================
# ALVO PADRÃO
# =====================================================

all: prepare $(BIN_DIR)/run_both

prepare:
	mkdir -p $(OBJ_DIR) $(BIN_DIR)

# =====================================================
# COMPILAÇÃO CPU
# =====================================================

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# =====================================================
# COMPILAÇÃO CUDA
# =====================================================

ifeq ($(NVCC),)
$(error "ERRO: nvcc não encontrado! Instale CUDA toolkit.")
endif

$(OBJ_DIR)/kmeans_cuda.o: $(SRC_DIR)/kmeans_cuda.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# =====================================================
# EXECUTÁVEL NORMAL (usa main.cpp)
# =====================================================

$(BIN_DIR)/kmeans: $(OBJ_MAIN)
	$(CXX) $(CXXFLAGS) -o $@ $^

# =====================================================
# EXECUTÁVEL run_both (USA nvcc e NÃO inclui main.cpp)
# =====================================================

$(BIN_DIR)/run_both: $(OBJ_RUN) $(CUDA_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# =====================================================
# COMANDOS
# =====================================================

run:
	./$(BIN_DIR)/run_both $(ARGS)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)