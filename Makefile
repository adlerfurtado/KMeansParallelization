# Compiler and flags
CXX = g++
CXXFLAGS = -O3 -Wall -fopenmp -std=c++17

# CUDA (future support)
NVCC := $(shell which nvcc 2>/dev/null)
# nvcc nem sempre suporta c++17 conforme a versão instalada; usar c++14 é mais compatível
NVCCFLAGS = -O3 -std=c++14

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Sources and objects
SRC = $(SRC_DIR)/main.cpp \
      $(SRC_DIR)/kmeans_seq.cpp \
      $(SRC_DIR)/kmeans_omp.cpp \
      $(SRC_DIR)/common.cpp
OBJ = $(addprefix $(OBJ_DIR)/, $(notdir $(SRC:.cpp=.o)))
# CUDA object
OBJ_CU = $(OBJ_DIR)/kmeans_cuda.o

# Executable
TARGET = $(BIN_DIR)/kmeans

# Default rule
all: prepare $(TARGET)

# Create directories if not exist
prepare:
	mkdir -p $(OBJ_DIR) $(BIN_DIR)

# Link
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# CUDA rules: somente se nvcc estiver disponível; caso contrário, alvos informativos
ifneq ($(NVCC),)
# CUDA executable (built with nvcc to link CUDA runtime)
$(BIN_DIR)/kmeans_cuda: $(OBJ) $(OBJ_CU)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -o $@ $^

else
$(BIN_DIR)/kmeans_cuda:
	@echo "nvcc not found -- CUDA target skipped"

endif

# Compile object files into obj/
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp src/common.h src/CycleTimer.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

ifneq ($(NVCC),)
# regra para compilar CUDA source
$(OBJ_DIR)/kmeans_cuda.o: $(SRC_DIR)/kmeans_cuda.cu
	$(NVCC) $(NVCCFLAGS) -Iinclude -c $< -o $@
else
$(OBJ_DIR)/kmeans_cuda.o:
	@echo "nvcc not found -- skipping CUDA object build"
endif

# Clean build files
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean prepare

# ====================================
# Execução rápida
# ====================================
run: $(TARGET)
	./$(TARGET) $(ARGS)

# Run CUDA build
ifneq ($(NVCC),)
run-cuda: $(BIN_DIR)/kmeans_cuda
	./$(BIN_DIR)/kmeans_cuda $(ARGS)
else
run-cuda:
	@echo "nvcc not found -- cannot run CUDA build (install CUDA toolkit to enable)"
endif

# ==================================== 
# Instruções 
# ==================================== 
# Compilar todas as versões → make 
# Rodar versão sequencial e OpenMP → make run ARGS="data/housing.csv 6" 
# Limpar binários → make clean