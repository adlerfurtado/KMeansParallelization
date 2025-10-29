# Compiler and flags
CXX = g++
CXXFLAGS = -O3 -Wall -fopenmp -std=c++17

# CUDA (future support)
NVCC = nvcc
NVCCFLAGS = -O3 -std=c++17

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

# Compile object files into obj/
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp src/common.h src/CycleTimer.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean prepare

# ====================================
# Execução rápida
# ====================================
run: $(TARGET)
	./$(TARGET) $(ARGS)

# ==================================== 
# Instruções 
# ==================================== 
# Compilar todas as versões → make 
# Rodar versão sequencial e OpenMP → make run ARGS="data/housing.csv 6" 
# Limpar binários → make clean