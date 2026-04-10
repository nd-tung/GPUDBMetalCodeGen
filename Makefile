# GPU Database Metal CodeGen Makefile
# Builds: GPUDBCodegen – SQL → Metal shader codegen pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_NAME = GPUDBCodegen
CODEGEN_DIR  = codegen
SOURCE_DIR   = src
BUILD_DIR    = build
BIN_DIR      = $(BUILD_DIR)/bin
OBJ_DIR      = $(BUILD_DIR)/obj

CXX      = clang++
CXXFLAGS = -std=c++20 -Wall -Wextra -O3
INCLUDES = -Ithird_party/metal-cpp -Ithird_party/libpg_query \
           -Icodegen -Icodegen/core -Icodegen/operators -Icodegen/execution -Icodegen/planning \
           -Isrc
FRAMEWORKS = -framework Metal -framework Foundation -framework QuartzCore \
             -L/opt/homebrew/opt/llvm/lib/c++ -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++

# Codegen subdirectories
CODEGEN_SUBDIRS = core operators execution planning

# Codegen sources — gather from subdirs (main stays at codegen/ root)
CODEGEN_SUB_SOURCES = $(foreach d,$(CODEGEN_SUBDIRS),$(wildcard $(CODEGEN_DIR)/$(d)/*.cpp))
CODEGEN_LIB_OBJECTS = $(foreach src,$(CODEGEN_SUB_SOURCES),$(OBJ_DIR)/codegen_$(notdir $(basename $(src))).o)
CODEGEN_MAIN_OBJ    = $(OBJ_DIR)/codegen_codegen_main.o

# Shared infra
INFRA_OBJ = $(OBJ_DIR)/infra.o

# libpg_query
PG_QUERY_DIR = third_party/libpg_query
PG_QUERY_LIB = $(PG_QUERY_DIR)/libpg_query.a

TARGET = $(BIN_DIR)/$(PROJECT_NAME)

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
.PHONY: all rebuild clean run

all: $(TARGET)

rebuild: clean all

$(TARGET): $(CODEGEN_MAIN_OBJ) $(CODEGEN_LIB_OBJECTS) $(INFRA_OBJ) | $(BIN_DIR)
	@echo "Linking $(PROJECT_NAME)..."
	$(CXX) $(CODEGEN_MAIN_OBJ) $(CODEGEN_LIB_OBJECTS) $(INFRA_OBJ) $(PG_QUERY_LIB) $(FRAMEWORKS) -o $@
	@echo "Build complete: $@"

$(OBJ_DIR)/codegen_codegen_main.o: $(CODEGEN_DIR)/codegen_main.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/codegen_%.o: $(CODEGEN_DIR)/core/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/codegen_%.o: $(CODEGEN_DIR)/operators/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/codegen_%.o: $(CODEGEN_DIR)/execution/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/codegen_%.o: $(CODEGEN_DIR)/planning/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(INFRA_OBJ): $(SOURCE_DIR)/infra.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR) $(OBJ_DIR) $(BUILD_DIR):
	@mkdir -p $@

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@echo "Clean complete"

# ---------------------------------------------------------------------------
# Run  —  make run  /  make run SF=sf10  /  make run SF=sf1 Q=q6
# ---------------------------------------------------------------------------
SF ?=
Q  ?=
run: all
	@./$(TARGET) $(SF) $(Q)
