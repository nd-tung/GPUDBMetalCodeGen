# GPU Database Metal CodeGen Makefile
# Builds: GPUDBCodegen – SQL → Metal shader codegen pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_NAME = GPUDBCodegen
CODEGEN_DIR  = codegen
BUILD_DIR    = build
BIN_DIR      = $(BUILD_DIR)/bin
OBJ_DIR      = $(BUILD_DIR)/obj

CXX      = clang++
CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -MMD -MP
INCLUDES = -Ithird_party/metal-cpp -Ithird_party/libpg_query \
		   -Icodegen -Icodegen/core -Icodegen/operators -Icodegen/execution -Icodegen/planning
FRAMEWORKS = -framework Metal -framework Foundation -framework QuartzCore \
             -L/opt/homebrew/opt/llvm/lib/c++ -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++

# Codegen sources — gather recursively from library subtrees (main stays at codegen/ root)
CODEGEN_SUBDIRS = core operators execution planning
CODEGEN_SUB_SOURCES = $(shell find $(addprefix $(CODEGEN_DIR)/,$(CODEGEN_SUBDIRS)) -name '*.cpp' -type f | sort)
CODEGEN_LIB_OBJECTS = $(patsubst $(CODEGEN_DIR)/%.cpp,$(OBJ_DIR)/codegen_%.o,$(CODEGEN_SUB_SOURCES))
CODEGEN_MAIN_OBJ    = $(OBJ_DIR)/codegen_codegen_main.o

# Shared infra used by tools without linking the whole codegen binary
INFRA_OBJ = $(OBJ_DIR)/codegen_core/infra.o

# libpg_query
PG_QUERY_DIR = third_party/libpg_query
PG_QUERY_LIB = $(PG_QUERY_DIR)/libpg_query.a

TARGET = $(BIN_DIR)/$(PROJECT_NAME)

# Colbin ingestion tool
TBL2COLBIN_SRC = tools/tbl_to_colbin.cpp
TBL2COLBIN_OBJ = $(OBJ_DIR)/tbl_to_colbin.o
TBL2COLBIN     = $(BIN_DIR)/tbl_to_colbin

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
.PHONY: all rebuild clean run tools generic-only colbin-sf1 colbin-sf10 colbin-sf20 colbin-sf50 colbin-sf100 clean-colbin

all: $(TARGET)
tools: $(TBL2COLBIN)
generic-only: $(TARGET)
	@./scripts/run_generic_only.sh

rebuild: clean all

$(TARGET): $(CODEGEN_MAIN_OBJ) $(CODEGEN_LIB_OBJECTS) | $(BIN_DIR)
	@echo "Linking $(PROJECT_NAME)..."
	$(CXX) $(CODEGEN_MAIN_OBJ) $(CODEGEN_LIB_OBJECTS) $(PG_QUERY_LIB) $(FRAMEWORKS) -o $@
	@echo "Build complete: $@"

$(OBJ_DIR)/codegen_codegen_main.o: $(CODEGEN_DIR)/codegen_main.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/codegen_%.o: $(CODEGEN_DIR)/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(TBL2COLBIN_OBJ): $(TBL2COLBIN_SRC) | $(OBJ_DIR)
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(TBL2COLBIN): $(TBL2COLBIN_OBJ) $(INFRA_OBJ) | $(BIN_DIR)
	@echo "Linking tbl_to_colbin..."
	$(CXX) $(TBL2COLBIN_OBJ) $(INFRA_OBJ) $(FRAMEWORKS) -o $@
	@echo "Build complete: $@"

$(BIN_DIR) $(OBJ_DIR) $(BUILD_DIR):
	@mkdir -p $@

# Auto-generated header dependencies (-MMD -MP)
-include $(CODEGEN_MAIN_OBJ:.o=.d) $(CODEGEN_LIB_OBJECTS:.o=.d) $(TBL2COLBIN_OBJ:.o=.d)

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

# ---------------------------------------------------------------------------
# Columnar binary format (.colbin) — primary on-disk column store. Generates
# <data_dir>/<table>.colbin for every TPC-H table that has a .tbl. Regenerates
# automatically when the underlying .tbl's size or mtime changes.
# ---------------------------------------------------------------------------
colbin-sf1: $(TBL2COLBIN)
	./$(TBL2COLBIN) data/SF-1

colbin-sf10: $(TBL2COLBIN)
	./$(TBL2COLBIN) data/SF-10

colbin-sf20: $(TBL2COLBIN)
	./$(TBL2COLBIN) data/SF-20

colbin-sf50: $(TBL2COLBIN)
	./$(TBL2COLBIN) data/SF-50

colbin-sf100: $(TBL2COLBIN)
	./$(TBL2COLBIN) data/SF-100

clean-colbin:
	@echo "Removing .colbin binaries under data/SF-*/..."
	@find data -maxdepth 2 -name '*.colbin' -print -delete 2>/dev/null || true
	@find data -maxdepth 2 -name '*.colbin.tmp' -print -delete 2>/dev/null || true
