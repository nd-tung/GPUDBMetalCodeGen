// Metal-cpp private implementation — must be in exactly ONE translation unit
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "infra.h"

// Global dataset configuration — definitions
std::string g_dataset_path = "data/SF-1/"; // Default to SF-1
bool g_sf100_mode = false; // true when running SF100 chunked execution
size_t g_chunk_rows_override = 0;  // 0 = use adaptive
bool g_double_buffer = true;       // default: double-buffer
