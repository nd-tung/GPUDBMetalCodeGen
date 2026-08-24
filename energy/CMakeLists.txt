cmake_minimum_required(VERSION 3.25)

project(joule_operators VERSION 0.1.0 LANGUAGES CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(JOULE_BUILD_TESTS "Build the unit tests" ON)
option(JOULE_BUILD_METAL "Compile the Metal kernels" ON)

add_library(joule_core
    src/metrics/powermetrics.cpp
    src/metrics/shelly.cpp
    src/operators/cpu/aggregates.cpp
    src/operators/cpu/filter_project.cpp
    src/operators/cpu/hash_join.cpp
    src/operators/cpu/relational.cpp
    src/operators/cpu/scan_sum.cpp
    src/operators/cpu/topk.cpp
    src/operators/cpu/tpch_q1.cpp
    src/operators/cpu/tpch_q14.cpp
    src/operators/cpu/tpch_q6.cpp
    src/operators/cpu/tpch_q6_unfused.cpp
    src/tpch/colbin.cpp
    src/tpch/lineitem.cpp
    src/tpch/orders.cpp
    src/tpch/part.cpp
)
add_library(joule::core ALIAS joule_core)

target_include_directories(joule_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(joule_core PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
    target_compile_options(joule_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
    target_compile_options(joule_core PRIVATE $<$<CONFIG:Release>:-mcpu=native>)
endif()

add_executable(joule-measure src/measure_main.cpp)
target_link_libraries(joule-measure PRIVATE joule::core)

add_executable(joule-shelly-probe src/shelly_probe_main.cpp)
target_link_libraries(joule-shelly-probe PRIVATE joule::core)

add_executable(joule-benchmark src/benchmark_main.cpp)
target_link_libraries(joule-benchmark PRIVATE joule::core)

add_executable(joule-tpch-benchmark src/tpch_benchmark_main.cpp)
target_link_libraries(joule-tpch-benchmark PRIVATE joule::core)

if(JOULE_BUILD_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "Metal kernels can only be built on Apple platforms")
    endif()

    find_program(XCRUN_EXECUTABLE xcrun REQUIRED)

    FetchContent_Declare(
        metal_cpp
        GIT_REPOSITORY https://github.com/apple/metal-cpp.git
        GIT_TAG release/metal-cpp_macOS15.2_iOS18.2
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(metal_cpp)

    target_sources(joule_core PRIVATE
        src/operators/gpu/aggregates.cpp
        src/operators/gpu/filter_project.cpp
        src/operators/gpu/hash_join.cpp
        src/operators/gpu/relational.cpp
        src/operators/gpu/scan_sum.cpp
        src/operators/gpu/topk.cpp
        src/operators/gpu/tpch_q1.cpp
        src/operators/gpu/tpch_q14.cpp
        src/operators/gpu/tpch_q6.cpp
        src/operators/gpu/tpch_q6_unfused.cpp
    )
    target_include_directories(joule_core PRIVATE "${metal_cpp_SOURCE_DIR}")
    target_link_libraries(joule_core PUBLIC "-framework Foundation" "-framework Metal")

    set(METAL_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/metal")
    set(SCAN_SUM_METAL_AIR "${METAL_OUTPUT_DIR}/scan_sum.air")
    set(RELATIONAL_METAL_AIR "${METAL_OUTPUT_DIR}/relational.air")
    set(FILTER_PROJECT_METAL_AIR "${METAL_OUTPUT_DIR}/filter_project.air")
    set(HASH_JOIN_METAL_AIR "${METAL_OUTPUT_DIR}/hash_join.air")
    set(TPCH_Q6_METAL_AIR "${METAL_OUTPUT_DIR}/tpch_q6.air")
    set(TPCH_ANALYTICS_METAL_AIR "${METAL_OUTPUT_DIR}/tpch_analytics.air")
    set(METAL_LIBRARY "${METAL_OUTPUT_DIR}/joule.metallib")

    add_custom_command(
        OUTPUT "${METAL_LIBRARY}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${METAL_OUTPUT_DIR}"
        COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
                -c "${CMAKE_CURRENT_SOURCE_DIR}/src/operators/gpu/scan_sum.metal"
                -o "${SCAN_SUM_METAL_AIR}"
        COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
                -c "${CMAKE_CURRENT_SOURCE_DIR}/src/operators/gpu/relational.metal"
                -o "${RELATIONAL_METAL_AIR}"
        COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
                -c "${CMAKE_CURRENT_SOURCE_DIR}/src/operators/gpu/filter_project.metal"
                -o "${FILTER_PROJECT_METAL_AIR}"
        COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
                -c "${CMAKE_CURRENT_SOURCE_DIR}/src/operators/gpu/hash_join.metal"
                -o "${HASH_JOIN_METAL_AIR}"
        COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
                -c "${CMAKE_CURRENT_SOURCE_DIR}/src/operators/gpu/tpch_q6.metal"
                -o "${TPCH_Q6_METAL_AIR}"
        COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
                -c "${CMAKE_CURRENT_SOURCE_DIR}/src/operators/gpu/tpch_analytics.metal"
                -o "${TPCH_ANALYTICS_METAL_AIR}"
        COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib
                "${SCAN_SUM_METAL_AIR}" "${RELATIONAL_METAL_AIR}"
                "${FILTER_PROJECT_METAL_AIR}" "${HASH_JOIN_METAL_AIR}"
                "${TPCH_Q6_METAL_AIR}" "${TPCH_ANALYTICS_METAL_AIR}"
                -o "${METAL_LIBRARY}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/src/operators/gpu/scan_sum.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/operators/gpu/relational.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/operators/gpu/filter_project.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/operators/gpu/hash_join.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/operators/gpu/tpch_q6.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/operators/gpu/tpch_analytics.metal"
        COMMENT "Compiling Metal kernels"
        VERBATIM
    )
    add_custom_target(joule_metal ALL DEPENDS "${METAL_LIBRARY}")
    add_dependencies(joule-benchmark joule_metal)
    add_dependencies(joule-tpch-benchmark joule_metal)
    target_compile_definitions(
        joule-benchmark PRIVATE JOULE_DEFAULT_METALLIB_PATH="${METAL_LIBRARY}")
    target_compile_definitions(
        joule-tpch-benchmark PRIVATE JOULE_DEFAULT_METALLIB_PATH="${METAL_LIBRARY}")
endif()

if(JOULE_BUILD_TESTS)
    enable_testing()
    add_executable(powermetrics_parser_test tests/powermetrics_parser_test.cpp)
    target_link_libraries(powermetrics_parser_test PRIVATE joule::core)
    add_test(NAME powermetrics_parser COMMAND powermetrics_parser_test)

    add_executable(shelly_test tests/shelly_test.cpp)
    target_link_libraries(shelly_test PRIVATE joule::core)
    add_test(NAME shelly COMMAND shelly_test)

    add_executable(scan_sum_cpu_test tests/scan_sum_cpu_test.cpp)
    target_link_libraries(scan_sum_cpu_test PRIVATE joule::core)
    add_test(NAME scan_sum_cpu COMMAND scan_sum_cpu_test)

    add_executable(tpch_q6_test tests/tpch_q6_test.cpp)
    target_link_libraries(tpch_q6_test PRIVATE joule::core)
    add_test(NAME tpch_q6 COMMAND tpch_q6_test)

    add_executable(tpch_analytics_test tests/tpch_analytics_test.cpp)
    target_link_libraries(tpch_analytics_test PRIVATE joule::core)
    add_test(NAME tpch_analytics COMMAND tpch_analytics_test)

    if(JOULE_BUILD_METAL)
        add_dependencies(tpch_q6_test joule_metal)
        target_compile_definitions(
            tpch_q6_test PRIVATE JOULE_DEFAULT_METALLIB_PATH="${METAL_LIBRARY}")
        add_dependencies(tpch_analytics_test joule_metal)
        target_compile_definitions(
            tpch_analytics_test PRIVATE JOULE_DEFAULT_METALLIB_PATH="${METAL_LIBRARY}")
        foreach(GPU_KERNEL baseline multi-item simdgroup)
            add_test(
                NAME scan_sum_gpu_${GPU_KERNEL}
                COMMAND joule-benchmark
                    --backend gpu
                    --gpu-kernel ${GPU_KERNEL}
                    --threadgroup-width 256
                    --batch-size 1
                    --warmup-iterations 1
                    --input-pattern signed
                    --rows 1000003
                    --duration-ms 0)
        endforeach()
    endif()
endif()
