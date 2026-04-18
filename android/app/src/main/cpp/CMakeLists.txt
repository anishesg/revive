cmake_minimum_required(VERSION 3.22.1)
project(revive-llama)

# Path to llama.cpp source (cloned alongside the project)
set(LLAMA_DIR "${CMAKE_SOURCE_DIR}/../../../../../../llama.cpp")

# If llama.cpp is not found at the expected path, try relative to project
if(NOT EXISTS "${LLAMA_DIR}/CMakeLists.txt")
    set(LLAMA_DIR "${CMAKE_SOURCE_DIR}/../../../../../llama.cpp")
endif()

if(NOT EXISTS "${LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp not found. Clone it: git clone --depth 1 https://github.com/ggerganov/llama.cpp <project-root>/llama.cpp")
endif()

# Build llama.cpp as a static library
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_DIR} llama.cpp)

# JNI bridge
add_library(revive-llama SHARED native-lib.cpp)

target_include_directories(revive-llama PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
)

target_link_libraries(revive-llama
    llama
    ggml
    common
    log
)
