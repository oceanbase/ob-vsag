

cmake_minimum_required(VERSION 3.0)
project(ObVectorDeps VERSION 1.0 LANGUAGES CXX)

# Set the default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
# introduce 3rd library vsag
include(FetchContent)
include (${CMAKE_BINARY_DIR}/extern/vsag.cmake)

# set path
set(VSAG_BASE_DIR ${CMAKE_BINARY_DIR})
set(VSAG_LIB_DIR ${VSAG_BASE_DIR}/_deps)
set(OPENBLAS_LINK_DIR "${VSAG_LIB_DIR}/vsag-build/openblas/install/lib/")

# Create shared library
link_directories(${OPENBLAS_LINK_DIR})
add_library(ob_vsag SHARED ob_vsag_lib.cpp)
target_compile_options(ob_vsag PRIVATE -std=c++17)
target_include_directories(ob_vsag PRIVATE
                           ${VSAG_LIB_DIR}/vsag-src/include
                           ${VSAG_LIB_DIR}/nlohmann_json-src/include
                           ${VSAG_LIB_DIR}/roaringbitmap-src/include
                           ${VSAG_LIB_DIR}/fmt-src/include)
target_compile_definitions(ob_vsag PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(ob_vsag PUBLIC vsag_static gomp -static-libstdc++ -static-libgcc)
add_dependencies(ob_vsag vsag_static)

# Create static library
add_library(ob_vsag_static STATIC ob_vsag_lib.cpp)
target_compile_options(ob_vsag_static PRIVATE -std=c++17)
target_compile_definitions(ob_vsag_static PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
target_include_directories(ob_vsag_static PUBLIC
                           ${VSAG_LIB_DIR}/vsag-src/include
                           ${VSAG_LIB_DIR}/nlohmann_json-src/include
                           ${VSAG_LIB_DIR}/roaringbitmap-src/include
                           ${VSAG_LIB_DIR}/fmt-src/include)
target_link_libraries(ob_vsag PUBLIC vsag_static gomp -static-libstdc++ -static-libgcc)
add_dependencies(ob_vsag vsag_static)

add_subdirectory (example)
