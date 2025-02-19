#set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
# download and compile vsag
include (FetchContent)
#set(vsag_BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/vsag-src/)
FetchContent_Declare(
  vsag
  URL https://vsagcache.oss-rg-china-mainland.aliyuncs.com/vsag/v0.13.1.tar.gz
  URL_HASH MD5=3801d7eb21233e9f2345a698ee2b8e87
                DOWNLOAD_NO_PROGRESS 0
                INACTIVITY_TIMEOUT 5
                TIMEOUT 120
)

set(ENABLE_INTEL_MKL OFF)
set(ENABLE_CXX11_ABI OFF)
set (ROARING_DISABLE_AVX512 ON)
set (ENABLE_PYBINDS ON)
FetchContent_GetProperties(vsag)
if(NOT vsag_POPULATED)
  FetchContent_Populate(vsag)
  add_subdirectory(${vsag_SOURCE_DIR} ${vsag_BINARY_DIR})
endif()

FetchContent_MakeAvailable(vsag)
