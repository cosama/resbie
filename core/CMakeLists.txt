cmake_minimum_required(VERSION 3.22)
project(resbie_offline LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-g)

find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(GTSAM CONFIG REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters registration)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# vendor/ holds edited copies of the two upstreams (see DESIGN.md).
set(BIEVR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/bievr")
set(RESPLE "${CMAKE_CURRENT_SOURCE_DIR}/vendor/resple")

pybind11_add_module(_core NO_EXTRAS
  cpp/bindings.cpp
  "${BIEVR}/src/bievr_map.cpp"
  "${BIEVR}/src/preprocess.cpp"
  "${BIEVR}/modules/scancontext/src/scan_context.cpp"
  "${BIEVR}/modules/pgo/src/loop_closer.cpp"
)

target_include_directories(_core PRIVATE
  cpp
  "${RESPLE}"
  "${BIEVR}/include"
  "${BIEVR}/modules/scancontext/include"
  "${BIEVR}/modules/pgo/include"
  "${GTSAM_INCLUDE_DIR}"
  ${PCL_INCLUDE_DIRS}
)
target_compile_definitions(_core PRIVATE
  # Eigen sizes its GEMM blocks from the cache sizes CPUID reports on the core
  # that runs the first product. Hybrid CPUs (P/E cores: 48 vs 32 KB L1d)
  # then sum in a different order and runs differ in the last bit, which the
  # filter grows into metres. Fixed sizes keep results core-independent.
  EIGEN_NO_CPUID
  # With OpenMP on, Eigen splits large products over omp_get_max_threads()
  # threads and then blocks the summed dimension differently than on one
  # thread: OMP_NUM_THREADS=1 and >1 differ once an update exceeds 320 rows.
  EIGEN_DONT_PARALLELIZE)
target_compile_options(_core PRIVATE -O3)
target_link_libraries(_core PRIVATE
  Eigen3::Eigen
  TBB::tbb
  OpenMP::OpenMP_CXX
  yaml-cpp::yaml-cpp
  gtsam
  ${PCL_LIBRARIES}
)

install(TARGETS _core LIBRARY DESTINATION resbie)
