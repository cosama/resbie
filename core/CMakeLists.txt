cmake_minimum_required(VERSION 3.22)
project(resbie_offline LANGUAGES C CXX)

# BIEVR's headers need C++20 (concepts).
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
find_package(PCL REQUIRED COMPONENTS common io filters registration)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# vendor/ holds edited copies of the two upstreams (see vendor/README.md).
set(BIEVR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/bievr")
set(RESPLE "${CMAKE_CURRENT_SOURCE_DIR}/vendor/resple")

pybind11_add_module(_core NO_EXTRAS
  cpp/bindings.cpp
  "${BIEVR}/src/bievr_map.cpp"
  "${BIEVR}/src/common.cpp"
  "${BIEVR}/src/preprocess.cpp"
  "${BIEVR}/src/utils.cpp"
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
# BIEVR's compile mode (log++ in LPP mode).
target_compile_definitions(_core PRIVATE MODE_LPP)
target_compile_options(_core PRIVATE -O3 -fconcepts)
target_link_libraries(_core PRIVATE
  Eigen3::Eigen
  TBB::tbb
  OpenMP::OpenMP_CXX
  yaml-cpp::yaml-cpp
  gtsam
  ${PCL_LIBRARIES}
)

install(TARGETS _core LIBRARY DESTINATION resbie)
