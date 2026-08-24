cmake_minimum_required(VERSION 3.15...4.0)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

if(MSVC AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.30)
    set(OpenMP_RUNTIME_MSVC experimental)
endif()
find_package(OpenMP)

set(MDESCRIPTOR_CPP_SOURCES
    cpp/src/compute_control.cpp
    cpp/src/soap.cpp
    cpp/src/soap_turbo.cpp
    cpp/src/acsf.cpp
    cpp/src/c00ps_mlff.cpp
    cpp/src/coulomb_matrix.cpp
    cpp/src/sine_matrix.cpp
    cpp/src/ewald_sum_matrix.cpp
    cpp/src/matrix_dispatch.cpp
    cpp/src/mbtr.cpp
    cpp/src/ead.cpp
    cpp/src/mtp.cpp
    cpp/src/mtp4.cpp
    cpp/src/pyxtal.cpp
    cpp/src/neighbor.cpp
    cpp/src/nep.cpp
    cpp/src/atomic_composition.cpp
    cpp/src/sorted_distances.cpp
    cpp/src/neighbor_list.cpp
    cpp/src/spherical_expansion.cpp
    cpp/src/spherical_expansion_by_pair.cpp
    cpp/src/bindings.cpp
)

pybind11_add_module(_descriptor_cpp ${MDESCRIPTOR_CPP_SOURCES})
target_compile_features(_descriptor_cpp PRIVATE cxx_std_17)
target_include_directories(_descriptor_cpp PRIVATE cpp/include)
if(OpenMP_CXX_FOUND)
    if(MSVC AND CMAKE_VERSION VERSION_LESS 3.30)
        # /openmp:experimental subsumes /openmp and enables omp simd.
        target_compile_options(_descriptor_cpp PRIVATE /openmp:experimental)
    else()
        target_link_libraries(_descriptor_cpp PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

install(
    TARGETS _descriptor_cpp
    LIBRARY DESTINATION mdescriptor
    RUNTIME DESTINATION mdescriptor
)
