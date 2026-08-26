cmake_minimum_required(VERSION 3.15...4.0)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

if(MSVC AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.30)
    set(OpenMP_RUNTIME_MSVC experimental)
endif()
find_package(OpenMP)

set(MDESCRIPTOR_CPP_SOURCES
    cpp/src/common/control.cpp
    cpp/src/common/neighbor.cpp
    cpp/src/standalone/soap.cpp
    cpp/src/standalone/soap_turbo.cpp
    cpp/src/standalone/acsf.cpp
    cpp/src/standalone/c00ps_mlff.cpp
    cpp/src/standalone/coulomb_matrix.cpp
    cpp/src/standalone/sine_matrix.cpp
    cpp/src/standalone/ewald_sum_matrix.cpp
    cpp/src/standalone/matrix_dispatch.cpp
    cpp/src/standalone/mbtr.cpp
    cpp/src/standalone/ead.cpp
    cpp/src/standalone/mtp.cpp
    cpp/src/standalone/mtp4.cpp
    cpp/src/standalone/rotational_descriptors.cpp
    cpp/src/common/nep.cpp
    cpp/src/model_backed/dpa4.cpp
    cpp/src/model_backed/dpa4_wigner.cpp
    cpp/src/model_backed/dpa4c.cpp
    cpp/src/standalone/atomic_composition.cpp
    cpp/src/standalone/sorted_distances.cpp
    cpp/src/standalone/neighbor_list.cpp
    cpp/src/standalone/spherical_expansion.cpp
    cpp/src/standalone/spherical_expansion_by_pair.cpp
    cpp/src/bindings/module.cpp
)

pybind11_add_module(_native ${MDESCRIPTOR_CPP_SOURCES})
target_compile_features(_native PRIVATE cxx_std_17)
target_include_directories(_native PRIVATE cpp/include cpp/src cpp/src/common cpp/src/standalone)
if(OpenMP_CXX_FOUND)
    if(MSVC AND CMAKE_VERSION VERSION_LESS 3.30)
        # /openmp:experimental subsumes /openmp and enables omp simd.
        target_compile_options(_native PRIVATE /openmp:experimental)
    else()
        target_link_libraries(_native PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

install(
    TARGETS _native
    LIBRARY DESTINATION mdescriptor
    RUNTIME DESTINATION mdescriptor
)
