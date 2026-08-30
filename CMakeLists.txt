cmake_minimum_required(VERSION 3.15...4.0)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# OpenBLAS is a build-time dependency and is vendored into the resulting
# wheel.  scipy-openblas32 deliberately exports a prefixed CBLAS ABI so that
# this extension cannot accidentally bind to NumPy/SciPy's copy of OpenBLAS.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import importlib.metadata as m; v=m.version('scipy-openblas32'); print(v)"
    RESULT_VARIABLE SCIPY_OPENBLAS_VERSION_RESULT
    OUTPUT_VARIABLE SCIPY_OPENBLAS_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE SCIPY_OPENBLAS_VERSION_ERROR
)
if(
    SCIPY_OPENBLAS_VERSION_RESULT
    OR NOT SCIPY_OPENBLAS_VERSION STREQUAL "0.3.34.106.0"
)
    message(FATAL_ERROR
        "scipy-openblas32==0.3.34.106.0 is required to build MDescriptor; "
        "found '${SCIPY_OPENBLAS_VERSION}'. ${SCIPY_OPENBLAS_VERSION_ERROR}")
endif()
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import scipy_openblas32 as p; print(p.get_include_dir())"
    RESULT_VARIABLE SCIPY_OPENBLAS_INCLUDE_RESULT
    OUTPUT_VARIABLE SCIPY_OPENBLAS_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE SCIPY_OPENBLAS_INCLUDE_ERROR
)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import scipy_openblas32 as p; print(p.get_lib_dir())"
    RESULT_VARIABLE SCIPY_OPENBLAS_LIBDIR_RESULT
    OUTPUT_VARIABLE SCIPY_OPENBLAS_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE SCIPY_OPENBLAS_LIBDIR_ERROR
)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import os, scipy_openblas32 as p; print(os.path.join(p.get_lib_dir(), p.get_library(fullname=True)))"
    RESULT_VARIABLE SCIPY_OPENBLAS_LIBRARY_RESULT
    OUTPUT_VARIABLE SCIPY_OPENBLAS_LIBRARY
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE SCIPY_OPENBLAS_LIBRARY_ERROR
)
if(
    SCIPY_OPENBLAS_INCLUDE_RESULT OR SCIPY_OPENBLAS_LIBDIR_RESULT
    OR SCIPY_OPENBLAS_LIBRARY_RESULT
)
    message(FATAL_ERROR
        "scipy-openblas32 is required to build MDescriptor: "
        "${SCIPY_OPENBLAS_INCLUDE_ERROR}${SCIPY_OPENBLAS_LIBDIR_ERROR}"
        "${SCIPY_OPENBLAS_LIBRARY_ERROR}")
endif()
if(NOT EXISTS "${SCIPY_OPENBLAS_LIBRARY}")
    message(FATAL_ERROR "scipy-openblas32 library does not exist: ${SCIPY_OPENBLAS_LIBRARY}")
endif()
if(NOT EXISTS "${SCIPY_OPENBLAS_INCLUDE_DIR}/cblas.h")
    message(FATAL_ERROR "scipy-openblas32 CBLAS headers were not found")
endif()

get_filename_component(SCIPY_OPENBLAS_PACKAGE_DIR "${SCIPY_OPENBLAS_INCLUDE_DIR}" DIRECTORY)
set(SCIPY_OPENBLAS_METADATA "${SCIPY_OPENBLAS_PACKAGE_DIR}/../scipy_openblas32-0.3.34.106.0.dist-info/METADATA")
if(NOT EXISTS "${SCIPY_OPENBLAS_METADATA}")
    # The exact dist-info directory can contain a normalized version name.
    file(GLOB SCIPY_OPENBLAS_METADATA_CANDIDATES
        "${SCIPY_OPENBLAS_PACKAGE_DIR}/../scipy_openblas32-*.dist-info/METADATA")
    list(LENGTH SCIPY_OPENBLAS_METADATA_CANDIDATES SCIPY_OPENBLAS_METADATA_COUNT)
    if(SCIPY_OPENBLAS_METADATA_COUNT EQUAL 1)
        list(GET SCIPY_OPENBLAS_METADATA_CANDIDATES 0 SCIPY_OPENBLAS_METADATA)
    endif()
endif()
if(NOT EXISTS "${SCIPY_OPENBLAS_METADATA}")
    message(FATAL_ERROR "scipy-openblas32 license metadata was not found")
endif()
get_filename_component(SCIPY_OPENBLAS_DIST_INFO_DIR "${SCIPY_OPENBLAS_METADATA}" DIRECTORY)
set(SCIPY_OPENBLAS_LICENSE "${SCIPY_OPENBLAS_DIST_INFO_DIR}/licenses/LICENSE.txt")
if(NOT EXISTS "${SCIPY_OPENBLAS_LICENSE}")
    message(FATAL_ERROR "scipy-openblas32 license file was not found")
endif()

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
    cpp/src/standalone/ace.cpp
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
option(MDESCRIPTOR_DPA4_PROFILE "Enable private DPA4 stage timing" OFF)
if(MDESCRIPTOR_DPA4_PROFILE)
    target_compile_definitions(_native PRIVATE MDESCRIPTOR_DPA4_PROFILE=1)
endif()
target_include_directories(
    _native PRIVATE
    cpp/include cpp/src cpp/src/common cpp/src/standalone
    "${SCIPY_OPENBLAS_INCLUDE_DIR}")
target_link_libraries(_native PRIVATE "${SCIPY_OPENBLAS_LIBRARY}")
if(OpenMP_CXX_FOUND)
    if(MSVC AND CMAKE_VERSION VERSION_LESS 3.30)
        # /openmp:experimental subsumes /openmp and enables omp simd.
        target_compile_options(_native PRIVATE /openmp:experimental)
    else()
        target_link_libraries(_native PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# Keep source-built wheels usable before auditwheel/delocate/delvewheel runs.
# The repair tools may rename these files and rewrite the loader paths, but
# the initial artifact already contains the complete runtime closure.
if(WIN32)
    file(GLOB SCIPY_OPENBLAS_RUNTIME_LIBS "${SCIPY_OPENBLAS_LIB_DIR}/*.dll")
    list(LENGTH SCIPY_OPENBLAS_RUNTIME_LIBS SCIPY_OPENBLAS_RUNTIME_COUNT)
    if(SCIPY_OPENBLAS_RUNTIME_COUNT EQUAL 0)
        message(FATAL_ERROR "scipy-openblas32 runtime DLLs were not found")
    endif()
    install(FILES ${SCIPY_OPENBLAS_RUNTIME_LIBS} DESTINATION mdescriptor)
elseif(APPLE)
    # scipy-openblas32's macOS wheels keep OpenBLAS in lib/ and its Fortran
    # runtime dependencies in the adjacent .dylibs/ directory.  Preserve that
    # relative layout because libscipy_openblas.dylib refers to those libraries
    # via @loader_path/../.dylibs.
    file(GLOB SCIPY_OPENBLAS_RUNTIME_LIBS
        "${SCIPY_OPENBLAS_LIB_DIR}/libscipy_openblas*.dylib*")
    file(GLOB SCIPY_OPENBLAS_SUPPORT_LIBS
        "${SCIPY_OPENBLAS_PACKAGE_DIR}/.dylibs/*.dylib*")
    list(LENGTH SCIPY_OPENBLAS_RUNTIME_LIBS SCIPY_OPENBLAS_RUNTIME_COUNT)
    list(LENGTH SCIPY_OPENBLAS_SUPPORT_LIBS SCIPY_OPENBLAS_SUPPORT_COUNT)
    if(SCIPY_OPENBLAS_RUNTIME_COUNT EQUAL 0)
        message(FATAL_ERROR "scipy-openblas32 runtime libraries were not found")
    endif()
    if(SCIPY_OPENBLAS_SUPPORT_COUNT EQUAL 0)
        message(FATAL_ERROR "scipy-openblas32 support libraries were not found")
    endif()
    install(FILES ${SCIPY_OPENBLAS_RUNTIME_LIBS} DESTINATION mdescriptor/lib)
    install(FILES ${SCIPY_OPENBLAS_SUPPORT_LIBS} DESTINATION mdescriptor/.dylibs)
    set_property(TARGET _native PROPERTY INSTALL_RPATH "@loader_path/lib")
else()
    file(GLOB SCIPY_OPENBLAS_RUNTIME_LIBS
        "${SCIPY_OPENBLAS_LIB_DIR}/libscipy_openblas*.so*"
        "${SCIPY_OPENBLAS_LIB_DIR}/libgfortran*.so*"
        "${SCIPY_OPENBLAS_LIB_DIR}/libquadmath*.so*")
    list(LENGTH SCIPY_OPENBLAS_RUNTIME_LIBS SCIPY_OPENBLAS_RUNTIME_COUNT)
    if(SCIPY_OPENBLAS_RUNTIME_COUNT EQUAL 0)
        message(FATAL_ERROR "scipy-openblas32 runtime libraries were not found")
    endif()
    install(FILES ${SCIPY_OPENBLAS_RUNTIME_LIBS} DESTINATION mdescriptor/.libs)
    set_property(TARGET _native PROPERTY INSTALL_RPATH "$ORIGIN/.libs")
endif()
install(FILES "${SCIPY_OPENBLAS_METADATA}"
    DESTINATION mdescriptor/licenses
    RENAME scipy-openblas32-METADATA.txt)
install(FILES "${SCIPY_OPENBLAS_LICENSE}"
    DESTINATION mdescriptor/licenses
    RENAME scipy-openblas32-LICENSE.txt)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/docs/gui-adaptation-baseline.md"
    DESTINATION mdescriptor/docs)

install(
    TARGETS _native
    LIBRARY DESTINATION mdescriptor
    RUNTIME DESTINATION mdescriptor
)

# The CUDA backend is an independently loadable plugin.  It is opt-in so the
# CPU wheel and its build environment remain completely CUDA-free.  A release
# build must provide an explicit architecture list, which keeps the produced
# plugin's compatibility matrix visible in the build invocation.
option(MDESCRIPTOR_BUILD_CUDA "Build the optional MDescriptor CUDA plugin" OFF)
if(MDESCRIPTOR_BUILD_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        message(FATAL_ERROR
            "MDESCRIPTOR_BUILD_CUDA requires an explicit CMAKE_CUDA_ARCHITECTURES value")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(MDESCRIPTOR_CUDA_SOURCES
        cpp/cuda/src/context.cu
        cpp/cuda/src/batch.cu
        cpp/cuda/src/neighbor_graph.cu
        cpp/cuda/src/local_descriptors.cu
        cpp/cuda/src/backend.cu
        cpp/cuda/src/bindings.cu
        # The host graph builder remains the canonical source for deterministic
        # CSR ordering; descriptor coefficients and feature assembly run in the
        # CUDA translation unit below.
        cpp/src/common/control.cpp
        cpp/src/common/neighbor.cpp
        cpp/src/standalone/neighbor_list.cpp
        cpp/src/standalone/spherical_expansion.cpp
        cpp/src/standalone/spherical_expansion_by_pair.cpp
    )
    # pybind11 otherwise attaches its ``thin_lto`` interface target at module
    # creation time when the global IPO variable has not been declared yet.
    # Declare it as disabled for this target and restore the caller's value
    # afterwards so the CPU extension keeps its existing build flags.
    set(_MDESCRIPTOR_CUDA_IPO_WAS_DEFINED FALSE)
    if(DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
        set(_MDESCRIPTOR_CUDA_IPO_WAS_DEFINED TRUE)
        set(_MDESCRIPTOR_CUDA_IPO_SAVED "${CMAKE_INTERPROCEDURAL_OPTIMIZATION}")
    endif()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
    pybind11_add_module(_cuda ${MDESCRIPTOR_CUDA_SOURCES})
    if(_MDESCRIPTOR_CUDA_IPO_WAS_DEFINED)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION "${_MDESCRIPTOR_CUDA_IPO_SAVED}")
    else()
        unset(CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    endif()
    target_compile_features(_cuda PRIVATE cxx_std_17)
    # pybind11 enables host LTO for its extension targets.  CUDA relocatable
    # device code has its own device-link step; disabling host IPO here avoids
    # the CUDA toolchain dropping the fatbin registration symbol at the final
    # Python-module link.
    set_target_properties(
        _cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        INTERPROCEDURAL_OPTIMIZATION OFF)
    target_compile_options(_cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fno-lto>)
    target_link_options(_cuda PRIVATE -fno-lto)
    target_include_directories(
        _cuda PRIVATE
        cpp/include cpp/src cpp/src/common cpp/src/standalone
        cpp/cuda/include)
    target_link_libraries(_cuda PRIVATE CUDA::cudart)
    install(
        TARGETS _cuda
        LIBRARY DESTINATION mdescriptor
        RUNTIME DESTINATION mdescriptor
    )
endif()
