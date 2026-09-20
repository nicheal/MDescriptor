cmake_minimum_required(VERSION 3.15...4.0)
project(MDescriptorCppTests LANGUAGES CXX)

include(CTest)
find_package(OpenMP)

add_executable(
    mdescriptor_cpp_public_api
    test_public_api.cpp
    ../src/common/control.cpp
    ../src/common/neighbor.cpp
)
target_compile_features(mdescriptor_cpp_public_api PRIVATE cxx_std_17)
target_include_directories(mdescriptor_cpp_public_api PRIVATE ../include)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mdescriptor_cpp_public_api PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME cpp_public_api COMMAND mdescriptor_cpp_public_api)

add_executable(
    mdescriptor_cpp_neighbor_grid_bounds
    test_neighbor_grid_bounds.cpp
    ../src/common/control.cpp
    ../src/common/neighbor.cpp
)
target_compile_features(mdescriptor_cpp_neighbor_grid_bounds PRIVATE cxx_std_17)
target_include_directories(mdescriptor_cpp_neighbor_grid_bounds PRIVATE ../include)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mdescriptor_cpp_neighbor_grid_bounds PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(
    NAME cpp_neighbor_grid_bounds
    COMMAND mdescriptor_cpp_neighbor_grid_bounds)
