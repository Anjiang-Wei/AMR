project(AMRH)
cmake_minimum_required(VERSION 3.16)

cmake_policy(SET CMP0074 NEW)

add_compile_options(${CXX_BUILD_WARNING_FLAGS})


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CXX_WARNING_FLAGS -Wall -Wextra -Wpedantic)

find_package(HDF5 REQUIRED)
if (${HDF5_FOUND})
    include_directories(SYSTEM ${HDF5_INCLUDE_DIRS})
endif()

find_package(Legion REQUIRED)

add_subdirectory(demolibs)

file(GLOB src_files *.cpp)
foreach (src_file ${src_files})
    get_filename_component(exec_name ${src_file} NAME_WE)
    add_executable(${exec_name} ${src_file})
    target_link_libraries(${exec_name} demolibs ${HDF5_LIBRARIES})
endforeach()
