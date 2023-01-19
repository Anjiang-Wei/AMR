project(AMRH)
cmake_minimum_required(VERSION 3.16)

cmake_policy(SET CMP0074 NEW)

find_package(Legion REQUIRED)

add_subdirectory(util)
add_subdirectory(numerics)

add_compile_options(${CXX_BUILD_WARNING_FLAGS})

file(GLOB src_files *.cpp)
foreach (src_file ${src_files})
    get_filename_component(exec_name ${src_file} NAME_WE)
    add_executable(${exec_name} ${src_file})
    target_link_libraries(${exec_name} utilities numerics)
    target_compile_options(${exec_name} PRIVATE "-Wno-unknown-pragmas")
endforeach()
