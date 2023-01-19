file(GLOB hdr_files *.h)
add_library(numerics INTERFACE)
target_include_directories(numerics SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(numerics INTERFACE utilities)
