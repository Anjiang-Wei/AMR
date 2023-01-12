file(GLOB src_files *.cpp)
file(GLOB  hd_files *.h)

add_library(utilities OBJECT ${src_files} ${hd_files})

target_include_directories(utilities
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(utilities PUBLIC Legion::Legion)
