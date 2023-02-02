add_library(demolibs STATIC
        util.cpp
        tasks.cpp
    )
target_include_directories(demolibs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(demolibs SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(demolibs PUBLIC Legion::Legion ${HDF5_LIBRARIES})
target_compile_options(demolibs PUBLIC ${CXX_WARNING_FLAGS})
