cmake_minimum_required(VERSION 3.20)
project(tensorcraft VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -Wextra")

add_compile_options(-Wno-unused-variable -Wno-unused-parameter)

include_directories(src/include)

set(SOURCES
    src/memory.cc
    src/tensor.cc
    src/ops.cc
    src/slice.cc
)

set(HEADERS
    src/include/memory.h
    src/include/tensor.h
    src/include/common.h
    src/include/arch.h
    src/include/ops.h
    src/include/slice.h
)

add_library(tensorcraft STATIC ${SOURCES})

# Example executable
add_executable(tensor_examples examples/tensor_examples.cc)
target_link_libraries(tensor_examples tensorcraft)
add_executable(tensor_shape examples/tensor_shape.cc)
target_link_libraries(tensor_shape tensorcraft)
add_executable(tensor_init examples/tensor_init.cpp)
target_link_libraries(tensor_init tensorcraft)
add_executable(test_tensor_storage examples/test_tensor_storage.cc)
target_link_libraries(test_tensor_storage tensorcraft)
add_executable(tensor_slice_example examples/tensor_slice_example.cc)
target_link_libraries(tensor_slice_example tensorcraft)
add_executable(tensor_view_ops_test examples/tensor_view_ops_test.cc)
target_link_libraries(tensor_view_ops_test tensorcraft)

# Installation rules
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include)
