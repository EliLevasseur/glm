cmake_minimum_required(VERSION 3.20)
project(gmind LANGUAGES CXX)

# optimize training by default while allowing an explicit Debug build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(nnet CONFIG REQUIRED)

add_library(gmind_core STATIC
	src/text_file.cpp
	src/tokenizer.cpp
	src/training_windows.cpp
	src/batch.cpp
	src/operators/one_hot.cpp
)
target_include_directories(gmind_core PUBLIC include)
target_compile_options(gmind_core PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(gmind_core PUBLIC nnet::framework)

add_executable(gmind src/main.cpp)
target_compile_options(gmind PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(gmind PRIVATE gmind_core)
