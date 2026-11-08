cmake_minimum_required(VERSION 3.20)
project(gmind LANGUAGES CXX)

find_package(nnet CONFIG REQUIRED)

add_library(gmind_core STATIC src/text_file.cpp src/tokenizer.cpp src/training_windows.cpp)
target_include_directories(gmind_core PUBLIC src)
target_compile_options(gmind_core PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(gmind_core PUBLIC nnet::framework)

add_executable(gmind src/main.cpp)
target_compile_options(gmind PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(gmind PRIVATE gmind_core)
