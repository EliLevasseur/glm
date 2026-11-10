cmake_minimum_required(VERSION 3.20)
project(GLM LANGUAGES CXX)

# optimize training by default while allowing an explicit Debug build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# almost twice as fast, but the programs only run on CPUs like the one that built them
option(GLM_NATIVE "Compile for this machine's CPU (-march=native)" ON)
if(GLM_NATIVE)
	add_compile_options(-march=native)
endif()

# Use an installed nnet if CMake can find one, otherwise download it from GitHub
# and build it here. To build against a local clone instead, configure with
# -DFETCHCONTENT_SOURCE_DIR_NNET=/path/to/neural-network
set(GLM_NNET_VERSION "main" CACHE STRING "nnet branch, tag or commit to download")
find_package(nnet CONFIG QUIET)
if(nnet_FOUND)
	message(STATUS "Using the installed nnet in ${nnet_DIR}")
else()
	include(FetchContent)
	FetchContent_Declare(nnet
		GIT_REPOSITORY https://github.com/EliLevasseur/neural-network.git
		GIT_TAG ${GLM_NNET_VERSION}
		# GLM only needs nnet itself, not any nested repos
		GIT_SUBMODULES ""
	)
	FetchContent_MakeAvailable(nnet)
endif()

add_library(glm_core STATIC
	src/text_file.cpp
	src/tokenizer.cpp
	src/training_windows.cpp
	src/batch.cpp
	src/generate.cpp
	src/model.cpp
	src/book.cpp
	src/book_trainer.cpp
	src/checkpoints.cpp
)
target_include_directories(glm_core PUBLIC include)
target_compile_options(glm_core PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(glm_core PUBLIC nnet::framework)

add_executable(glm src/main.cpp)
target_compile_options(glm PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(glm PRIVATE glm_core)

# writes from the saved model without retraining
add_executable(glm_write src/write.cpp)
target_compile_options(glm_write PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(glm_write PRIVATE glm_core)
