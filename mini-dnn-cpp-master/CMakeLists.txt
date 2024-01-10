cmake_minimum_required (VERSION 3.0)
project (myproject)
enable_language(CUDA)
# Find the CUDA libraries and include files on this system
find_package(CUDA REQUIRED)	
include_directories(${CUDA_INCLUDE_DIRS})	
# Specify the CUDA include directories for the project
include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")
# Find all source files in the current directory
# and save their names to the DIR_SRCS variable
aux_source_directory(. DIR_SRCS)
# Add the math subdirectory
add_subdirectory(src)
# Specify the target to be generated
add_executable(demo demo.cc)
add_executable(train train.cc)
# Add the linking library
target_link_libraries(demo MiniDNNLib)
target_link_libraries(train MiniDNNLib)
# Link the CUDA libraries to the project
target_link_libraries(demo MiniDNNLib ${CUDA_LIBRARIES})
