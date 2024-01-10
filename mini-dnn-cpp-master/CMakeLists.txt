cmake_minimum_required (VERSION 3.0)
project (myproject)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.3/bin/nvcc") #Specify for local GPU
enable_language(CUDA)
find_package(CUDA REQUIRED)	# Find the CUDA libraries
include_directories(${CUDA_INCLUDE_DIRS})	# Specify the CUDA include directories for the project


include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")
# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)
# 添加 math 子目录
add_subdirectory(src)
# 指定生成目标 
add_executable(demo demo.cc)
add_executable(train train.cc)
# 添加链接库
target_link_libraries(demo MiniDNNLib)
target_link_libraries(train MiniDNNLib)
target_link_libraries(demo MiniDNNLib ${CUDA_LIBRARIES})    # Link the CUDA libraries to the project