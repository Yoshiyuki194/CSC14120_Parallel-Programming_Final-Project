# Find all source files in the current directory
# and save the names to the DIR_LIB_SRCS variable
aux_source_directory(. DIR_LIB_SRCS)
aux_source_directory(./layer DIR_LIB_SRCS)
aux_source_directory(./loss DIR_LIB_SRCS)
aux_source_directory(./optimizer DIR_LIB_SRCS)

# Generate the main library
add_library (MiniDNNLib ${DIR_LIB_SRCS})

# Find all source files in the ./layer/cuda directory
# and save the names to the DIR_LIB_CUDA_SRCS variable
aux_source_directory(./layer/cuda DIR_LIB_CUDA_SRCS)

# Generate the CUDA library
add_library (MiniDNNCudaLib ${DIR_LIB_CUDA_SRCS})

# Link the CUDA library with the main library
target_link_libraries(MiniDNNLib MiniDNNCudaLib)