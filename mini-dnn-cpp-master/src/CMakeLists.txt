# Find all source files in the current directory, including subdirectories
# and save the names to the DIR_LIB_SRCS variable
aux_source_directory(. DIR_LIB_SRCS)
aux_source_directory(./layer DIR_LIB_SRCS)
aux_source_directory(./loss DIR_LIB_SRCS)
aux_source_directory(./optimizer DIR_LIB_SRCS)

# Generate the main library named MiniDNNLib using the source files in DIR_LIB_SRCS
add_library(MiniDNNLib ${DIR_LIB_SRCS})

# Find all source files in the ./layer/cuda_functions directory
# and save the names to the DIR_LIB_CUDA_SRCS variable
aux_source_directory(./layer/cuda_functions DIR_LIB_CUDA_SRCS)

# Generate the CUDA library named MiniDNNCudaLib using the source files in DIR_LIB_CUDA_SRCS
add_library(MiniDNNCudaLib ${DIR_LIB_CUDA_SRCS})

# Link the CUDA library MiniDNNCudaLib with the main library MiniDNNLib
target_link_libraries(MiniDNNLib MiniDNNCudaLib)