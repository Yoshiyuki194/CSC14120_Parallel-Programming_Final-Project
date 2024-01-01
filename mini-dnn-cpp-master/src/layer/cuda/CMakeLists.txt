# Collect all CUDA source files in the current directory
file(GLOB CUDA_SRCS "*.cu" "*.cuh")

# Generate the CUDA library
cuda_add_library(MiniDNNCudaLib ${CUDA_SRCS})

# Specify the required CUDA libraries
target_link_libraries(MiniDNNCudaLib ${CUDA_LIBRARIES})