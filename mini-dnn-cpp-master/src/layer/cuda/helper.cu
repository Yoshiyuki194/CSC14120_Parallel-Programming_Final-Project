#include "helper.h"

char* cuda_helper::concatenate(const char *a, const char *b)
{
	char *result = (char *)malloc(strlen(a) + strlen(b) + 1);
	strcpy(result, a);
	strcat(result, b);
	return result;
}

void cuda_helper::print_device_info()
{
	CHECK(cudaGetDeviceProperties(&prop, 0));
	printf("**********GPU Device Properties**********\n");
	printf("Name: %s\n", prop.name);
	printf("Compute capability: %d.%d\n", prop.major, prop.minor);
	printf("Number of SMs: %d\n", prop.multiProcessorCount);
	printf("Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
	printf("Warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
	printf("GMEM: %zu bytes\n", prop.totalGlobalMem);
	printf("SMEM per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
	printf("SMEM per Block: %zu bytes\n", prop.sharedMemPerBlock);
	printf("*****************************************\n");
}