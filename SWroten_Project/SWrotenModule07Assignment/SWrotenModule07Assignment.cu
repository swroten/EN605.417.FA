
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h> 
#include <cuda.h> 

#define NUMBER_OF_TESTS 5
#define NUMBER_OF_ELEMENTS 16

// Function to retrieving time
__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

// Kernel Function for Operating on input arrays and returns output array
__global__ void operate(int *x, int *y, int *z)
{
	// Get Thread Index of Element
	const int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Exit if thread index is greater than number of elements
	if (threadId >= NUMBER_OF_ELEMENTS) { return; }

	// Perform operation on x and y to get z
	z[threadId] = (x[threadId] * y[threadId]) + y[threadId];
}

// Main Function
int main(int argc, char *argv[])
{
	// Initialize Variables
	int devices;
	int test = 0;
	int *gpu_xArray;
	int *gpu_yArray;
	int *gpu_zArray;
	int *cpu_xArray;
	int *cpu_yArray;
	int *cpu_zArray;
	cudaEvent_t stop;
	cudaEvent_t start;
	cudaStream_t stream;
	float ellapsedTimeInMs = 0.0;

	// Get Device Count
	cudaGetDeviceCount(&devices);

	// Step through each device
	for (int indexOfDevice = 0; indexOfDevice < devices; indexOfDevice++)
	{
		// Initialize Local Variable
		cudaDeviceProp prop;

		// Get Properties of this device
		cudaGetDeviceProperties(&prop, indexOfDevice);

		// Print Properties of this device
		printf("\n");
		printf("Device Number: %d\n", indexOfDevice);
		printf("  Device name: %s\n", prop.name);
		printf("  Warp Size: %i\n", prop.warpSize);
		printf("  Max Threads Per Block: %i\n", prop.maxThreadsPerBlock);
	}

	// Create Stream
	cudaStreamCreate(&stream);

	// Allocate Memory for GPU based params
	cudaMalloc((void **)&gpu_xArray, NUMBER_OF_ELEMENTS * sizeof(int));
	cudaMalloc((void **)&gpu_yArray, NUMBER_OF_ELEMENTS * sizeof(int));
	cudaMalloc((void **)&gpu_zArray, NUMBER_OF_ELEMENTS * sizeof(int));

	// Allocate Memory for CPU based params
	cudaHostAlloc((void **)&cpu_xArray, NUMBER_OF_ELEMENTS * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void **)&cpu_yArray, NUMBER_OF_ELEMENTS * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void **)&cpu_zArray, NUMBER_OF_ELEMENTS * sizeof(int), cudaHostAllocDefault);

	// Run Test Multiple times for better understanding
	while (test < NUMBER_OF_TESTS)
	{
		printf("\n");
		printf("Test %i: \n", test);

		// Assign random Numbers for xArray and yArray
		for (int i = 0; i < NUMBER_OF_ELEMENTS; i++)
		{
			cpu_xArray[i] = (rand() % 100);
			cpu_yArray[i] = (rand() % 100);
			cpu_zArray[i] = 0;
		}

		// Async memcopy from host to device using stream
		cudaMemcpyAsync(gpu_xArray, cpu_xArray, NUMBER_OF_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(gpu_yArray, cpu_yArray, NUMBER_OF_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(gpu_zArray, cpu_zArray, NUMBER_OF_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice, stream);

		// Keep Track of Start Time
		start = get_time();

		//Operate on Input Arrays using Stream
		operate << <1, NUMBER_OF_ELEMENTS, 0, stream >> >(gpu_xArray, gpu_yArray, gpu_zArray);

		// Wait for all Streams to complete
		cudaStreamSynchronize(stream);

		// Keep Track of Stop Time 
		stop = get_time();

		// Synchronize Events
		ellapsedTimeInMs = 0;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ellapsedTimeInMs, start, stop);

		// Copy result of operation from GPU to CPU Memory
		cudaMemcpy(cpu_zArray, gpu_zArray, NUMBER_OF_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);

		// Iterate through the arrays and print 
		for (int i = 0; i < NUMBER_OF_ELEMENTS; i++)
		{
			printf("  ((%i * %i) + %i) = %i -> %d\n", cpu_xArray[i], cpu_yArray[i], cpu_yArray[i], cpu_zArray[i], (((cpu_xArray[i] * cpu_yArray[i]) + cpu_yArray[i]) == cpu_zArray[i]));
		}

		// Write out total time to complete
		printf("Results:\n");
		printf("  Time: %f ms\n", ellapsedTimeInMs);

		// iterate test
		test++;
	}

	// Free Up Memory
	cudaFreeHost(cpu_xArray);
	cudaFreeHost(cpu_yArray);
	cudaFreeHost(cpu_zArray);
	cudaFree(gpu_xArray);
	cudaFree(gpu_yArray);
	cudaFree(gpu_zArray);

	// Reset Device
	cudaDeviceReset();

	// return
	return EXIT_SUCCESS;
}
