
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <tchar.h>
#include <cusolverSp.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define BLOCKSIZE 512

using namespace std;

// Function to retrieving time
__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

__global__ void max(int *maxValueIndexArray,
	const double *inputMatrixColumnArray,
	const int columnIndexInMatrix,
	const int numberOfElements)
{
	// Create Shared Memory Array
	extern volatile __shared__ double sharedDataArray[];
	extern volatile __shared__ int sharedDataArrayIndices[];

	// Initialize Shared Data Array Values to 0
	sharedDataArray[threadIdx.x] = inputMatrixColumnArray[threadIdx.x];

	// Sychronize all Threads
	__syncthreads();

	// Check if Block Size is greater than 512
	if (BLOCKSIZE >= 512)
	{
		if (threadIdx.x < 256)
		{
			// Update shared memory 
			sharedDataArray[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 256]) ?
				sharedDataArray[threadIdx.x] : sharedDataArray[threadIdx.x + 256];
			sharedDataArrayIndices[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 256]) ?
				threadIdx.x : threadIdx.x + 256;
		}

		// Sychronize all Threads
		__syncthreads();
	}

	if (BLOCKSIZE >= 256)
	{
		if (threadIdx.x < 128)
		{
			// Update shared memory 
			sharedDataArray[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 128]) ?
				sharedDataArray[threadIdx.x] : sharedDataArray[threadIdx.x + 128];
			sharedDataArrayIndices[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 128]) ?
				threadIdx.x : threadIdx.x + 128;
		}

		// Sychronize all Threads
		__syncthreads();
	}

	if (BLOCKSIZE >= 128)
	{
		if (threadIdx.x < 64)
		{
			// Update shared memory 
			sharedDataArray[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 64]) ?
				sharedDataArray[threadIdx.x] : sharedDataArray[threadIdx.x + 64];
			sharedDataArrayIndices[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 64]) ?
				threadIdx.x : threadIdx.x + 64;
		}

		// Sychronize all Threads
		__syncthreads();
	}

	if (threadIdx.x < 32)
	{
		if (BLOCKSIZE >= 64)
		{
			// Update shared memory 
			sharedDataArray[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 32]) ?
				sharedDataArray[threadIdx.x] : sharedDataArray[threadIdx.x + 32];
			sharedDataArrayIndices[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 32]) ?
				threadIdx.x : threadIdx.x + 32;
		}

		if (BLOCKSIZE >= 32)
		{
			// Update shared memory 
			sharedDataArray[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 16]) ?
				sharedDataArray[threadIdx.x] : sharedDataArray[threadIdx.x + 16];
			sharedDataArrayIndices[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 16]) ?
				threadIdx.x : threadIdx.x + 16;
		}

		if (BLOCKSIZE >= 16)
		{
			// Update shared memory 
			sharedDataArray[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 8]) ?
				sharedDataArray[threadIdx.x] : sharedDataArray[threadIdx.x + 8];
			sharedDataArrayIndices[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 8]) ?
				threadIdx.x : threadIdx.x + 8;
		}

		if (BLOCKSIZE >= 8)
		{
			// Update shared memory 
			sharedDataArray[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 4]) ?
				sharedDataArray[threadIdx.x] : sharedDataArray[threadIdx.x + 4];
			sharedDataArrayIndices[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 4]) ?
				threadIdx.x : threadIdx.x + 4;
		}

		if (BLOCKSIZE >= 4)
		{
			// Update shared memory 
			sharedDataArray[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 2]) ?
				sharedDataArray[threadIdx.x] : sharedDataArray[threadIdx.x + 2];
			sharedDataArrayIndices[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 2]) ?
				threadIdx.x : threadIdx.x + 2;
		}

		if (BLOCKSIZE >= 2)
		{
			// Update shared memory 
			sharedDataArray[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 1]) ?
				sharedDataArray[threadIdx.x] : sharedDataArray[threadIdx.x + 1];
			sharedDataArrayIndices[threadIdx.x] = (sharedDataArray[threadIdx.x] > sharedDataArray[threadIdx.x + 1]) ?
				threadIdx.x : threadIdx.x + 1;
		}
	}

	if (threadIdx.x == 0)
	{
		maxValueIndexArray[columnIndexInMatrix] = sharedDataArrayIndices[0];
	}
}

__global__ void ludecomposition(int *pivotMatrix,
	double *matrix,
	int *maxValueIndexMatrix,
	const int numberOfElements,
	const int squareMatrixDimension)
{
	// Get Thread Index of Element
	const int columnIndexInMatrix = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Create Shared Memory Array
	extern volatile __shared__ double columnValues[];

	// Exit if thread index is greater than number of elements
	if (columnIndexInMatrix >= numberOfElements) { return; }

	// Initialize Column Values in context of this column
	double matrixCurrentColumnValue = 0;
	double matrixLargestColumnValue = 0;
	int pivotMatrixCurrentColumnValue = 0;
	int pivotMatrixLargestColumnValue = 0;

#pragma unroll
	// Get all row elements in current column 
	for (int rowIndex = 0; rowIndex < squareMatrixDimension; rowIndex++)
	{
		columnValues[rowIndex] = matrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix];
	}

	// Sync Threads
	__syncthreads();

	// Get Max of Column Values
	//max << <1, squareMatrixDimension >> >(maxValueIndexMatrix, columnValues, columnIndexInMatrix, numberOfElements);

	// Update Pivot Matrix Indices and Values
	pivotMatrixCurrentColumnValue = pivotMatrix[columnIndexInMatrix];
	pivotMatrixLargestColumnValue = pivotMatrix[maxValueIndexMatrix[columnIndexInMatrix]];
	pivotMatrix[columnIndexInMatrix] = pivotMatrixLargestColumnValue;
	pivotMatrix[maxValueIndexMatrix[columnIndexInMatrix]] = pivotMatrixCurrentColumnValue;

	// Each Column Will Exchange Current Column Row Element with Pivot Element
	matrixCurrentColumnValue = matrix[(columnIndexInMatrix * squareMatrixDimension) + columnIndexInMatrix];
	matrixLargestColumnValue = matrix[(maxValueIndexMatrix[columnIndexInMatrix] * squareMatrixDimension) + columnIndexInMatrix];
	matrix[(columnIndexInMatrix * squareMatrixDimension) + columnIndexInMatrix] = matrixLargestColumnValue;
	matrix[(maxValueIndexMatrix[columnIndexInMatrix] * squareMatrixDimension) + columnIndexInMatrix] = matrixCurrentColumnValue;

#pragma unroll
	// Schur Complement
	for (int rowIndex = columnIndexInMatrix; rowIndex < squareMatrixDimension; rowIndex++)
	{
		// Divide each row element below diagonal of current column by diagonal element for current column
		matrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix] = matrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix] /
			matrix[(columnIndexInMatrix * squareMatrixDimension) + columnIndexInMatrix];

		// Subtract each row element below diagonal of current column by product of diagonal element and current row element
		matrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix] = matrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix] -
			(matrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix] * matrix[(columnIndexInMatrix * squareMatrixDimension) + columnIndexInMatrix]);
	}

	// Sync Threads
	__syncthreads();
}





float InvertMatrix(double *cpuInvertedMatrix,
	const double *cpuMatrix,
	const int numberOfElements,
	const int squareMatrixDimension)
{
	// Initialize Variables
	cudaError_t error;
	double *gpuMatrix = 0;
	int *gpuPivotMatrix = 0;
	int *gpuMaxValueIndexMatrix = 0;
	float timeToCompleteInMs = 0;
	int *cpuPivotMatrix = (int *)malloc(squareMatrixDimension * sizeof(int));
	int *cpuMaxValueIndexMatrix = (int *)malloc(squareMatrixDimension * sizeof(int));

	// Verify that Machine has GPU Installed by 
	//  selecting first GPU available.
	if (cudaSetDevice(0) != cudaSuccess)
	{
		// Print error regarding inability to select CUDA GPU Device
		fprintf(stderr, "Error: Failed to select CUDA capable GPU device!");

		// exit before executing any further
		return 0;
	}

	// Allocate GPU Memory for input matrix
	if (cudaMalloc((void**)&gpuMatrix, numberOfElements * sizeof(double)) != cudaSuccess)
	{
		// Print error regarding inability to allocate GPU Device Memory
		fprintf(stderr, "Error: Failed to allocate memory on CUDA capable GPU device!");

		// exit before executing any further
		return 0;
	}

	// Allocate GPU Memory for max value matrix
	if (cudaMalloc((void**)&gpuMaxValueIndexMatrix, numberOfElements * sizeof(int)) != cudaSuccess)
	{
		// Print error regarding inability to allocate GPU Device Memory
		fprintf(stderr, "Error: Failed to allocate memory on CUDA capable GPU device!");

		// exit before executing any further
		return 0;
	}

	// Allocate GPU Memory for input Pivot Matrix
	if (cudaMalloc((void**)&gpuPivotMatrix, squareMatrixDimension * sizeof(int)) != cudaSuccess)
	{
		// Print error regarding inability to allocate GPU Device Memory
		fprintf(stderr, "Error: Failed to allocate memory on CUDA capable GPU device!");

		// exit before executing any further
		return 0;
	}

	// Assign initial index to pivot matrix
	for (int i = 0; i < squareMatrixDimension; i++)
	{
		cpuPivotMatrix[i] = i;
		cpuMaxValueIndexMatrix[i] = 0;
	}

	// Copy Matrix Data From CPU Memory to GPU Memory
	cudaMemcpy(gpuMatrix, cpuMatrix, numberOfElements * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuPivotMatrix, cpuPivotMatrix, squareMatrixDimension * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuMaxValueIndexMatrix, cpuMaxValueIndexMatrix, numberOfElements * sizeof(int), cudaMemcpyHostToDevice);

	// Launch Kernel in context of each column in matrix for computing matrix LU Decomposition
	//  returns LU Matrix and Pivot Matrix for 
	ludecomposition<<<1, squareMatrixDimension, squareMatrixDimension * sizeof(int) >> >(gpuPivotMatrix, gpuMatrix, gpuMaxValueIndexMatrix, numberOfElements, squareMatrixDimension);

	// Need to do forward and reverse substitution to solve for inverse matrix
	//  using the LU Decomposition matrix, Pivot matrix and solving for ones array.






	// Check for any errors launching the kernel
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// Print error regarding failure during kernel launch
		fprintf(stderr, "Error: Failed to launch Matrix Inversion kernel because '%s'!\n", cudaGetErrorString(error));

		// exit before executing any further
		return 0;
	}

	// Wait for launched Matrix Inversion kernel to finish
	//  and log error if any occurred.
	error = cudaDeviceSynchronize();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Error: Device Sync returned error code %d after Matrix Inversion Kernel had been successfully launched!\n", error);

		// exit before executing any further
		return 0;
	}

	// return time required to complete matrix inversion
	return timeToCompleteInMs;
}