
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <sstream>
#include <stdio.h>
#include <tchar.h>
#include <cusolverSp.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "matrixSolver.h"

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

__global__ void ludecomposition(int *pivotMatrix,
	double *matrix,
	const int numberOfElements,
	const int squareMatrixDimension)
{
	// Get Thread Index of Element
	const int columnIndexInMatrix = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Create Shared Memory Array
	extern volatile __shared__ int maxColumnIndices[];

	// Exit if thread index is greater than number of elements
	if (columnIndexInMatrix >= numberOfElements) { return; }

	// Initialize Column Values in context of this column
	double largestValue = 0;
	double matrixCurrentColumnValue = 0;
	double matrixLargestColumnValue = 0;
	int pivotMatrixCurrentColumnValue = 0;
	int pivotMatrixLargestColumnValue = 0;

	// Get all row elements in current column 
	for (int rowIndex = 0; rowIndex < squareMatrixDimension; rowIndex++)
	{			
		// Get Max Row Value in Column
		if (matrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix] > largestValue)
		{
			largestValue = matrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix];
			maxColumnIndices[columnIndexInMatrix] = rowIndex;
		}
	}

	// Sync Threads
	__syncthreads();
	
	// Update Pivot Matrix Indices and Values
	pivotMatrixCurrentColumnValue = pivotMatrix[columnIndexInMatrix];
	pivotMatrixLargestColumnValue = pivotMatrix[maxColumnIndices[columnIndexInMatrix]];
	pivotMatrix[columnIndexInMatrix] = pivotMatrixLargestColumnValue;
	pivotMatrix[maxColumnIndices[columnIndexInMatrix]] = pivotMatrixCurrentColumnValue;

	// Each Column Will Exchange Current Column Row Element with Pivot Element
	matrixCurrentColumnValue = matrix[(columnIndexInMatrix * squareMatrixDimension) + columnIndexInMatrix];
	matrixLargestColumnValue = matrix[(maxColumnIndices[columnIndexInMatrix] * squareMatrixDimension) + columnIndexInMatrix];
	matrix[(columnIndexInMatrix * squareMatrixDimension) + columnIndexInMatrix] = matrixLargestColumnValue;
	matrix[(maxColumnIndices[columnIndexInMatrix] * squareMatrixDimension) + columnIndexInMatrix] = matrixCurrentColumnValue;

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
	float timeToCompleteInMs = 0;
	double *gpuInvertedMatrix = 0;
	int *cpuPivotMatrix = (int *)malloc(squareMatrixDimension * sizeof(int));

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

	// Allocate GPU Memory for input matrix
	if (cudaMalloc((void**)&gpuInvertedMatrix, numberOfElements * sizeof(double)) != cudaSuccess)
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
	}

	// Copy Matrix Data From CPU Memory to GPU Memory
	cudaMemcpy(gpuMatrix, cpuMatrix, numberOfElements * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuPivotMatrix, cpuPivotMatrix, squareMatrixDimension * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuInvertedMatrix, cpuInvertedMatrix, numberOfElements * sizeof(double), cudaMemcpyHostToDevice);

	// Launch Kernel in context of each column in matrix for computing matrix LU Decomposition
	//  returns LU Matrix and Pivot Matrix for 
	ludecomposition<<<1, squareMatrixDimension, squareMatrixDimension * sizeof(int)>>>(gpuPivotMatrix, gpuMatrix, numberOfElements, squareMatrixDimension);

	// Need to do forward and reverse substitution to solve for inverse matrix
	//  using the LU Decomposition matrix, Pivot matrix and solving for ones array.
	
	// Copy result of operation from GPU to CPU Memory
	cudaMemcpy(cpuInvertedMatrix, gpuMatrix, numberOfElements * sizeof(double), cudaMemcpyDeviceToHost);
	
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
