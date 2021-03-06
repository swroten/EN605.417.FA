
#include "matrixSolver.h"
#include "cuda_runtime.h"
#include "math_functions.h"
#include "device_launch_parameters.h"

#include <iomanip>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <tchar.h>
#include <curand.h>
#include <cusolverSp.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cublas_v2.h"
#include "cusolverDn.h"
#include "helper_cusolver.h"

using namespace std;

void PrintDeviceProperties()
{
	// Initialize Variables
	int device = 0;
	cudaDeviceProp prop;

	// Get Properties of this device
	cudaGetDeviceProperties(&prop, device);

	// Print Properties of this device
	printf("\n");
	printf("Device Number: %d\n", device);
	printf("  Device name: %s\n", prop.name);
	printf("  Warp Size: %i\n", prop.warpSize);
	printf("  Max Threads Per Block: %i\n", prop.maxThreadsPerBlock);
}

// Function to retrieving time
__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

__global__ void init(unsigned int seed, curandState_t* states, const int numberOfElements)
{
	// Initialize Variables
	const int currentThreadIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Exit if Thread out of bounds
	if (currentThreadIndex > numberOfElements) { return; }

	// Initialize Random Value
	curand_init(seed, currentThreadIndex, 0, &states[currentThreadIndex]);
}

__global__ void randoms(curandState_t* states, double* matrix, const int numberOfElements)
{
	// Initialize Variables
	const int currentThreadIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Exit if Thread out of bounds
	if (currentThreadIndex > numberOfElements) { return; }

	// Set Random Number in Thread Index - make sure non-zero to prevent singular matrix for testing
	matrix[currentThreadIndex] = max(curand(&states[currentThreadIndex]) % 100, 1);
}

__global__ void get_forward_sub(double *fowardSubstitutionArray,
										  const double *matrix,
										  const double *solveArray,
										  const int *cpuPivotMatrix,
										  const int squareMatrixDimension,
										  const int currentRowIndex)
{
	// Initialize
	extern __shared__ double lowerTriangleContributionToSumInSharedMemory[];

	// Initialize Variables
	const int numberOfThreads = blockDim.x;
	const int numberOfElements = (squareMatrixDimension);
	const int currentThreadIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Default Shared Memory Value to 0
	lowerTriangleContributionToSumInSharedMemory[currentThreadIndex] = 0;

	// Exit if Thread out of bounds
	if (currentThreadIndex > numberOfElements) { return; }

	// Initialize Variables
	const double solveArrayValueFromPivotMatrixRowIndexCrossReference = solveArray[cpuPivotMatrix[currentRowIndex]];

	// Initialize Shared Data Array Values
	lowerTriangleContributionToSumInSharedMemory[currentThreadIndex] = ((currentThreadIndex < currentRowIndex) ? (matrix[(currentRowIndex * squareMatrixDimension) + currentThreadIndex] * fowardSubstitutionArray[currentThreadIndex]) : 0);

	// Sychronize all Threads
	__syncthreads();

	// Check if Block Size is greater than 512
	if (numberOfThreads >= 512)
	{
		if (currentThreadIndex < 256)
		{
			// Update shared memory 
			lowerTriangleContributionToSumInSharedMemory[currentThreadIndex]  += lowerTriangleContributionToSumInSharedMemory[currentThreadIndex + 256];
		}

		// Sychronize all Threads
		__syncthreads();
	}

	if (numberOfThreads >= 256)
	{
		if (currentThreadIndex < 128)
		{
			// Update shared memory 
			lowerTriangleContributionToSumInSharedMemory[currentThreadIndex] += lowerTriangleContributionToSumInSharedMemory[currentThreadIndex + 128];
		}

		// Sychronize all Threads
		__syncthreads();
	}

	if (numberOfThreads >= 128)
	{
		if (currentThreadIndex < 64)
		{
			// Update shared memory 
			lowerTriangleContributionToSumInSharedMemory[currentThreadIndex] += lowerTriangleContributionToSumInSharedMemory[currentThreadIndex + 64];
		}

		// Sychronize all Threads
		__syncthreads();
	}

	if (currentThreadIndex < 32)
	{
		if (numberOfThreads >= 64)
		{
			// Update shared memory 
			lowerTriangleContributionToSumInSharedMemory[currentThreadIndex] += lowerTriangleContributionToSumInSharedMemory[currentThreadIndex + 32];
		}

		if (numberOfThreads >= 32)
		{
			// Update shared memory 
			lowerTriangleContributionToSumInSharedMemory[currentThreadIndex] += lowerTriangleContributionToSumInSharedMemory[currentThreadIndex + 16];
		}

		if (numberOfThreads >= 16)
		{
			// Update shared memory 
			lowerTriangleContributionToSumInSharedMemory[currentThreadIndex] += lowerTriangleContributionToSumInSharedMemory[currentThreadIndex + 8];
		}

		if (numberOfThreads >= 8)
		{
			// Update shared memory 
			lowerTriangleContributionToSumInSharedMemory[currentThreadIndex] += lowerTriangleContributionToSumInSharedMemory[currentThreadIndex + 4];
		}

		if (numberOfThreads >= 4)
		{
			// Update shared memory 
			lowerTriangleContributionToSumInSharedMemory[currentThreadIndex] += lowerTriangleContributionToSumInSharedMemory[currentThreadIndex + 2];
		}

		if (numberOfThreads >= 2)
		{
			// Update shared memory 
			lowerTriangleContributionToSumInSharedMemory[currentThreadIndex] += lowerTriangleContributionToSumInSharedMemory[currentThreadIndex + 1];
		}
	}

	// if this is the thread that is in the column that corresponds to the current row, update the forward substitution array
	if (currentThreadIndex == currentRowIndex)
	{
		fowardSubstitutionArray[currentRowIndex] = solveArrayValueFromPivotMatrixRowIndexCrossReference - lowerTriangleContributionToSumInSharedMemory[0];
	}
}

__global__ void get_backward_sub(double *backwardSubstitutionArray,
										   const double *fowardSubstitutionArray,
										   const double *matrix,
										   const int squareMatrixDimension,
										   const int currentRowIndex)
{
	// Initialize
	extern __shared__ double upperTriangleContributionToSumInSharedMemory[];

	// Initialize Variables
	const int numberOfThreads = blockDim.x;
	const int numberOfElements = (squareMatrixDimension);
	const int currentThreadIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Default Shared Memory Value to 0
	upperTriangleContributionToSumInSharedMemory[currentThreadIndex] = 0;

	// Exit if Thread out of bounds
	if (currentThreadIndex > numberOfElements) { return; }
	
	// Initialize Shared Data Array Values
	upperTriangleContributionToSumInSharedMemory[currentThreadIndex] = ((currentThreadIndex > currentRowIndex) ? (matrix[(currentRowIndex * squareMatrixDimension) + currentThreadIndex] * backwardSubstitutionArray[currentThreadIndex]) : 0);

	// Sychronize all Threads
	__syncthreads();

	// Check if Block Size is greater than 512
	if (numberOfThreads >= 512)
	{
		if (currentThreadIndex < 256)
		{
			// Update shared memory 
			upperTriangleContributionToSumInSharedMemory[currentThreadIndex] += upperTriangleContributionToSumInSharedMemory[currentThreadIndex + 256];
		}

		// Sychronize all Threads
		__syncthreads();
	}

	if (numberOfThreads >= 256)
	{
		if (currentThreadIndex < 128)
		{
			// Update shared memory 
			upperTriangleContributionToSumInSharedMemory[currentThreadIndex] += upperTriangleContributionToSumInSharedMemory[currentThreadIndex + 128];
		}

		// Sychronize all Threads
		__syncthreads();
	}

	if (numberOfThreads >= 128)
	{
		if (currentThreadIndex < 64)
		{
			// Update shared memory 
			upperTriangleContributionToSumInSharedMemory[currentThreadIndex] += upperTriangleContributionToSumInSharedMemory[currentThreadIndex + 64];
		}

		// Sychronize all Threads
		__syncthreads();
	}

	if (currentThreadIndex < 32)
	{
		if (numberOfThreads >= 64)
		{
			// Update shared memory 
			upperTriangleContributionToSumInSharedMemory[currentThreadIndex] += upperTriangleContributionToSumInSharedMemory[currentThreadIndex + 32];
		}

		if (numberOfThreads >= 32)
		{
			// Update shared memory 
			upperTriangleContributionToSumInSharedMemory[currentThreadIndex] += upperTriangleContributionToSumInSharedMemory[currentThreadIndex + 16];
		}

		if (numberOfThreads >= 16)
		{
			// Update shared memory 
			upperTriangleContributionToSumInSharedMemory[currentThreadIndex] += upperTriangleContributionToSumInSharedMemory[currentThreadIndex + 8];
		}

		if (numberOfThreads >= 8)
		{
			// Update shared memory 
			upperTriangleContributionToSumInSharedMemory[currentThreadIndex] += upperTriangleContributionToSumInSharedMemory[currentThreadIndex + 4];
		}

		if (numberOfThreads >= 4)
		{
			// Update shared memory 
			upperTriangleContributionToSumInSharedMemory[currentThreadIndex] += upperTriangleContributionToSumInSharedMemory[currentThreadIndex + 2];
		}

		if (numberOfThreads >= 2)
		{
			// Update shared memory 
			upperTriangleContributionToSumInSharedMemory[currentThreadIndex] += upperTriangleContributionToSumInSharedMemory[currentThreadIndex + 1];
		}
	}

	// if this is the thread that is in the column that corresponds to the current row, update the forward substitution array
	if (currentThreadIndex == 0)
	{
		backwardSubstitutionArray[currentRowIndex] = ((fowardSubstitutionArray[currentRowIndex] - upperTriangleContributionToSumInSharedMemory[0]) / matrix[(currentRowIndex * squareMatrixDimension) + currentRowIndex]);
	}
}

__global__ void get_solve(double *matrix,
								  const double *backwardSubstitutionArray,
								  const int squareMatrixDimension,
								  const int currentRowIndex)
{
	// Initialize Variables
	const int currentThreadIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int numberOfElements = (squareMatrixDimension);

	// Exit if Thread out of bounds
	if (currentThreadIndex > numberOfElements) { return; }

	// Update Inverse Matrix
	matrix[(currentThreadIndex * squareMatrixDimension) + currentRowIndex] = backwardSubstitutionArray[currentThreadIndex];
}

__global__ void get_shur_complement(double *matrix, 
											   const int currentColumnInMatrix,
											   const int squareMatrixDimension)
{
	// Initialize
	extern __shared__ double matrixInSharedMemory[];

	// Initialize Variables
	const int numberOfElements = (squareMatrixDimension * squareMatrixDimension);
	const int threadIndexWithinEntireMatrix = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Exit if Thread out of bounds
	if (threadIndexWithinEntireMatrix > numberOfElements) { return; }

	// Initialize Variables
	const int threadIndexWithinEntireMatrixAsRowIndex = (threadIndexWithinEntireMatrix / squareMatrixDimension);
	const int threadIndexWithinEntireMatrixColumnIndex = (threadIndexWithinEntireMatrix % squareMatrixDimension);
	const bool isADivisionColumn = (currentColumnInMatrix == threadIndexWithinEntireMatrixColumnIndex);
	const bool isBelowCurrentRow = (threadIndexWithinEntireMatrixAsRowIndex > currentColumnInMatrix);
	const bool isElementInSubMatrix = (isBelowCurrentRow && (threadIndexWithinEntireMatrixColumnIndex >= currentColumnInMatrix));
	const double diagonalMatrixValue = matrix[(currentColumnInMatrix * squareMatrixDimension) + currentColumnInMatrix];
	const int sameRowFixedColumnIndex = (threadIndexWithinEntireMatrixAsRowIndex * squareMatrixDimension) + currentColumnInMatrix;
	const int fixedRowSameColumnIndex = (currentColumnInMatrix * squareMatrixDimension) + threadIndexWithinEntireMatrixColumnIndex;

	// Load Into Shared Memory
	matrixInSharedMemory[threadIndexWithinEntireMatrix] = matrix[threadIndexWithinEntireMatrix];

	// Sychronize all Threads
	__syncthreads();

	// Perform Division on Elements of First Column
	matrixInSharedMemory[threadIndexWithinEntireMatrix] = (isADivisionColumn && isBelowCurrentRow) ? (matrixInSharedMemory[threadIndexWithinEntireMatrix] / diagonalMatrixValue) : matrixInSharedMemory[threadIndexWithinEntireMatrix];

	// Sychronize all Threads
	__syncthreads();

	// Perform Subtraction on Elements Not in First Column
	matrixInSharedMemory[threadIndexWithinEntireMatrix] = (!isADivisionColumn && isElementInSubMatrix) ? (matrixInSharedMemory[threadIndexWithinEntireMatrix] -
		(matrixInSharedMemory[sameRowFixedColumnIndex] * matrixInSharedMemory[fixedRowSameColumnIndex])) : matrixInSharedMemory[threadIndexWithinEntireMatrix];

	// Sychronize all Threads
	__syncthreads();

	// Update Global Memory
	matrix[threadIndexWithinEntireMatrix] = matrixInSharedMemory[threadIndexWithinEntireMatrix];
}

void GetRandomNumbersForMatrix(double *cpuMatrix, const int numberOfElements)
{
	// Initialize Variables
	int device = 0;
	int numberOfBlocks;
	cudaDeviceProp prop;
	int numberOfThreads;
	double *gpuMatrix = 0;
	curandState_t* states;
	int maxThreadsPerBlock;
	const int numberOfBytesInMatrix = numberOfElements * sizeof(double);
	
	// Get Properties of this device
	cudaGetDeviceProperties(&prop, device);

	// Get Max Threads Per Block
	maxThreadsPerBlock = prop.maxThreadsPerBlock;

	// Verify that Machine has GPU Installed by 
	//  selecting first GPU available.
	checkCudaErrors(cudaSetDevice(0));
	
	// Get Number of Blocks Required and Number of Threads
	numberOfBlocks = (int)(numberOfElements / maxThreadsPerBlock) + 1;
	numberOfThreads = maxThreadsPerBlock;

	// Allocate GPU Memory for States
	checkCudaErrors(cudaMalloc((void**)&states, numberOfElements * sizeof(curandState_t)));

	// Run Initialization
	init<<<numberOfBlocks, numberOfThreads>>>((unsigned int)time(0), states, numberOfElements);

	// Allocate GPU Memory for input matrix
	checkCudaErrors(cudaMalloc((void**)&gpuMatrix, numberOfBytesInMatrix));

	// Add Random Numbers to Matrix
	randoms<<<numberOfBlocks, numberOfThreads>>>(states, gpuMatrix, numberOfElements);
	
	// Copy Matrix Data From CPU Memory to GPU Memory
	checkCudaErrors(cudaMemcpy(cpuMatrix, gpuMatrix, numberOfBytesInMatrix, cudaMemcpyDeviceToHost));

	// Free Allocated Memory
	checkCudaErrors(cudaFree(gpuMatrix));
}

float GetInvertedMatrixCPU(double *cpuInvertedMatrix,
									const double *cpuLUMatrix,
									const int *cpuPivotMatrix,
									const int squareMatrixDimension)
{
	// Initialize Variables
	LARGE_INTEGER stopTime;
	LARGE_INTEGER frequency;
	LARGE_INTEGER startTime;
	double sumLowerTriangle = 0;
	double sumUpperTriangle = 0;
	float timeToCompleteInMs = 0;
	vector<double> solveArray(squareMatrixDimension, 0.0);
	vector<double> fowardSubstitutionArray(squareMatrixDimension, 0.0);
	vector<double> backwardSubtitutionArray(squareMatrixDimension, 0.0);

	// Get Frequency
	QueryPerformanceFrequency(&frequency);

	// Keep Track of Start Time
	QueryPerformanceCounter(&startTime);

	// Solve for the Identity Matrix using resuls of LU Decomposition
	//  Step through each row and solve
	for (int overallRowIndex = 0; overallRowIndex < squareMatrixDimension; overallRowIndex++)
	{
		// Initialize
		solveArray = vector<double>(squareMatrixDimension, 0.0);

		// Set to Identity
		solveArray[overallRowIndex] = 1;

		// Solve by doing foward substition
		for (int rowIndex = 0; rowIndex < squareMatrixDimension; rowIndex++)
		{
			// Set Accumulating sum to 0
			sumLowerTriangle = 0;

			// Step through Each Column
			for (int columnIndex = 0; columnIndex < rowIndex; columnIndex++)
			{
				// Accumulate Lower Triangle Sum
				sumLowerTriangle += (cpuLUMatrix[(rowIndex * squareMatrixDimension) + columnIndex] * fowardSubstitutionArray[columnIndex]);
			}

			// Perform Foward Substituition using Pivot Array and Accumulating Lower Triangle Sum
			fowardSubstitutionArray[rowIndex] = solveArray[cpuPivotMatrix[rowIndex]] - sumLowerTriangle;
		}

		// Solve by doing backward substition
		for (int rowIndex = squareMatrixDimension - 1; rowIndex >= 0; rowIndex--)
		{
			// Set Accumulating sum to 0
			sumUpperTriangle = 0;

			// Step through Each Column
			for (int columnIndex = rowIndex + 1; columnIndex < squareMatrixDimension; columnIndex++)
			{
				sumUpperTriangle += (cpuLUMatrix[(rowIndex * squareMatrixDimension) + columnIndex] * backwardSubtitutionArray[columnIndex]);
			}

			backwardSubtitutionArray[rowIndex] = ((fowardSubstitutionArray[rowIndex] - sumUpperTriangle) /
				cpuLUMatrix[(rowIndex * squareMatrixDimension) + rowIndex]);
		}

		// Perform final update to get Inverted Matrix
		for (int overallColumnIndex = 0; overallColumnIndex < squareMatrixDimension; overallColumnIndex++)
		{
			// Update Inverse Matrix
			cpuInvertedMatrix[(overallColumnIndex * squareMatrixDimension) + overallRowIndex] = backwardSubtitutionArray[overallColumnIndex];
		}
	}

	// Keep Track of Stop Time 
	QueryPerformanceCounter(&stopTime);

	// Get Total Time to Complete
	timeToCompleteInMs = (float)((stopTime.QuadPart - startTime.QuadPart) * 1000.0 / frequency.QuadPart);

	// Return time required to complete
	return timeToCompleteInMs;
}


float GetLUDecompositionMatrixCPU(double *cpuInvertedMatrix,
											 int *cpuPivotMatrix,
											 const double *cpuMatrix,
											 const int numberOfElements,
											 const int squareMatrixDimension)
{
	// Initialize Variables
	int maxValueIndex = 0;
	LARGE_INTEGER stopTime;
	LARGE_INTEGER frequency;
	LARGE_INTEGER startTime;
	double largestValue = 0.0;
	float timeToCompleteInMs = 0;
	double matrixCurrentColumnValue = 0;
	double matrixLargestColumnValue = 0;
	int pivotMatrixCurrentColumnValue = 0;
	int pivotMatrixLargestColumnValue = 0;
	int *cpuMaxValueIndex = (int *)malloc(sizeof(int));

	// Copy Initial Matrix into Inverted Matrix
	cudaMemcpy(cpuInvertedMatrix, cpuMatrix, numberOfElements * sizeof(double), cudaMemcpyHostToHost);

	// Get Frequency
	QueryPerformanceFrequency(&frequency);

	// Keep Track of Start Time
	QueryPerformanceCounter(&startTime);

	// for each column in matrix
	for (int columnIndexInMatrix = 0; columnIndexInMatrix < squareMatrixDimension; columnIndexInMatrix++)
	{
		// Get all row elements in current column 
		largestValue = 0;
		for (int rowIndex = columnIndexInMatrix; rowIndex < squareMatrixDimension; rowIndex++)
		{
			// Get Max Row Value in Column
			if (abs(cpuInvertedMatrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix]) > largestValue)
			{
				largestValue = abs(cpuInvertedMatrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix]);
				maxValueIndex = rowIndex;
			}
		}

		// Update Pivot Matrix Indices and Values
		pivotMatrixCurrentColumnValue = cpuPivotMatrix[columnIndexInMatrix];
		pivotMatrixLargestColumnValue = cpuPivotMatrix[maxValueIndex];
		cpuPivotMatrix[columnIndexInMatrix] = pivotMatrixLargestColumnValue;
		cpuPivotMatrix[maxValueIndex] = pivotMatrixCurrentColumnValue;

		// Each Column Will Exchange Current Column Row Element with Pivot Element
		for (int columnIndex = 0; columnIndex < squareMatrixDimension; columnIndex++)
		{
			matrixCurrentColumnValue = cpuInvertedMatrix[(columnIndexInMatrix * squareMatrixDimension) + columnIndex];
			matrixLargestColumnValue = cpuInvertedMatrix[(maxValueIndex * squareMatrixDimension) + columnIndex];
			cpuInvertedMatrix[(columnIndexInMatrix * squareMatrixDimension) + columnIndex] = matrixLargestColumnValue;
			cpuInvertedMatrix[(maxValueIndex * squareMatrixDimension) + columnIndex] = matrixCurrentColumnValue;
		}

		// Perform Shurs Complement
		for (int rowIndex = columnIndexInMatrix + 1; rowIndex < squareMatrixDimension; rowIndex++)
		{
			cpuInvertedMatrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix] /= cpuInvertedMatrix[(columnIndexInMatrix * squareMatrixDimension) + columnIndexInMatrix];

			for (int columnIndex = columnIndexInMatrix + 1; columnIndex < squareMatrixDimension; columnIndex++)
			{
				cpuInvertedMatrix[(rowIndex * squareMatrixDimension) + columnIndex] -= (cpuInvertedMatrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix] *
					cpuInvertedMatrix[(columnIndexInMatrix * squareMatrixDimension) + columnIndex]);
			}
		}
	}		

	// Keep Track of Stop Time 
	QueryPerformanceCounter(&stopTime);

	// Get Total Time to Complete
	timeToCompleteInMs = (float)((stopTime.QuadPart - startTime.QuadPart) * 1000.0 / frequency.QuadPart);

	// Return time required to complete
	return timeToCompleteInMs;
}

float GetInvertedMatrixGPU(double *cpuInvertedMatrix,
									const double *cpuLUMatrix,
									const int *cpuPivotMatrix,
									const int numberOfElements,
									const int squareMatrixDimension)
{
	// Initialize Variables
	int device = 0;
	cudaEvent_t stop;
	cudaEvent_t start;
	int numberOfBlocks;
	int numberOfThreads;
	cudaDeviceProp prop;
	vector<double> doubles;
	int maxThreadsPerBlock;
	int *gpuPivotMatrix = 0;
	double *gpuLUMatrix = 0;
	double *gpuSolveArray = 0;
	float timeToCompleteInMs = 0;
	double *gpuInvertedMatrix = 0;
	double *gpuForwardSubstitutionArray = 0;
	double *gpuBackwardSubstitutionArray = 0;
	vector<double> fowardSubstitutionVector(squareMatrixDimension, 0.0);
	vector<double> backwardSubtitutionVector(squareMatrixDimension, 0.0);
	const int numberOfBytesInMatrix = numberOfElements * sizeof(double);
	const int numberOfBytesInArray = squareMatrixDimension * sizeof(int);
	double *cpuSolveArray = (double *)malloc(squareMatrixDimension * sizeof(double));
	double *cpuForwardSubstitutionArray = (double *)malloc(squareMatrixDimension * sizeof(double));
	double *cpuBackwardSubstitutionArray = (double *)malloc(squareMatrixDimension * sizeof(double));
	
	// Get Properties of this device
	cudaGetDeviceProperties(&prop, device);

	// Get Max Threads Per Block
	maxThreadsPerBlock = prop.maxThreadsPerBlock;

	// Verify that Machine has GPU Installed by 
	//  selecting first GPU available.
	checkCudaErrors(cudaSetDevice(0));

	// Allocate GPU Memory for input matrix
	checkCudaErrors(cudaMalloc((void**)&gpuLUMatrix, numberOfBytesInMatrix));

	// Allocate GPU Memory for input GPU Inverted
	checkCudaErrors(cudaMalloc((void**)&gpuInvertedMatrix, numberOfBytesInMatrix));
	
	// Allocate GPU Memory for input Pivot Matrix
	checkCudaErrors(cudaMalloc((void**)&gpuPivotMatrix, numberOfBytesInArray));

	// Allocate GPU Memory for input Solve Array
	checkCudaErrors(cudaMalloc((void**)&gpuSolveArray, squareMatrixDimension * sizeof(double)));

	// Allocate GPU Memory for input Forward Sub Array
	checkCudaErrors(cudaMalloc((void**)&gpuForwardSubstitutionArray, squareMatrixDimension * sizeof(double)));

	// Allocate GPU Memory for input Backward Sub Array
	checkCudaErrors(cudaMalloc((void**)&gpuBackwardSubstitutionArray, squareMatrixDimension * sizeof(double)));

	// Get Number of Blocks Required and Number of Threads
	numberOfBlocks = (int)(squareMatrixDimension / maxThreadsPerBlock) + 1;
	numberOfThreads = maxThreadsPerBlock;
	
	// Keep Track of Start Time
	start = get_time();

	// Initialize to 0 values
	memset(cpuForwardSubstitutionArray, 0, squareMatrixDimension * sizeof(double));
	memset(cpuBackwardSubstitutionArray, 0, squareMatrixDimension * sizeof(double));

	// Copy CPU Array to GPU Array
	checkCudaErrors(cudaMemcpy(gpuLUMatrix, cpuLUMatrix, numberOfBytesInMatrix, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpuPivotMatrix, cpuPivotMatrix, numberOfBytesInArray, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpuForwardSubstitutionArray, cpuForwardSubstitutionArray, squareMatrixDimension * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpuBackwardSubstitutionArray, cpuBackwardSubstitutionArray, squareMatrixDimension * sizeof(double), cudaMemcpyHostToDevice));

	// Solve for the Identity Matrix using resuls of LU Decomposition Step through each row and solve
	for (int overallRowIndex = 0; overallRowIndex < squareMatrixDimension; overallRowIndex++)
	{
		// Initialize
		for (int index = 0; index < squareMatrixDimension; index++)
		{
			// Set to Identity
			cpuSolveArray[index] = ((index != overallRowIndex) ? 0 : 1);
		}
		
		// Copy Matrix Data From CPU Memory to GPU Memory
		checkCudaErrors(cudaMemcpy(gpuSolveArray, cpuSolveArray, squareMatrixDimension * sizeof(double), cudaMemcpyHostToDevice));

		// Solve by doing foward substition
		for (int rowIndex = 0; rowIndex < squareMatrixDimension; rowIndex++)
		{
			// Launch a Forward Substitution Kernel for each Column in this row
			get_forward_sub<<<numberOfBlocks, numberOfThreads, numberOfBlocks*numberOfThreads * sizeof(double)>>>(gpuForwardSubstitutionArray,
																																				   gpuLUMatrix,
																																				   gpuSolveArray,
																																				   gpuPivotMatrix,
																																				   squareMatrixDimension,
																																				   rowIndex);
			
			// Wait for launched Matrix Inversion kernel to finish  and log error if any occurred.
			checkCudaErrors(cudaDeviceSynchronize());
		}
		
		// Solve by doing backward substition
		for (int rowIndex = squareMatrixDimension - 1; rowIndex >= 0; rowIndex--)
		{
			// Launch a Backward Substitution Kernel for each Column in this row
			get_backward_sub<<<numberOfBlocks, numberOfThreads, numberOfBlocks*numberOfThreads * sizeof(double) >>>(gpuBackwardSubstitutionArray,
																																				 gpuForwardSubstitutionArray,
																																				 gpuLUMatrix,
																																				 squareMatrixDimension,
																																				 rowIndex);
			
			// Wait for launched Matrix Inversion kernel to finish and log error if any occurred.
			checkCudaErrors(cudaDeviceSynchronize());
		}
		
		// Launch Kernel for Final Solving of Inverted Matrix
		get_solve<<<numberOfBlocks, numberOfThreads>>>(gpuInvertedMatrix, 
																	  gpuBackwardSubstitutionArray, 
																	  squareMatrixDimension, 
																	  overallRowIndex);
		// Wait for launched Matrix Inversion kernel to finish  and log error if any occurred.
		checkCudaErrors(cudaDeviceSynchronize());
		
		// Copy Matrix Data From CPU Memory to GPU Memory
		checkCudaErrors(cudaMemcpy(cpuInvertedMatrix, gpuInvertedMatrix, numberOfBytesInMatrix, cudaMemcpyDeviceToHost));
	}

	// Keep Track of Stop Time 
	stop = get_time();

	// Synchronize Events
	timeToCompleteInMs = 0;
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&timeToCompleteInMs, start, stop));

	// Deallocate Memory
	cudaFreeHost(cpuSolveArray);
	cudaFreeHost(cpuForwardSubstitutionArray);
	cudaFreeHost(cpuBackwardSubstitutionArray);
	checkCudaErrors(cudaFree(gpuPivotMatrix));
	checkCudaErrors(cudaFree(gpuLUMatrix));
	checkCudaErrors(cudaFree(gpuSolveArray));
	checkCudaErrors(cudaFree(gpuInvertedMatrix));
	checkCudaErrors(cudaFree(gpuForwardSubstitutionArray));
	checkCudaErrors(cudaFree(gpuBackwardSubstitutionArray));

	// Return time required to complete
	return timeToCompleteInMs;
}

float GetLUDecompositionMatrixGPU(double *cpuInvertedMatrix,
											 int *cpuPivotMatrix,
											 const double *cpuMatrix,
											 const int numberOfElements,
											 const int squareMatrixDimension)
{
	// Initialize Variables
	int device = 0;
	cudaEvent_t stop;
	cudaEvent_t start;
	cudaDeviceProp prop;
	int maxValueIndex = 0;
	int maxThreadsPerBlock;
	double *gpuLUMatrix = 0;
	int *gpuPivotMatrix = 0;
	double largestValue = 0.0;
	int *gpuMaxValueIndex = 0;
	float timeToCompleteInMs = 0;
	double matrixCurrentColumnValue = 0;
	double matrixLargestColumnValue = 0;
	int pivotMatrixCurrentColumnValue = 0;
	int pivotMatrixLargestColumnValue = 0;
	const int numberOfBytesInMatrix = numberOfElements * sizeof(double);
	const int numberOfBytesInArray = squareMatrixDimension * sizeof(int);

	// Copy Initial Matrix into Inverted Matrix
	cudaMemcpy(cpuInvertedMatrix, cpuMatrix, numberOfBytesInMatrix, cudaMemcpyHostToHost);
	
	// Get Properties of this device
	cudaGetDeviceProperties(&prop, device);

	// Get Max Threads Per Block
	maxThreadsPerBlock = prop.maxThreadsPerBlock;

	// Verify that Machine has GPU Installed by 
	//  selecting first GPU available.
	checkCudaErrors(cudaSetDevice(0));
	
	// Allocate GPU Memory for input matrix
	checkCudaErrors(cudaMalloc((void**)&gpuLUMatrix, numberOfBytesInMatrix));
	
	// Allocate GPU Memory for Max Value Index
	checkCudaErrors(cudaMalloc((void**)&gpuMaxValueIndex, sizeof(int)));
	
	// Allocate GPU Memory for input Pivot Matrix
	checkCudaErrors(cudaMalloc((void**)&gpuPivotMatrix, numberOfBytesInArray));
	
	// Keep Track of Start Time
	start = get_time();
	
	// for each column in matrix
	for (int columnIndexInMatrix = 0; columnIndexInMatrix < squareMatrixDimension; columnIndexInMatrix++)
	{
		//// Get Maximum in this Column
		//// TODO: Make work for more than 1 block
		//max<<<1, squareMatrixDimension, squareMatrixDimension * sizeof(double)>>>(gpuMaxValueIndex, gpuInvertedMatrix, columnIndexInMatrix, squareMatrixDimension);

		//// Copy result of operation from GPU to CPU Memory
		//cudaMemcpy(cpuMaxValueIndex, gpuMaxValueIndex, sizeof(int), cudaMemcpyDeviceToHost);

		// Get all row elements in current column 
		largestValue = 0;
		for (int rowIndex = columnIndexInMatrix; rowIndex < squareMatrixDimension; rowIndex++)
		{
			// Get Max Row Value in Column
			if (abs(cpuInvertedMatrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix]) > largestValue)
			{
				largestValue = abs(cpuInvertedMatrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix]);
				maxValueIndex = rowIndex;
			}
		}
		
		// Update Pivot Matrix Indices and Values
		pivotMatrixCurrentColumnValue = cpuPivotMatrix[columnIndexInMatrix];
		pivotMatrixLargestColumnValue = cpuPivotMatrix[maxValueIndex];
		cpuPivotMatrix[columnIndexInMatrix] = pivotMatrixLargestColumnValue;
		cpuPivotMatrix[maxValueIndex] = pivotMatrixCurrentColumnValue;

		// Each Column Will Exchange Current Column Row Element with Pivot Element
		for (int columnIndex = 0; columnIndex < squareMatrixDimension; columnIndex++)
		{
			// Get Matrix Values
			matrixCurrentColumnValue = cpuInvertedMatrix[(columnIndexInMatrix * squareMatrixDimension) + columnIndex];
			matrixLargestColumnValue = cpuInvertedMatrix[(maxValueIndex * squareMatrixDimension) + columnIndex];

			// Perform update on Real Matrix
			cpuInvertedMatrix[(columnIndexInMatrix * squareMatrixDimension) + columnIndex] = matrixLargestColumnValue;
			cpuInvertedMatrix[(maxValueIndex * squareMatrixDimension) + columnIndex] = matrixCurrentColumnValue;
		}

		// Copy Matrix Data From CPU Memory to GPU Memory
		checkCudaErrors(cudaMemcpy(gpuLUMatrix, cpuInvertedMatrix, numberOfBytesInMatrix, cudaMemcpyHostToDevice));

		// Get Number of Threads Required for Shurs Complement
		int numberOfBlocks = (int)(numberOfElements / maxThreadsPerBlock) + 1;      // Offset of 1, not 0
		int numberOfThreads = maxThreadsPerBlock;

		// Perform Shur Complement
		get_shur_complement<<<numberOfBlocks, numberOfThreads, numberOfBlocks*numberOfThreads*sizeof(double)>>>(gpuLUMatrix, columnIndexInMatrix, squareMatrixDimension);

		// Copy result of operation from GPU to CPU Memory
		checkCudaErrors(cudaMemcpy(cpuInvertedMatrix, gpuLUMatrix, numberOfBytesInMatrix, cudaMemcpyDeviceToHost));
	}

	// Keep Track of Stop Time 
	stop = get_time();

	// Synchronize Events
	timeToCompleteInMs = 0;
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&timeToCompleteInMs, start, stop));
	
	// Wait for launched Matrix Inversion kernel to finish and log error if any occurred.
	checkCudaErrors(cudaDeviceSynchronize());
	
	// Deallocate Memory
	checkCudaErrors(cudaFree(gpuLUMatrix));
	checkCudaErrors(cudaFree(gpuPivotMatrix));
	checkCudaErrors(cudaFree(gpuMaxValueIndex));

	// return time required to complete matrix inversion
	return timeToCompleteInMs;
}

float GetCuSparseInvertedMatrixGPU(double *cpuInvertedMatrix, 
											  const double *cpuMatrix, 
											  const int squareMatrixDimension)
{
	// Initialize Variables
	int batchSize = 1;
	int *info = NULL;
	cublasHandle_t handle;
	LARGE_INTEGER stopTime;
	LARGE_INTEGER frequency;
	LARGE_INTEGER startTime;
	int *gpuPivotMatrix = NULL;
	float timeToCompleteInMs = 0;
	double *gpuInvertedMatrix = NULL;
	double *gpuLUDecompositionMatrix = NULL;
	int numberOfElements = squareMatrixDimension*squareMatrixDimension;

	// Get Frequency
	QueryPerformanceFrequency(&frequency);

	// Keep Track of Start Time
	QueryPerformanceCounter(&startTime);

	// Allocate Device Memory
	checkCudaErrors(cudaMalloc((void **)&gpuPivotMatrix, sizeof(int)*squareMatrixDimension));
	checkCudaErrors(cudaMalloc((void **)&gpuLUDecompositionMatrix, sizeof(double)*numberOfElements));
	checkCudaErrors(cudaMalloc((void **)&gpuInvertedMatrix, sizeof(double)*numberOfElements));

	// Copy Data from CPU to GPU
	checkCudaErrors(cudaMemcpy(gpuLUDecompositionMatrix, cpuMatrix, sizeof(double)*numberOfElements, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpuInvertedMatrix, cpuInvertedMatrix, sizeof(double)*numberOfElements, cudaMemcpyHostToDevice));

	// Initialize More Variables
	double **gpuInvertedMatrixArrayOfPointers = NULL;
	double **gpuLUDecompositionMatrixArrayOfPointers = NULL;
	double *cpuInvertedMatrixArray[] = { gpuInvertedMatrix };
	double *cpuLUDecompositionMatrixArray[] = { gpuLUDecompositionMatrix };

	// Create Handle
	cublasCreate_v2(&handle);

	// Allocate Memory to device arrays
	checkCudaErrors(cudaMalloc<double*>(&gpuInvertedMatrixArrayOfPointers, sizeof(cpuInvertedMatrixArray)));
	checkCudaErrors(cudaMalloc<double*>(&gpuLUDecompositionMatrixArrayOfPointers, sizeof(cpuLUDecompositionMatrixArray)));

	// Copy Data from CPU to GPU
	checkCudaErrors(cudaMemcpy(gpuInvertedMatrixArrayOfPointers, cpuInvertedMatrixArray, sizeof(cpuInvertedMatrixArray), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpuLUDecompositionMatrixArrayOfPointers, cpuLUDecompositionMatrixArray, sizeof(cpuLUDecompositionMatrixArray), cudaMemcpyHostToDevice));
	
	// Create Buffer
	checkCudaErrors(cudaMalloc((void **)&info, sizeof(int)));

	// Initiailize Memory for Info
	checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

	// Perform LU Decomposition
	cublasStatus_t luDecompResult = cublasDgetrfBatched(handle, squareMatrixDimension, gpuLUDecompositionMatrixArrayOfPointers, squareMatrixDimension, gpuPivotMatrix, info, batchSize);

	// Compute Matrix Inverse
	cublasStatus_t invertResult = cublasDgetriBatched(handle, squareMatrixDimension, (const double **)gpuLUDecompositionMatrixArrayOfPointers, squareMatrixDimension, gpuPivotMatrix, gpuInvertedMatrixArrayOfPointers, squareMatrixDimension, info, batchSize);
	cudaDeviceSynchronize();

	// Copy results from GPU Memory to Host Memory
	checkCudaErrors(cudaMemcpy(cpuInvertedMatrix, gpuInvertedMatrix, sizeof(double)*numberOfElements, cudaMemcpyDeviceToHost));
	
	// Free up allocated memory
	if (handle) { checkCudaErrors(cublasDestroy_v2(handle)); }
	if (gpuPivotMatrix) { checkCudaErrors(cudaFree(gpuPivotMatrix)); }
	if (gpuInvertedMatrix) { checkCudaErrors(cudaFree(gpuInvertedMatrix)); }
	if (gpuInvertedMatrixArrayOfPointers) { checkCudaErrors(cudaFree(gpuInvertedMatrixArrayOfPointers)); }
	if (gpuLUDecompositionMatrixArrayOfPointers) { checkCudaErrors(cudaFree(gpuLUDecompositionMatrixArrayOfPointers)); }

	// Keep Track of Stop Time 
	QueryPerformanceCounter(&stopTime);

	// Get Total Time to Complete
	timeToCompleteInMs = (float)((stopTime.QuadPart - startTime.QuadPart) * 1000.0 / frequency.QuadPart);

	// Return time required to complete
	return timeToCompleteInMs;
}

double ComputeMagnitudeOfMatrix(const double *cpuInvertedMatrix, const int numberOfElements)
{
	// Initialize Variables
	double magnitudeOfMatrix = 0.0;
	thrust::host_vector<double> cpuInputMatrix(cpuInvertedMatrix, cpuInvertedMatrix + numberOfElements);
	
	// Square all matrix values using Thrust transform function
	thrust::transform(cpuInputMatrix.begin(), cpuInputMatrix.end(), cpuInputMatrix.begin(), cpuInputMatrix.begin(), thrust::multiplies<double>());

	// Sum the transformed matrix of squared values using Thrust reduce function
	magnitudeOfMatrix = std::sqrt(thrust::reduce(cpuInputMatrix.begin(), cpuInputMatrix.end(), (double) 0.0, thrust::plus<double>()));

	// return computed magnitude
	return magnitudeOfMatrix;
}

double GetMagnitudeOfMatrixWithSpecifiedPrecision(const double magnitude, const int precision)
{
	// Initialize Variable
	std::ostringstream magnitudeWithSpecifiedPrecision;

	// Get specified precision
	magnitudeWithSpecifiedPrecision << setprecision(precision) << fixed << magnitude;

	// return result as double
	return std::stod(magnitudeWithSpecifiedPrecision.str());
}
