
#include "matrixSolver.h"
#include "cuda_runtime.h"
#include "math_functions.h"
#include "device_launch_parameters.h"

#include <sstream>
#include <vector>
#include <stdio.h>
#include <tchar.h>
#include <curand.h>
#include <cusolverSp.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

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

__global__ void get_max(int *maxValueIndex,
							   const double *matrix,
							   const int columnIndexInMatrix,
							   const int squareMatrixDimension)
{
	// Create Shared Memory Array
	extern volatile __shared__ double sharedDataArray[];
	extern volatile __shared__ int sharedDataArrayIndices[];

	// Initialize Shared Data Array Values
	sharedDataArray[threadIdx.x] = matrix[(threadIdx.x * squareMatrixDimension) + columnIndexInMatrix];

	// Sychronize all Threads
	__syncthreads();

	// Check if Block Size is greater than 512
	if (blockDim.x >= 512)
	{
		if (threadIdx.x < 256)
		{
			// Update shared memory 
			sharedDataArrayIndices[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 256])) ?
				threadIdx.x : threadIdx.x + 256;
			sharedDataArray[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 256])) ?
				abs(sharedDataArray[threadIdx.x]) : abs(sharedDataArray[threadIdx.x + 256]);
		}

		// Sychronize all Threads
		__syncthreads();
	}

	if (blockDim.x >= 256)
	{
		if (threadIdx.x < 128)
		{
			// Update shared memory 
			sharedDataArrayIndices[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 128])) ?
				threadIdx.x : threadIdx.x + 128;
			sharedDataArray[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 128])) ?
				abs(sharedDataArray[threadIdx.x]) : abs(sharedDataArray[threadIdx.x + 128]);
		}

		// Sychronize all Threads
		__syncthreads();
	}

	if (blockDim.x >= 128)
	{
		if (threadIdx.x < 64)
		{
			// Update shared memory 
			sharedDataArrayIndices[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 64])) ?
				threadIdx.x : threadIdx.x + 64;
			sharedDataArray[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 64])) ?
				abs(sharedDataArray[threadIdx.x]) : abs(sharedDataArray[threadIdx.x + 64]);
		}

		// Sychronize all Threads
		__syncthreads();
	}

	if (threadIdx.x < 32)
	{
		if (blockDim.x >= 64)
		{
			// Update shared memory 
			sharedDataArrayIndices[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 32])) ?
				threadIdx.x : threadIdx.x + 32;
			sharedDataArray[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 32])) ?
				abs(sharedDataArray[threadIdx.x]) : abs(sharedDataArray[threadIdx.x + 32]);
		}

		if (blockDim.x >= 32)
		{
			// Update shared memory 
			sharedDataArrayIndices[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 16])) ?
				threadIdx.x : threadIdx.x + 16;
			sharedDataArray[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 16])) ?
				abs(sharedDataArray[threadIdx.x]) : abs(sharedDataArray[threadIdx.x + 16]);
		}

		if (blockDim.x >= 16)
		{
			// Update shared memory 
			sharedDataArrayIndices[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 8])) ?
				threadIdx.x : threadIdx.x + 8;
			sharedDataArray[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 8])) ?
				abs(sharedDataArray[threadIdx.x]) : abs(sharedDataArray[threadIdx.x + 8]);
		}

		if (blockDim.x >= 8)
		{
			// Update shared memory 
			sharedDataArrayIndices[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 4])) ?
				threadIdx.x : threadIdx.x + 4;
			sharedDataArray[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 4])) ?
				abs(sharedDataArray[threadIdx.x]) : abs(sharedDataArray[threadIdx.x + 4]);
		}

		if (blockDim.x >= 4)
		{
			// Update shared memory 
			sharedDataArrayIndices[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 2])) ?
				threadIdx.x : threadIdx.x + 2;
			sharedDataArray[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 2])) ?
				abs(sharedDataArray[threadIdx.x]) : abs(sharedDataArray[threadIdx.x + 2]);
		}

		if (blockDim.x >= 2)
		{
			// Update shared memory 
			sharedDataArrayIndices[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 1])) ?
				threadIdx.x : threadIdx.x + 1;
			sharedDataArray[threadIdx.x] = (abs(sharedDataArray[threadIdx.x]) > abs(sharedDataArray[threadIdx.x + 1])) ?
				abs(sharedDataArray[threadIdx.x]) : abs(sharedDataArray[threadIdx.x + 1]);
		}
	}

	if (threadIdx.x == 0)
	{
		maxValueIndex[0] = sharedDataArrayIndices[0];
	}
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

__global__ void get_shur_complement_mod(double *matrix,
													 const int currentColumnInMatrix,
													 const int squareMatrixDimension)
{
	// Get Constant Values
	const double diagonalMatrixValue = matrix[(currentColumnInMatrix * squareMatrixDimension) + currentColumnInMatrix];

	// Initialize Variables
	const int threadIndexWithinSubMatrix = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int columnsInSubMatrix = (squareMatrixDimension - currentColumnInMatrix);
	const int rowIndexInSubMatrix = (int)(threadIndexWithinSubMatrix / columnsInSubMatrix);
	const int threadIndexWithinEntireMatrix = ((currentColumnInMatrix + 1) * squareMatrixDimension) + currentColumnInMatrix + threadIndexWithinSubMatrix + (rowIndexInSubMatrix * (squareMatrixDimension - columnsInSubMatrix));
	const int threadIndexWithinEntireMatrixAsRowIndex = (threadIndexWithinEntireMatrix / squareMatrixDimension);
	const int threadIndexWithinEntireMatrixColumnIndex = (threadIndexWithinEntireMatrix % squareMatrixDimension);
	const int sameRowFixedColumnIndex = (threadIndexWithinEntireMatrixAsRowIndex * squareMatrixDimension) + currentColumnInMatrix;
	const int fixedRowSameColumnIndex = (currentColumnInMatrix * squareMatrixDimension) + threadIndexWithinEntireMatrixColumnIndex;
	const bool isADivisionColumn = (currentColumnInMatrix == threadIndexWithinEntireMatrixColumnIndex);

	// Perform Division on Elements of First Column
	matrix[threadIndexWithinEntireMatrix] = isADivisionColumn ? (matrix[threadIndexWithinEntireMatrix] / diagonalMatrixValue) : matrix[threadIndexWithinEntireMatrix];

	// Sychronize all Threads
	__syncthreads();

	// Perform Subtraction on Elements Not in First Column
	matrix[threadIndexWithinEntireMatrix] = !isADivisionColumn ? (matrix[threadIndexWithinEntireMatrix] -
		(matrix[sameRowFixedColumnIndex] * matrix[fixedRowSameColumnIndex])) : matrix[threadIndexWithinEntireMatrix];

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
	cudaEvent_t stop;
	cudaEvent_t start;
	double sumLowerTriangle = 0;
	double sumUpperTriangle = 0;
	float timeToCompleteInMs = 0;
	vector<double> solveArray(squareMatrixDimension, 0.0);
	vector<double> fowardSubstitutionArray(squareMatrixDimension, 0.0);
	vector<double> backwardSubtitutionArray(squareMatrixDimension, 0.0);

	// Keep Track of Start Time
	start = get_time();

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
	stop = get_time();

	// Synchronize Events
	timeToCompleteInMs = 0;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeToCompleteInMs, start, stop);

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
	cudaEvent_t stop;
	cudaEvent_t start;
	int maxValueIndex = 0;
	double largestValue = 0.0;
	float timeToCompleteInMs = 0;
	double matrixCurrentColumnValue = 0;
	double matrixLargestColumnValue = 0;
	int pivotMatrixCurrentColumnValue = 0;
	int pivotMatrixLargestColumnValue = 0;
	int *cpuMaxValueIndex = (int *)malloc(sizeof(int));

	// Copy Initial Matrix into Inverted Matrix
	cudaMemcpy(cpuInvertedMatrix, cpuMatrix, numberOfElements * sizeof(double), cudaMemcpyHostToHost);

	// Keep Track of Start Time
	start = get_time();

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
	stop = get_time();

	// Synchronize Events
	timeToCompleteInMs = 0;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeToCompleteInMs, start, stop);

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
		//int numberOfThreads = get_max((int)(numberOfElements % maxThreadsPerBlock), 1); // Ensure at Least 1 Thread when executing
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
	cudaEvent_t stop;
	cudaEvent_t start;
	int bufferSize = 0;
	cudaStream_t stream = NULL;
	float timeToCompleteInMs = 0;
	cusolverDnHandle_t handle = NULL;
	double *cpuResultMatrix = NULL; // a copy of d_x
	double *cpuSolveMatrix = NULL; // b = ones(m,1)
	double *gpuMatrix = NULL; // a copy of h_A
	double *gpuResultMatrix = NULL; // x = A \ b
	double *gpuSolveMatrix = NULL; // a copy of h_b
	int h_info = 0;
	int *info = NULL;
	double *buffer = NULL;
	int *gpuPivotMatrix = NULL; // pivoting sequence

	// Allocate Memory to Host
	cpuResultMatrix = (double*)malloc(sizeof(double)*squareMatrixDimension);
	cpuSolveMatrix = (double*)malloc(sizeof(double)*squareMatrixDimension);
	
	// Create Handle
	checkCudaErrors(cudaStreamCreate(&stream));
	checkCudaErrors(cusolverDnCreate(&handle));
	checkCudaErrors(cusolverDnSetStream(handle, stream));

	// Allocate Memory to device arrays
	checkCudaErrors(cudaMalloc((void **)&gpuMatrix, sizeof(double)*squareMatrixDimension*squareMatrixDimension));
	checkCudaErrors(cudaMalloc((void **)&gpuResultMatrix, sizeof(double)*squareMatrixDimension));
	checkCudaErrors(cudaMalloc((void **)&gpuSolveMatrix, sizeof(double)*squareMatrixDimension));

	// Copy Data from CPU to GPU
	checkCudaErrors(cudaMemcpy(gpuMatrix, cpuMatrix, sizeof(double)*squareMatrixDimension*squareMatrixDimension, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpuSolveMatrix, cpuSolveMatrix, sizeof(double)*squareMatrixDimension, cudaMemcpyHostToDevice));

	// Keep Track of Start Time
	start = get_time();

	// Create Buffer
	checkCudaErrors(cusolverDnDgetrf_bufferSize(handle,
															  squareMatrixDimension,
															  squareMatrixDimension,
															  (double*)gpuMatrix,
														 	  squareMatrixDimension,
															  &bufferSize));

	// Allocate Memory on GPU
	checkCudaErrors(cudaMalloc(&info, sizeof(int)));
	checkCudaErrors(cudaMalloc(&buffer, sizeof(double)*bufferSize));
	checkCudaErrors(cudaMalloc(&gpuPivotMatrix, sizeof(int)*squareMatrixDimension));

	// Initiailize Memory for Info
	checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

	// Perform LU Decomposition
	checkCudaErrors(cusolverDnDgetrf(handle, 
												squareMatrixDimension, 
												squareMatrixDimension,
												cpuInvertedMatrix, 
											   squareMatrixDimension,
												buffer, 
												gpuPivotMatrix, 
												info));
	checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));
	
	// Compute Matrix Inverse
	checkCudaErrors(cudaMemcpy(gpuResultMatrix, gpuSolveMatrix, sizeof(double)*squareMatrixDimension, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cusolverDnDgetrs(handle, 
												CUBLAS_OP_N, 
												squareMatrixDimension, 
												1, 
												gpuMatrix,
												squareMatrixDimension,
												gpuPivotMatrix,
												gpuResultMatrix, 
												squareMatrixDimension, 
												info));
	checkCudaErrors(cudaDeviceSynchronize());

	// Copy results from GPU Memory to Host Memory
	checkCudaErrors(cudaMemcpy(cpuResultMatrix, gpuResultMatrix, sizeof(double)*squareMatrixDimension, cudaMemcpyDeviceToHost));
	
	// Keep Track of Stop Time 
	stop = get_time();

	// Synchronize Events
	timeToCompleteInMs = 0;
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&timeToCompleteInMs, start, stop));
	
	// Free up allocated memory
	if (cpuResultMatrix) { free(cpuResultMatrix); }
	if (cpuSolveMatrix) { free(cpuSolveMatrix); }
	if (gpuResultMatrix) { checkCudaErrors(cudaFree(gpuResultMatrix)); }
	if (gpuSolveMatrix) { checkCudaErrors(cudaFree(gpuSolveMatrix)); }
	if (handle) { checkCudaErrors(cusolverDnDestroy(handle)); }
	if (stream) { checkCudaErrors(cudaStreamDestroy(stream)); }
	if (gpuMatrix) { checkCudaErrors(cudaFree(gpuMatrix)); }

	// return time required to complete matrix inversion
	return timeToCompleteInMs;
}
