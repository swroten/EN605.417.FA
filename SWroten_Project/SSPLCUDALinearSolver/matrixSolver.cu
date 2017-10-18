
#include "cuda_runtime.h"
#include "math_functions.h"
#include "device_launch_parameters.h"

#include <sstream>
#include <stdio.h>
#include <tchar.h>
#include <cusolverSp.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "matrixSolver.h"

using namespace std;

// Function to retrieving time
__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

__global__ void max(int *maxValueIndex,
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

__global__ void shur_complement(double *matrix, 
										  const int currentColumnInMatrix,
										  const int squareMatrixDimension)
{
	// Initialize
	extern __shared__ double matrixInSharedMemory[];

	// Initialize Variables
	const int threadIndexWithinEntireMatrix = (blockIdx.x * blockDim.x) + threadIdx.x;
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

__global__ void shur_complement_works(double *matrix,
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

	// Initialize Pivot Matrix Value
	pivotMatrix[columnIndexInMatrix] = columnIndexInMatrix;

	// Get all row elements in current column 
	for (int rowIndex = columnIndexInMatrix; rowIndex < squareMatrixDimension; rowIndex++)
	{			
		// Get Max Row Value in Column
		if (abs(matrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix]) > largestValue)
		{
			largestValue = abs(matrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix]);
			maxColumnIndices[columnIndexInMatrix] = rowIndex;
		}
	}

	// Singular Matrix
	if (largestValue == 0) 
	{ 
		// Stop all Threads and kill Kernel
		__threadfence();
		asm("trap;");
	}
	
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

	// Schur Complement
	for (int rowIndex = columnIndexInMatrix + 1; rowIndex < squareMatrixDimension; rowIndex++)
	{
		// Divide each row element below diagonal of current column by diagonal element for current column
		matrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix] = matrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix] /
			matrix[(columnIndexInMatrix * squareMatrixDimension) + columnIndexInMatrix];

		// Sync Threads
		__syncthreads();

		if (columnIndexInMatrix != 0)
		{
			// Subtract each row element below diagonal of current column by product of diagonal element and current row element
			matrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix] = matrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix] -
				(matrix[(rowIndex * squareMatrixDimension) + columnIndexInMatrix] * matrix[(columnIndexInMatrix * squareMatrixDimension) + columnIndexInMatrix]);
		}

		// Sync Threads
		__syncthreads();
	}

	// Sync Threads
	__syncthreads();
}

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


float GetInvertedMatrixCPU(double *cpuInvertedMatrix,
									int *cpuPivotMatrix,
									const int squareMatrixDimension)
{
	// Initialize Variables
	cudaEvent_t stop;
	cudaEvent_t start;
	double elementValue = 0.0;
	double sumLowerTriangle = 0;
	double sumUpperTriangle = 0;
	float timeToCompleteInMs = 0;
	double *solveArray = (double *)malloc(squareMatrixDimension * sizeof(double));
	double *fowardSubstitutionArray = (double *)malloc(squareMatrixDimension * sizeof(double));
	double *backwardSubtitutionArray = (double *)malloc(squareMatrixDimension * sizeof(double));

	// Keep Track of Start Time
	start = get_time();

	// for each column in matrix
	for (int columnIndexInMatrix = 0; columnIndexInMatrix < squareMatrixDimension; columnIndexInMatrix++)
	{
		// Initialize Forward and Backward Substition Array
		fowardSubstitutionArray[columnIndexInMatrix] = 0.0;
		backwardSubtitutionArray[columnIndexInMatrix] = 0.0;
	}

	// Solve for the Identity Matrix using resuls of LU Decomposition
	//  Step through each row and solve
	for (int overallRowIndex = 0; overallRowIndex < squareMatrixDimension; overallRowIndex++)
	{
		// Initialize Solve Array to be all 0 except for diagonal for 
		for (int columnIndexInMatrix = 0; columnIndexInMatrix < squareMatrixDimension; columnIndexInMatrix++)
		{
			// If Diagonal
			if (columnIndexInMatrix == overallRowIndex)
			{
				// Set to Identity
				solveArray[columnIndexInMatrix] = 1;
			}
			else
			{
				// Otherwise set to 0
				solveArray[columnIndexInMatrix] = 0;
			}
		}

		// Solve by doing foward substition
		for (int rowIndex = 0; rowIndex < squareMatrixDimension; rowIndex++)
		{
			// Set Accumulating sum to 0
			sumLowerTriangle = 0;

			// Step through Each Column
			for (int columnIndex = 0; columnIndex < squareMatrixDimension; columnIndex++)
			{
				// Get if this is the Diaganol
				if (rowIndex == columnIndex)
				{
					// Set to Identity if it is
					elementValue = 1.0;
				}
				else
				{
					// Get LU Matrix Value if not
					elementValue = cpuInvertedMatrix[(rowIndex * squareMatrixDimension) + columnIndex];
				}

				// Accumulate Lower Triangle Sum
				sumLowerTriangle += (elementValue * fowardSubstitutionArray[columnIndex]);
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
				sumUpperTriangle += (cpuInvertedMatrix[(rowIndex * squareMatrixDimension) + columnIndex] *
					backwardSubtitutionArray[columnIndex]);
			}

			backwardSubtitutionArray[rowIndex] = ((fowardSubstitutionArray[rowIndex] - sumUpperTriangle) /
				cpuInvertedMatrix[(rowIndex * squareMatrixDimension) + rowIndex]);
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
									int *cpuPivotMatrix,
									const int squareMatrixDimension)
{
	return 0;
}

float GetLUDecompositionMatrixGPU(double *cpuInvertedMatrix,
											 int *cpuPivotMatrix,
											 const double *cpuMatrix,
											 const int numberOfElements,
											 const int squareMatrixDimension)
{
	// Initialize Variables
	int device = 0;
	cudaError_t error;
	cudaEvent_t stop;
	cudaEvent_t start;
	cudaDeviceProp prop;
	int maxValueIndex = 0;
	int maxThreadsPerBlock;
	int *gpuPivotMatrix = 0;
	double largestValue = 0.0;
	int *gpuMaxValueIndex = 0;
	float timeToCompleteInMs = 0;
	double *gpuInvertedMatrix = 0;
	double matrixCurrentColumnValue = 0;
	double matrixLargestColumnValue = 0;
	int pivotMatrixCurrentColumnValue = 0;
	int pivotMatrixLargestColumnValue = 0;
	int *cpuMaxValueIndex = (int *)malloc(sizeof(int));
	const int numberOfBytesInMatrix = numberOfElements * sizeof(double);

	// Copy Initial Matrix into Inverted Matrix
	cudaMemcpy(cpuInvertedMatrix, cpuMatrix, numberOfBytesInMatrix, cudaMemcpyHostToHost);
	
	// Get Properties of this device
	cudaGetDeviceProperties(&prop, device);

	// Get Max Threads Per Block
	maxThreadsPerBlock = prop.maxThreadsPerBlock;

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
	if (cudaMalloc((void**)&gpuInvertedMatrix, numberOfBytesInMatrix) != cudaSuccess)
	{
		// Print error regarding inability to allocate GPU Device Memory
		fprintf(stderr, "Error: Failed to allocate memory on CUDA capable GPU device!");

		// exit before executing any further
		return 0;
	}

	// Allocate GPU Memory for Max Value Index
	if (cudaMalloc((void**)&gpuMaxValueIndex, sizeof(int)) != cudaSuccess)
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
		cudaMemcpy(gpuInvertedMatrix, cpuInvertedMatrix, numberOfBytesInMatrix, cudaMemcpyHostToDevice);

		// Get Number of Threads Required for Shurs Complement
		int numberOfBlocks = max((int)(numberOfElements / maxThreadsPerBlock), 1);
		int numberOfThreads = max((int)(numberOfElements % maxThreadsPerBlock), 1);

		// Perform Shur Complement
		shur_complement<<<numberOfBlocks, numberOfThreads, numberOfBytesInMatrix>>>(gpuInvertedMatrix, columnIndexInMatrix, squareMatrixDimension);

		// Copy result of operation from GPU to CPU Memory
		cudaMemcpy(cpuInvertedMatrix, gpuInvertedMatrix, numberOfBytesInMatrix, cudaMemcpyDeviceToHost);


		//// Get Number of Threads Required for Shurs Complement
		//int submatrixRowDimension = (squareMatrixDimension - (columnIndexInMatrix + 1));
		//int submatrixColumnDimension = (squareMatrixDimension - columnIndexInMatrix);
		//int numberOfSubMatrixElements = (submatrixColumnDimension * submatrixRowDimension);
		//int numberOfBlocks = max((int)(numberOfSubMatrixElements / maxThreadsPerBlock), 1);
		//int numberOfThreads = max((int)(numberOfSubMatrixElements % maxThreadsPerBlock), 1);

		//// Verify Number of Elements is greater than 0
		//if (numberOfSubMatrixElements > 0)
		//{
		//	// Perform Shur Complement
		//	shur_complement_works<<<numberOfBlocks, numberOfThreads>>>(gpuInvertedMatrix, columnIndexInMatrix, squareMatrixDimension);

		//	// Copy result of operation from GPU to CPU Memory
		//	cudaMemcpy(cpuInvertedMatrix, gpuInvertedMatrix, numberOfBytesInMatrix, cudaMemcpyDeviceToHost);
		//}
	}

	// Keep Track of Stop Time 
	stop = get_time();

	// Synchronize Events
	timeToCompleteInMs = 0;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeToCompleteInMs, start, stop);
		
	// Check for any errors launching the kernel
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// Print error regarding failure during kernel launch
		fprintf(stderr, "Error: Failed to complete Matrix Inversion kernel because '%s'!\n", cudaGetErrorString(error));

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


