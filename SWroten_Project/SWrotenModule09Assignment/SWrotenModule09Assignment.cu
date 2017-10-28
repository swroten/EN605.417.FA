
#include "cuda_runtime.h"
#include "math_functions.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <curand.h>
#include <cusolverSp.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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
	cudaSetDevice(0);

	// Get Number of Blocks Required and Number of Threads
	numberOfBlocks = (int)(numberOfElements / maxThreadsPerBlock) + 1;
	numberOfThreads = (int)(numberOfElements % maxThreadsPerBlock);

	// Allocate GPU Memory for States
	cudaMalloc((void**)&states, numberOfElements * sizeof(curandState_t));

	// Run Initialization
	init << <numberOfBlocks, numberOfThreads >> >((unsigned int)time(0), states, numberOfElements);

	// Allocate GPU Memory for input matrix
	cudaMalloc((void**)&gpuMatrix, numberOfBytesInMatrix);

	// Add Random Numbers to Matrix
	randoms << <numberOfBlocks, numberOfThreads >> >(states, gpuMatrix, numberOfElements);

	// Copy Matrix Data From CPU Memory to GPU Memory
	cudaMemcpy(cpuMatrix, gpuMatrix, numberOfBytesInMatrix, cudaMemcpyDeviceToHost);

	// Free Allocated Memory
	cudaFree(gpuMatrix);
}

std::string GetMatrixAsString(double *matrixElementsPntr, int squareMatrixDimension)
{
	// Initialize Variable
	std::ostringstream matrixAsStringStream;

	matrixAsStringStream << "{" << std::endl;

	// Step through each row in matrix
	for (int i = 0; i < squareMatrixDimension; i++)
	{
		// Spacing for initial elements
		matrixAsStringStream << "  ";

		// Step through each column in this row
		for (int j = 0; j < squareMatrixDimension; j++)
		{
			matrixAsStringStream << matrixElementsPntr[((i * squareMatrixDimension) + j)] << " ";
		}

		matrixAsStringStream << std::endl;
	}

	matrixAsStringStream << "};" << std::endl;

	// Return Matrix as String
	return matrixAsStringStream.str();
}

float GetInvertedMatrixCPU(double *cpuInvertedMatrix, const double *cpuLUMatrix, const int *cpuPivotMatrix, const int squareMatrixDimension)
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

float GetLUDecompositionMatrixCPU(double *cpuInvertedMatrix, int *cpuPivotMatrix, const double *cpuMatrix, const int numberOfElements, const int squareMatrixDimension)
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

float GetCuSparseInvertedMatrixGPU(double *cpuInvertedMatrix, const double *cpuMatrix, const int squareMatrixDimension)
{
	// Initialize Variables
	int batch = 1;
	int *info = NULL;
	cudaEvent_t stop;
	cudaEvent_t start;
	cublasHandle_t handle;
	double *gpuLUDecompositionMatrix = NULL;
	int *gpuPivotMatrix = NULL;
	double *gpuInvertedMatrix = NULL;
	float timeToCompleteInMs = 0;

	// Allocate Device Memory
	cudaMalloc((void **)&gpuPivotMatrix, sizeof(int)*squareMatrixDimension);
	cudaMalloc((void **)&gpuLUDecompositionMatrix, sizeof(double)*squareMatrixDimension*squareMatrixDimension);
	cudaMalloc((void **)&gpuInvertedMatrix, sizeof(double)*squareMatrixDimension*squareMatrixDimension);

	// Copy Data from CPU to GPU
	cudaMemcpy(gpuLUDecompositionMatrix, cpuMatrix, sizeof(double)*squareMatrixDimension*squareMatrixDimension, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuInvertedMatrix, cpuInvertedMatrix, sizeof(double)*squareMatrixDimension*squareMatrixDimension, cudaMemcpyHostToDevice);

	// Initialize More Variables
	double **gpuInvertedMatrixArrayOfPointers = NULL;
	double **gpuLUDecompositionMatrixArrayOfPointers = NULL;
	double *cpuInvertedMatrixArray[] = { gpuInvertedMatrix };
	double *cpuLUDecompositionMatrixArray[] = { gpuLUDecompositionMatrix };

	// Create Handle
	cublasCreate_v2(&handle);

	// Allocate Memory to device arrays
	cudaMalloc((void **)&gpuInvertedMatrixArrayOfPointers, sizeof(cpuInvertedMatrixArray));
	cudaMalloc((void **)&gpuLUDecompositionMatrixArrayOfPointers, sizeof(cpuLUDecompositionMatrixArray));

	// Copy Data from CPU to GPU
	cudaMemcpy(gpuInvertedMatrixArrayOfPointers, cpuInvertedMatrixArray, sizeof(cpuInvertedMatrixArray), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuLUDecompositionMatrixArrayOfPointers, cpuLUDecompositionMatrixArray, sizeof(cpuLUDecompositionMatrixArray), cudaMemcpyHostToDevice);

	// Keep Track of Start Time
	start = get_time();

	// Create Buffer
	cudaMalloc((void **)&info, sizeof(int));

	// Initiailize Memory for Info
	cudaMemset(info, 0, sizeof(int));

	// Perform LU Decomposition
	cublasDgetrfBatched(handle, squareMatrixDimension, gpuLUDecompositionMatrixArrayOfPointers, squareMatrixDimension, gpuPivotMatrix, info, batch);

	// Compute Matrix Inverse
	cublasDgetriBatched(handle, squareMatrixDimension, (const double **)gpuLUDecompositionMatrixArrayOfPointers, squareMatrixDimension, gpuPivotMatrix, gpuInvertedMatrixArrayOfPointers, squareMatrixDimension, info, batch);
	cudaDeviceSynchronize();

	// Copy results from GPU Memory to Host Memory
	cudaMemcpy(cpuInvertedMatrix, gpuInvertedMatrix, sizeof(double)*squareMatrixDimension*squareMatrixDimension, cudaMemcpyDeviceToHost);

	// Keep Track of Stop Time 
	stop = get_time();

	// Synchronize Events
	timeToCompleteInMs = 0;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeToCompleteInMs, start, stop);

	// Free up allocated memory
	if (handle) { cublasDestroy_v2(handle); }
	if (gpuPivotMatrix) { cudaFree(gpuPivotMatrix); }
	if (gpuInvertedMatrix) { cudaFree(gpuInvertedMatrix); }
	if (gpuInvertedMatrixArrayOfPointers) { cudaFree(gpuInvertedMatrixArrayOfPointers); }
	if (gpuLUDecompositionMatrixArrayOfPointers) { cudaFree(gpuLUDecompositionMatrixArrayOfPointers); }

	// return time required to complete matrix inversion
	return timeToCompleteInMs;
}

float InvertCPU(double *cpuInvertedMatrix, const double *cpuMatrix, const int squareMatrixDimension)
{
	// Initialize Variables
	float timeToGetLUDecompositionMatrix;
	float timeToInvertMatrixFromLUDecompositionAndPivotMatrix;
	const int numberOfElements = squareMatrixDimension * squareMatrixDimension;
	int *cpuPivotMatrixElementsPntr = (int *)malloc(squareMatrixDimension * sizeof(int));
	double *cpuLUMatrixElementsPntr = (double *)malloc(squareMatrixDimension * squareMatrixDimension * sizeof(double));

	// Initialize Pivot Matrix
	for (int i = 0; i < squareMatrixDimension; i++)
	{
		cpuPivotMatrixElementsPntr[i] = i;
	}

	// Add elements to matrix
	for (int i = 0; i < numberOfElements; i++)
	{
		cpuInvertedMatrix[i] = cpuMatrix[i];
		cpuLUMatrixElementsPntr[i] = cpuMatrix[i];
	}

	// On the CPU - Perform LU Decomposition to get LU Matrix and Pivot Matrix - returns time required to complete in ms
	timeToGetLUDecompositionMatrix = GetLUDecompositionMatrixCPU(cpuLUMatrixElementsPntr,
		cpuPivotMatrixElementsPntr,
		cpuMatrix,
		numberOfElements,
		squareMatrixDimension);

	// On the CPU - Use the LU Matrix and Pivot Matrix to get Inverte Matrix - returns time required to complete in ms 
	timeToInvertMatrixFromLUDecompositionAndPivotMatrix = GetInvertedMatrixCPU(cpuInvertedMatrix,
		cpuLUMatrixElementsPntr,
		cpuPivotMatrixElementsPntr,
		squareMatrixDimension);

	// Accumulate all Time Required to invert Matrix on cpu
	return (timeToGetLUDecompositionMatrix + timeToInvertMatrixFromLUDecompositionAndPivotMatrix);
}

float InvertGPU(double *cpuInvertedMatrix, const double *cpuMatrix, const int squareMatrixDimension)
{
	return GetCuSparseInvertedMatrixGPU(cpuInvertedMatrix, cpuMatrix, squareMatrixDimension);
}

// Main Function
int main(int argc, char *argv[])
{
	// Print Arguments for Debugging 
	std::cout << "Number of Arguments: " << argc << endl;
	std::cout << endl;

	// First Arg is Binary Name
	std::cout << "Binary Name: " << argv[0] << endl;
	std::cout << endl;

	// Second Arg is # of Threads 
	std::cout << "Matrix Dimension: " << argv[1] << endl;
	std::cout << endl;

	// Initialize Variables
	std::string userInput{ "" };
	bool invertSuccess = false;
	float cpuTimeToCompleteInMs = 0;
	float gpuTimeToCompleteInMs = 0;
	int numberOfRows = atoi(argv[1]);
	int numberOfColumns = atoi(argv[1]);
	double *cpuMatrixElementsPntr = 0;
	std::string cpuMatrixInversionResult{ "" };
	std::string gpuMatrixInversionResult{ "" };
	int numberOfElements = numberOfRows * numberOfColumns;
	double *cpuInvertedMatrixElementsPntrFromCPUComputation = 0;
	double *cpuInvertedMatrixElementsPntrFromGPUComputation = 0;
	int squareMatrixDimension = min(numberOfRows, numberOfColumns);

	// Allocate Memory
	cpuMatrixElementsPntr = (double *)malloc(numberOfElements * sizeof(double));
	cpuInvertedMatrixElementsPntrFromGPUComputation = (double *)malloc(numberOfElements * sizeof(double));
	cpuInvertedMatrixElementsPntrFromCPUComputation = (double *)malloc(numberOfElements * sizeof(double));

	// Get Random Values for Elements
	GetRandomNumbersForMatrix(cpuMatrixElementsPntr, numberOfElements);

	// Print Matrix as String
	std::cout << "Original Matrix:" << endl;
	std::cout << GetMatrixAsString(cpuMatrixElementsPntr, squareMatrixDimension) << endl;
	std::cout << endl;

	// Perform GPU Matrix Inversion 
	gpuTimeToCompleteInMs = InvertGPU(cpuInvertedMatrixElementsPntrFromGPUComputation, cpuMatrixElementsPntr, squareMatrixDimension);

	// Get GPU Computed Matrix Inversion as String
	gpuMatrixInversionResult = GetMatrixAsString(cpuInvertedMatrixElementsPntrFromGPUComputation, squareMatrixDimension);

	// Print Inverted Matrix (GPU) as String
	std::cout << "Inverted Matrix (GPU):" << endl;
	std::cout << gpuMatrixInversionResult << endl;
	std::cout << endl;

	// Perform CPU Matrix Inversion
	cpuTimeToCompleteInMs = InvertCPU(cpuInvertedMatrixElementsPntrFromCPUComputation, cpuMatrixElementsPntr, squareMatrixDimension);

	// Get CPU Computed Matrix Inversion as String
	cpuMatrixInversionResult = GetMatrixAsString(cpuInvertedMatrixElementsPntrFromCPUComputation, squareMatrixDimension);

	// Print Inverted Matrix (CPU) as String
	std::cout << "Inverted Matrix (CPU):" << endl;
	std::cout << cpuMatrixInversionResult << endl;
	std::cout << endl;

	// Check Results for success
	invertSuccess = (cpuMatrixInversionResult == gpuMatrixInversionResult);

	// Print out Results
	std::cout << "Results for Dimension " << squareMatrixDimension << ":" << endl;
	std::cout << "  Invert Equivalent: " << ((invertSuccess == 1) ? "Success" : "Failed") << endl;
	std::cout << "  CPU Time (ms):     " << cpuTimeToCompleteInMs << endl;
	std::cout << "  GPU Time (ms):     " << gpuTimeToCompleteInMs << endl;
	std::cout << "  Fastest:           " << ((cpuTimeToCompleteInMs < gpuTimeToCompleteInMs) ? "CPU" : "GPU") << endl;
	std::cout << endl;

	// Wait for user to close application
	std::cout << "Press Any Button to Exit..." << endl;

	// Get User Input
	getline(cin, userInput);

	// return
	return EXIT_SUCCESS;
}
