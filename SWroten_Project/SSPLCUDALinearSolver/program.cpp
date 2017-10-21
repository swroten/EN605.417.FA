// program.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Matrix.h"
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

using namespace std;





void DetectFailureCasesForMatrixInversion(int dimension)
{
	// Variables
	bool luSuccess = false;
	bool invertSuccess = false;
	std::string cpuLUMatrix;
	std::string gpuLUMatrix;
	std::string cpuInvertedMatrix;
	std::string gpuInvertedMatrix;
	int matrixDimension = dimension;
	std::vector<std::vector<double>> matrixElements;

	// Get Matrix Dimension from Input
	matrixDimension = dimension;

	// Initialize Matrix with random values for test
	for (int i = 0; i < matrixDimension; i++)
	{
		// Initialize Vector to empty vector
		matrixElements.push_back({ std::vector<double>{} });

		// Step through each element and asign random value
		for (int j = 0; j < matrixDimension; j++)
		{
			// Add Random Value
			matrixElements[i].push_back(std::max(rand() % 100, 1));
		}
	}

	// Create Matrix for Test
	Matrix matrix{ matrixElements };

	// Begin Inversion of Matrix on CPU
	matrix.InvertCPU();

	// Get CPU LU Matrix as String
	cpuLUMatrix = matrix.GetLUMatrixToString();

	// Get CPU Inverted Matrix as String
	cpuInvertedMatrix = matrix.GetInvertedMatrixToString();

	// Begin Inversion of Matrix on GPU
	matrix.InvertGPU();

	// Get CPU LU Matrix as String
	gpuLUMatrix = matrix.GetLUMatrixToString();

	// Get CPU Inverted Matrix as String
	gpuInvertedMatrix = matrix.GetInvertedMatrixToString();

	// Check Results
	luSuccess = (cpuLUMatrix == gpuLUMatrix);
	invertSuccess = (cpuInvertedMatrix == gpuInvertedMatrix);

	// Print out Results
	if (!luSuccess || !invertSuccess)
	{
		std::cout << "Failure Detected for Dimension " << dimension << ":" << endl;
		std::cout << "  LU Equivalent:     " << ((luSuccess == 1) ? "Success" : "Failed") << endl;
		std::cout << "  Invert Equivalent: " << ((invertSuccess == 1) ? "Success" : "Failed") << endl;
		std::cout << endl;
	}
}

bool MatrixInversionTestForCPUvsGPU(int dimension)
{
	// Variables
	bool luSuccess = false;
	bool invertSuccess = false;
	std::string cpuLUMatrix;
	std::string gpuLUMatrix;
	std::string cpuInvertedMatrix;
	std::string gpuInvertedMatrix;
	int matrixDimension = dimension;
	std::vector<std::vector<double>> matrixElements;

	// Get Matrix Dimension from Input
	matrixDimension = dimension;

	// Initialize Matrix with random values for test
	for (int i = 0; i < matrixDimension; i++)
	{
		// Initialize Vector to empty vector
		matrixElements.push_back({ std::vector<double>{} });

		// Step through each element and asign random value
		for (int j = 0; j < matrixDimension; j++)
		{
			// Add Random Value
			matrixElements[i].push_back(std::max(rand() % 100, 1));
		}
	}

	// Create Matrix for Test
	Matrix matrix{ matrixElements };

	// Print out Device Properties
	matrix.PrintGPUProperties();

	// Print out Matrix to Console
	std::cout << "Original Matrix:" << endl;
	std::cout << matrix.ToString() << endl;
	std::cout << endl;

	// Begin Inversion of Matrix on CPU
	matrix.InvertCPU();

	// Get CPU LU Matrix as String
	cpuLUMatrix = matrix.GetLUMatrixToString();

	// Get CPU Inverted Matrix as String
	cpuInvertedMatrix = matrix.GetInvertedMatrixToString();

	// Begin Inversion of Matrix on GPU
	matrix.InvertGPU();

	// Get CPU LU Matrix as String
	gpuLUMatrix = matrix.GetLUMatrixToString();

	// Get CPU Inverted Matrix as String
	gpuInvertedMatrix = matrix.GetInvertedMatrixToString();
	
	// Print out CPU LU Matrix to Console
	std::cout << "LU Matrix (CPU):" << endl;
	std::cout << cpuLUMatrix << endl;
	std::cout << endl;

	// Print out GPU LU Matrix to Console
	std::cout << "LU Matrix (GPU):" << endl;
	std::cout << gpuLUMatrix << endl;
	std::cout << endl;

	// Print out CPU Inverted Matrix to Console
	std::cout << "Inverted Matrix (CPU):" << endl;
	std::cout << cpuInvertedMatrix << endl;
	std::cout << endl;

	// Print out GPU Inverted Matrix to Console
	std::cout << "Inverted Matrix (GPU):" << endl;
	std::cout << gpuInvertedMatrix << endl;
	std::cout << endl;

	// Check Results
	luSuccess = (cpuLUMatrix == gpuLUMatrix);
	invertSuccess = (cpuInvertedMatrix == gpuInvertedMatrix);

	// Print out Results
	std::cout << "Results for Dimension " << dimension << ":" << endl;
	std::cout << "  LU Equivalent:     " << ((luSuccess == 1) ? "Success" : "Failed") << endl;
	std::cout << "  Invert Equivalent: " << ((invertSuccess == 1) ? "Success" : "Failed") << endl;
	std::cout << "  CPU Time (ms):     " << matrix.GetCPUTimeToInvertInMs() << endl;
	std::cout << "  GPU Time (ms):     " << matrix.GetGPUTimeToInvertInMs() << endl;
	std::cout << "  Fastest:           " << ((matrix.GetCPUTimeToInvertInMs() < matrix.GetGPUTimeToInvertInMs()) ? "CPU" : "GPU") << endl;
	std::cout << endl;

	// Return Result
	return invertSuccess;
}

void main(int argc, char *argv[])
{
	// Initialize Variables
	std::string userInput{ "" };
	int numberOfTests = 31;

	// Perform Test 3-Elements
	for (int i = 0; i < numberOfTests; i++)
	{
		DetectFailureCasesForMatrixInversion(i);
	}
		
	// Wait for user to close application
	std::cout << "Press Any Button to Exit..." << endl;

	// Get User Input
	getline(cin, userInput);
}
