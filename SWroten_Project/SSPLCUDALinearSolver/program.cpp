// program.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Matrix.h"
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

using namespace std;

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
	//std::vector<std::vector<double>> matrixElements;
	std::vector<std::vector<double>> matrixElements{ { 41, 67, 34 },{ 1, 69, 24 },{ 78, 58, 62 } };

	//// Print Arguments for Debugging printf("Number of Arguments: %i\n", argc);
	//std::cout << "Number of Arguments: " << argc << endl;
	//std::cout << "Binary Name:         " << argv[0] << endl;
	//std::cout << "Matrix Dimension:    " << argv[1] << endl;

	//// Get Matrix Dimension from Commandline
	////matrixDimension = std::atoi(argv[1]);
	//matrixDimension = 2;

	// Initialize Matrix with random values for test
	//for (int i = 0; i < matrixDimension; i++)
	//{
	//	// Initialize Vector to empty vector
	//	matrixElements.push_back({ std::vector<double>{} });

	//	// Step through each element and asign random value
	//	for (int j = 0; j < matrixDimension; j++)
	//	{
	//		// Add Random Value
	//		matrixElements[i].push_back(std::max(rand() % 100, 1));
	//	}
	//}

	// Create Matrix for Test
	Matrix matrix{ matrixElements };

	// Print out Device Properties
	//matrix.PrintGPUProperties();

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

	// Perform Test 3-Elements
	MatrixInversionTestForCPUvsGPU(31);

	//// Perform Test for 5-Elements
	//MatrixInversionTestForCPUvsGPU(5);
	//
	//// Perform Test for 10-Elements
	//MatrixInversionTestForCPUvsGPU(10);

	//// Perform Test for 32-Elements
	//MatrixInversionTestForCPUvsGPU(32);
	
	// Wait for user to close application
	std::cout << "Press Any Button to Exit..." << endl;

	// Get User Input
	getline(cin, userInput);
}
