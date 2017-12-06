#include "stdafx.h"
#include "Matrix.h"
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <tuple>

using namespace std;

std::tuple<double, double> MatrixInversionPerformanceComparisonCublasVsCPU(int dimension)
{
	// Initialize Variables
	std::string cpuInvertedMatrix;
	std::string gpuInvertedMatrix;
	int matrixDimension = dimension;
	double cpuInvertedMatrixMagnitude = 0;
	double gpuInvertedMatrixMagnitude = 0;
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
	
	// Get CPU Inverted Matrix as String
	cpuInvertedMatrix = matrix.GetInvertedMatrixToString();

	// Get CPU Inverted Matrix Magnitude
	cpuInvertedMatrixMagnitude = matrix.GetMagnitudeOfInvertedMatrixElements();

	// Begin Inversion of Matrix on GPU with CUBLAS
	matrix.InvertWithCuBLASOnGPU();

	// Get GPU with CUBLAS Inverted Matrix as String
	gpuInvertedMatrix = matrix.GetInvertedMatrixToString();

	// Get CPU Inverted Matrix Magnitude
	gpuInvertedMatrixMagnitude = matrix.GetMagnitudeOfInvertedMatrixElements();

	// Print out Results
	std::cout << "Results for Dimension " << dimension << ":" << endl;
	std::cout << "  CPU Magnitude:     " << cpuInvertedMatrixMagnitude << endl;
	std::cout << "  GPU Magnitude:     " << gpuInvertedMatrixMagnitude << endl;
	std::cout << "  CPU Time (ms):     " << matrix.GetCPUTimeToInvertInMs() << endl;
	std::cout << "  GPU Time (ms):     " << matrix.GetGPUTimeToInvertInMs() << endl;
	std::cout << "  Fastest:           " << ((matrix.GetCPUTimeToInvertInMs() < matrix.GetGPUTimeToInvertInMs()) ? "CPU" : "GPU") << endl;
	std::cout << endl;

	return std::make_tuple(matrix.GetCPUTimeToInvertInMs(), matrix.GetGPUTimeToInvertInMs());
}

void RunTestsForSpecifiedTestCases(const int numberOfTrials, const std::vector<int> testCases)
{
	// Initialize Variables
	std::vector<double> cpuTimesForThisTestCase;
	std::vector<double> gpuTimesForThisTestCase;
	std::vector<std::vector<double>> cpuTimesPerTestCase;
	std::vector<std::vector<double>> gpuTimesPerTestCase;
	std::tuple<double, double> resultingTimeFromInMsWithFirstIndexCPUAndSecondIndexGPU;

	// Perform Test for each test case
	for (int i = 0; i < testCases.size(); i++)
	{
		// Create new vector for CPU & GPU Times for this Test Case
		cpuTimesForThisTestCase = std::vector<double>();
		gpuTimesForThisTestCase = std::vector<double>();

		for (int j = 0; j < numberOfTrials; j++)
		{
			// Get CPU and GPU Time
			resultingTimeFromInMsWithFirstIndexCPUAndSecondIndexGPU = MatrixInversionPerformanceComparisonCublasVsCPU(testCases[i]);

			// Add CPU Time for this Test Case
			cpuTimesForThisTestCase.push_back(std::get<0>(resultingTimeFromInMsWithFirstIndexCPUAndSecondIndexGPU));

			// Add GPU Time for this Test Case
			gpuTimesForThisTestCase.push_back(std::get<1>(resultingTimeFromInMsWithFirstIndexCPUAndSecondIndexGPU));
		}

		// Print Results for this Dimension
		std::cout << "Dimension " << testCases[i] << ": " << endl;

		// Handle CPU Times First
		std::cout << "  CPU (ms) - ";

		// Step through each time
		for (int j = 0; j < cpuTimesForThisTestCase.size(); j++)
		{
			std::cout << cpuTimesForThisTestCase[j] << ", ";
		}

		// Add End Line
		std::cout << endl;

		// Next Handle GPU Times
		std::cout << "  GPU (ms) - ";

		// Step through each time
		for (int j = 0; j < gpuTimesForThisTestCase.size(); j++)
		{
			std::cout << gpuTimesForThisTestCase[j] << ", ";
		}

		// Add End Line
		std::cout << endl;
		std::cout << endl;

		// Add to Collection of CPU & GPU times Per Test Case
		cpuTimesPerTestCase.push_back(cpuTimesForThisTestCase);
		gpuTimesPerTestCase.push_back(gpuTimesForThisTestCase);
	}

	// Print line for spacing
	std::cout << endl;

	// Print out Resulting Times in Comma Separate String
	for (int i = 0; i < testCases.size(); i++)
	{
		// Print Results for this Dimension
		std::cout << "Dimension " << testCases[i] << ": " << endl;

		// Handle CPU Times First
		std::cout << "  CPU (ms) - ";

		// Step through each time
		for (int j = 0; j < cpuTimesPerTestCase[i].size(); j++)
		{
			std::cout << cpuTimesPerTestCase[i][j] << ", ";
		}

		// Add End Line
		std::cout << endl;

		// Next Handle GPU Times
		std::cout << "  GPU (ms) - ";

		// Step through each time
		for (int j = 0; j < gpuTimesPerTestCase[i].size(); j++)
		{
			std::cout << gpuTimesPerTestCase[i][j] << ", ";
		}

		// Add End Line
		std::cout << endl;
	}
}

void main(int argc, char *argv[])
{
	// Initialize Variables
	std::string userInput{ "" };
	const int numberOfTrials = 10;
	const std::vector<int> testCases = { 5, 10, 25, 50, 100, 250, 500, 750, 1000 };

	// Run Test Cases
	RunTestsForSpecifiedTestCases(numberOfTrials, testCases);	
		
	// Wait for user to close application
	std::cout << "Press Any Button to Exit..." << endl;

	// Get User Input
	getline(cin, userInput);
}
