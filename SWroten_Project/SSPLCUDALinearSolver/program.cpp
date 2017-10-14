// program.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "matrixSolver.h"
#include "Matrix.h"
#include <vector>
#include <string>
#include <iostream>

using namespace std;

int main()
{
	// Variables
	Matrix matrix{ std::vector<std::vector<double>>({ { 3.0, 1.0 },{ 4.0, 2.0 } }) };
	
	// Allocate Memory for matrix & inverted matrix array pointer
	double *cpuMatrixPntr = (double *)malloc(matrix.GetNumberOfElements() * sizeof(double));
	double *cpuInvertedMatrixPntr = (double *)malloc(matrix.GetNumberOfElements() * sizeof(double));
	
	// Begin Inversion of Matrix
	float timeToCompleteInMs = InvertMatrix(cpuInvertedMatrixPntr, cpuMatrixPntr, matrix.GetNumberOfElements());

	// Print out Matrix to Console
	std::cout << "Result:" << endl;
	std::cout << matrix.ToString() << endl;
	std::cout << "Time (ms): " << timeToCompleteInMs << endl;
}
