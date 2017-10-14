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
	
	// Begin Inversion of Matrix
	float timeToCompleteInMs = InvertMatrix(matrix.GetInvertMatrixElementsPointer(), 
														 matrix.GetMatrixElementsPointer(), 
														 matrix.GetNumberOfElements(),
														 matrix.GetSquareMatrixDimension());

	// Print out Matrix to Console
	std::cout << "Original Matrix:" << endl;
	std::cout << matrix.ToString() << endl;
	std::cout << "Inverted Matrix Result:" << endl;
	std::cout << matrix.InvertedMatrixToString() << endl;
	std::cout << "Time (ms): " << timeToCompleteInMs << endl;
}
