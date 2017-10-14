#include "Matrix.h"

#include <sstream>
#include <algorithm>

Matrix::Matrix(std::vector<std::vector<double>> elements)
{
	// Set Variables
	this->elements = elements;
	this->numberOfElements = GetNumberOfElementsInMatrix(elements);
	this->rowDimension = std::sqrt(this->numberOfElements);
	this->columnDimension = std::sqrt(this->numberOfElements);

	// Allocate Memory for matrix & inverted matrix array pointer
	this->cpuMatrixPntr = (double *)malloc(this->numberOfElements * sizeof(double));
	this->cpuInvertedMatrixPntr = (double *)malloc(this->numberOfElements * sizeof(double));

	// Get all Elements in Matrix
	std::vector<double> allElements = GetElements();

	// Add elements to matrix
	for (int i = 0; i < this->numberOfElements; i++)
	{
		cpuMatrixPntr[i] = allElements[i];
		cpuInvertedMatrixPntr[i] = 0.0;
	}
}

Matrix::~Matrix()
{
}

double * Matrix::GetMatrixElementsPointer()
{
	return this->cpuMatrixPntr;
}

double * Matrix::GetInvertMatrixElementsPointer()
{
	return this->cpuInvertedMatrixPntr;
}

int Matrix::GetNumberOfElements()
{
	return this->numberOfElements;
}

int Matrix::GetSquareMatrixDimension()
{
	return std::min(this->rowDimension, this->columnDimension);
}

std::vector<double> Matrix::GetElements()
{
	// Initialize Variables
	std::vector<double> allElements;

	// Step through each row in matrix
	for each (std::vector<double> row in this->elements)
	{
		// Step through each column in this row
		for each (double column in row)
		{
			allElements.push_back(column);
		}
	}

	// return all elements
	return allElements;
}

std::string Matrix::ToString()
{
	// Initialize Variable
	std::ostringstream matrixAsStringStream;

	matrixAsStringStream << "Matrix = {" << std::endl;

	// Step through each row in matrix
	for each (std::vector<double> row in this->elements)
	{
		// Step through each column in this row
		for each (double column in row)
		{
			matrixAsStringStream << column << " ";
		}

		matrixAsStringStream << std::endl;
	}

	matrixAsStringStream << "};" << std::endl;

	// Return Matrix as String
	return matrixAsStringStream.str();
}

std::string Matrix::InvertedMatrixToString()
{
	// Initialize Variable
	std::ostringstream matrixAsStringStream;

	matrixAsStringStream << "Inverted Matrix = {" << std::endl;

	// Step through each row in matrix
	for (int i = 0; i < this->rowDimension; i++)
	{
		// Step through each column in this row
		for (int j = 0; j < this->columnDimension; j++)
		{
			matrixAsStringStream << this->cpuInvertedMatrixPntr[((i * j) + j)] << " ";
		}

		matrixAsStringStream << std::endl;
	}

	matrixAsStringStream << "};" << std::endl;

	// Return Matrix as String
	return matrixAsStringStream.str();
}

int Matrix::GetNumberOfElementsInMatrix(std::vector<std::vector<double>> matrix)
{
	// Initialize Variables
	int numberOfElements = 0;

	// Step through each row in matrix
	for each (std::vector<double> row in matrix)
	{
		// Step through each column in this row
		for each (double column in row)
		{
			// Increment total number of elements
			numberOfElements++;
		}
	}

	// return total number of elements in matrix
	return numberOfElements;
}
