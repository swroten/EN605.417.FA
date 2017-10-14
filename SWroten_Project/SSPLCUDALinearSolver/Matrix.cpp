#include "Matrix.h"

#include <sstream>
#include <algorithm>

Matrix::Matrix(std::vector<std::vector<double>> elements)
{
	this->elements = elements;
	this->numberOfElements = GetNumberOfElementsInMatrix(elements);
}

Matrix::~Matrix()
{
}

int Matrix::GetNumberOfElements()
{
	return this->numberOfElements;
}

int Matrix::GetSquareMatrixDimension()
{
	return std::min(this->rowDimension, this->columnDimension);
}

std::string Matrix::ToString()
{
	// Initialize Variable
	std::ostringstream matrixAsStringStream;

	matrixAsStringStream << "Inverted Matrix = {" << std::endl;

	// Step through each row in matrix
	for each (std::vector<double> row in this->elements)
	{
		// Step through each column in this row
		for each (double column in row)
		{
			matrixAsStringStream << column << " " << std::endl;
		}
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
