#include "Matrix.h"

#include <sstream>
#include <algorithm>

Matrix::Matrix(int rowDim, int colDim)
{
	// Set Variables
	this->rowDimension = rowDim;
	this->columnDimension = colDim;
	this->elements = { std::vector<std::vector<double>>() };
	this->numberOfElements = this->rowDimension * this->columnDimension;
	this->squareMatrixDimension = std::min(this->rowDimension, this->columnDimension);
	
	// Allocate Memory for matrix & inverted matrix array pointer
	this->cpuPivotMatrixElementsPntr = (int *)malloc(this->squareMatrixDimension * sizeof(int));
	this->cpuMatrixElementsPntr = (double *)malloc(this->numberOfElements * sizeof(double));
	this->cpuLUMatrixElementsPntr = (double *)malloc(this->numberOfElements * sizeof(double));
	this->cpuInvertedMatrixElementsPntr = (double *)malloc(this->numberOfElements * sizeof(double));

	// Get Random Values for Elements
	GetRandomNumbersForMatrix(this->cpuMatrixElementsPntr, this->numberOfElements);

	// Add Random numbers to elements double vector
	for (int row = 0; row < this->rowDimension; row++)
	{
		// Add new vector for this row
		this->elements.push_back(std::vector<double>());

		// for each element
		for (int col = 0; col < this->columnDimension; col++)
		{
			// Add Randomly generated Element into CPU Matrix of elements
			this->elements[row].push_back(this->cpuMatrixElementsPntr[(row * this->squareMatrixDimension) + col]);
		}
	}
}

Matrix::Matrix(std::vector<std::vector<double>> elements)
{
	// Set Variables
	this->elements = elements;
	this->numberOfElements = GetNumberOfElementsInMatrix(elements);
	this->rowDimension = (int)std::sqrt(this->numberOfElements);
	this->columnDimension = (int)std::sqrt(this->numberOfElements);
	this->squareMatrixDimension = std::min(this->rowDimension, this->columnDimension);

	// Allocate Memory for matrix & inverted matrix array pointer
	this->cpuPivotMatrixElementsPntr = (int *)malloc(this->squareMatrixDimension * sizeof(int));
	this->cpuMatrixElementsPntr = (double *)malloc(this->numberOfElements * sizeof(double));
	this->cpuLUMatrixElementsPntr = (double *)malloc(this->numberOfElements * sizeof(double));
	this->cpuInvertedMatrixElementsPntr = (double *)malloc(this->numberOfElements * sizeof(double));

	// Get all Elements in Matrix
	std::vector<double> allElements = GetElements();	
	
	// Add elements to matrix
	for (int i = 0; i < this->numberOfElements; i++)
	{
		this->cpuMatrixElementsPntr[i] = allElements[i];
	}
}

Matrix::~Matrix()
{
}

std::string Matrix::GetInvertedMatrixToString() const
{
	return this->invertedMatrixAsString;
}

std::string Matrix::GetLUMatrixToString() const
{
	return this->luMatrixAsString;
}

void Matrix::InvertCPU()
{
	// Initialize Variables
	float timeToGetLUDecompositionMatrix;
	float timeToInvertMatrixFromLUDecompositionAndPivotMatrix;

	// Initialize Pivot Matrix
	for (int i = 0; i < this->squareMatrixDimension; i++)
	{
		this->cpuPivotMatrixElementsPntr[i] = i;
	}

	// Add elements to matrix
	for (int i = 0; i < this->numberOfElements; i++)
	{
		this->cpuLUMatrixElementsPntr[i] = this->cpuMatrixElementsPntr[i];
		this->cpuInvertedMatrixElementsPntr[i] = this->cpuMatrixElementsPntr[i];
	}

	// On the CPU - Perform LU Decomposition to get LU Matrix and Pivot Matrix - returns time required to complete in ms
	timeToGetLUDecompositionMatrix = GetLUDecompositionMatrixCPU(this->cpuLUMatrixElementsPntr,
		this->cpuPivotMatrixElementsPntr,
		this->cpuMatrixElementsPntr,
		this->numberOfElements,
		this->squareMatrixDimension);
	
	// Build LU Decomposition Matrix As String
	this->luMatrixAsString = GetMatrixAsString(this->cpuLUMatrixElementsPntr, this->squareMatrixDimension);

	// Compute LU Decomposition Matrix Magnitude
	this->luMatrixMagnitude = ComputeMagnitudeOfMatrix(this->cpuLUMatrixElementsPntr, this->numberOfElements);

	// On the CPU - Use the LU Matrix and Pivot Matrix to get Inverte Matrix - returns time required to complete in ms 
	timeToInvertMatrixFromLUDecompositionAndPivotMatrix = GetInvertedMatrixCPU(this->cpuInvertedMatrixElementsPntr,
		this->cpuLUMatrixElementsPntr,
		this->cpuPivotMatrixElementsPntr,
		this->squareMatrixDimension);
	
	// Build Inverted Matrix As String
	this->invertedMatrixAsString = GetMatrixAsString(this->cpuInvertedMatrixElementsPntr, this->squareMatrixDimension);

	// Compute LU Decomposition Matrix Magnitude
	this->inverseMatrixMagnitude = ComputeMagnitudeOfMatrix(this->cpuInvertedMatrixElementsPntr, this->numberOfElements);

	// Accumulate all Time Required to invert Matrix on cpu
	this->cpuTimeToInvertInMs = timeToGetLUDecompositionMatrix + timeToInvertMatrixFromLUDecompositionAndPivotMatrix;
}

void Matrix::InvertGPU()
{
	// Initialize Variables
	float timeToGetLUDecompositionMatrix;
	float timeToInvertMatrixFromLUDecompositionAndPivotMatrix;

	// Initialize Pivot Matrix
	for (int i = 0; i < this->squareMatrixDimension; i++)
	{
		this->cpuPivotMatrixElementsPntr[i] = i;
	}

	// Add elements to matrix
	for (int i = 0; i < this->numberOfElements; i++)
	{
		this->cpuLUMatrixElementsPntr[i] = this->cpuMatrixElementsPntr[i];
		this->cpuInvertedMatrixElementsPntr[i] = this->cpuMatrixElementsPntr[i];
	}

	// On the GPU - Perform LU Decomposition to get LU Matrix and Pivot Matrix - returns time required to complete in ms
	timeToGetLUDecompositionMatrix = GetLUDecompositionMatrixGPU(this->cpuLUMatrixElementsPntr,
																					 this->cpuPivotMatrixElementsPntr,
																					 this->cpuMatrixElementsPntr,
																					 this->numberOfElements,
																					 this->squareMatrixDimension);
	
	// Build LU Decomposition Matrix As String
	this->luMatrixAsString = GetMatrixAsString(this->cpuLUMatrixElementsPntr, this->squareMatrixDimension);

	// Compute LU Decomposition Matrix Magnitude
	this->luMatrixMagnitude = ComputeMagnitudeOfMatrix(this->cpuLUMatrixElementsPntr, this->numberOfElements);

	// On the GPU - Use the LU Matrix and Pivot Matrix to get Inverte Matrix - returns time required to complete in ms 
	timeToInvertMatrixFromLUDecompositionAndPivotMatrix = GetInvertedMatrixGPU(this->cpuInvertedMatrixElementsPntr,
																										this->cpuLUMatrixElementsPntr,
																										this->cpuPivotMatrixElementsPntr,
																										this->numberOfElements,
																										this->squareMatrixDimension);
	
	// Build Inverted Matrix As String
	this->invertedMatrixAsString = GetMatrixAsString(this->cpuInvertedMatrixElementsPntr, this->squareMatrixDimension);

	// Compute LU Decomposition Matrix Magnitude
	this->inverseMatrixMagnitude = ComputeMagnitudeOfMatrix(this->cpuInvertedMatrixElementsPntr, this->numberOfElements);

	// Accumulate all Time Required to invert Matrix on cpu
	this->gpuTimeToInvertInMs = timeToGetLUDecompositionMatrix + timeToInvertMatrixFromLUDecompositionAndPivotMatrix;
}

void Matrix::InvertWithCuBLASOnGPU()
{
	// Initialize Variables
	double *luDecompMatrix = (double*)malloc((sizeof(double)*this->numberOfElements));
	double *luDecompMatrixArray[] = { luDecompMatrix };
	double **cpuLUDecompositionMatrixArrayOfPointers = (double**)malloc(sizeof(luDecompMatrixArray));
	
	// Initialize Pivot Matrix
	for (int i = 0; i < this->squareMatrixDimension; i++)
	{
		this->cpuPivotMatrixElementsPntr[i] = i;
	}

	// Add elements to matrix
	for (int i = 0; i < this->numberOfElements; i++)
	{
		this->cpuLUMatrixElementsPntr[i] = this->cpuMatrixElementsPntr[i];
		this->cpuInvertedMatrixElementsPntr[i] = this->cpuMatrixElementsPntr[i];
	}

	this->gpuTimeToInvertInMs = GetCuSparseInvertedMatrixGPU(this->cpuInvertedMatrixElementsPntr,
																				this->cpuMatrixElementsPntr,
																				this->squareMatrixDimension);

	// Build Inverted Matrix As String
	this->invertedMatrixAsString = GetMatrixAsString(this->cpuInvertedMatrixElementsPntr, this->squareMatrixDimension);

	// Compute LU Decomposition Matrix Magnitude
	this->inverseMatrixMagnitude = ComputeMagnitudeOfMatrix(this->cpuInvertedMatrixElementsPntr, this->numberOfElements);
}

float Matrix::GetCPUTimeToInvertInMs() const
{
	return this->cpuTimeToInvertInMs;
}

std::vector<double> Matrix::GetElements() const
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

void Matrix::PrintGPUProperties() const
{
	PrintDeviceProperties();
}

float Matrix::GetGPUTimeToInvertInMs() const
{
	return this->gpuTimeToInvertInMs;
}

double * Matrix::GetInvertMatrixElementsPointer() const
{
	return this->cpuInvertedMatrixElementsPntr;
}

double * Matrix::GetMatrixElementsPointer() const
{
	return this->cpuMatrixElementsPntr;
}

double * Matrix::GetLUMatrixElementsPointer() const
{
	return this->cpuLUMatrixElementsPntr;
}

int * Matrix::GetPivotMatrixElementsPointer() const
{
	return this->cpuPivotMatrixElementsPntr;
}

int Matrix::GetNumberOfElements() const
{
	return this->numberOfElements;
}

std::string Matrix::GetMatrixAsString(double *matrixElementsPntr, int squareMatrixDimension) const
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

int Matrix::GetNumberOfElementsInMatrix(std::vector<std::vector<double>> matrix) const
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

int Matrix::GetSquareMatrixDimension() const
{
	return this->squareMatrixDimension;
}

double Matrix::GetMagnitudeOfInvertedMatrixElements() const
{
	return this->inverseMatrixMagnitude;
}

double Matrix::GetMagnitudeOfLUDecompositionMatrixElements() const
{
	return this->luMatrixMagnitude;
}

std::string Matrix::ToString() const
{
	return GetMatrixAsString(this->cpuMatrixElementsPntr, this->squareMatrixDimension);
}
