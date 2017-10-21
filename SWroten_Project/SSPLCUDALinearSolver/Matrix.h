#pragma once

#include "matrixSolver.h"

#include <string>
#include <vector>

class Matrix
{
public:
	Matrix(int rowDim, int colDim);
	Matrix(std::vector<std::vector<double>> elements);
	~Matrix();
	void InvertCPU();
	void InvertGPU();
	float GetCPUTimeToInvertInMs() const;
	std::vector<double> GetElements() const;
	float GetGPUTimeToInvertInMs() const;
	std::string GetInvertedMatrixToString() const;
	double* GetInvertMatrixElementsPointer() const;
	std::string GetLUMatrixToString() const;
	double* GetLUMatrixElementsPointer() const;
	double* GetMatrixElementsPointer() const;
	int* GetPivotMatrixElementsPointer() const;
	int GetNumberOfElements() const;
	int GetSquareMatrixDimension() const;
	void PrintGPUProperties() const;
	std::string ToString() const;
private:
	std::string luMatrixAsString{ "" };
	std::string invertedMatrixAsString{ "" };
	std::string GetMatrixAsString(double *matrixElementsPntr, int squareMatrixDimension) const;
	int GetNumberOfElementsInMatrix(std::vector<std::vector<double>> matrix) const;
	int squareMatrixDimension{ 0 };
	int rowDimension{ 0 };
	int columnDimension{ 0 };
	int numberOfElements{ 0 };
	float cpuTimeToInvertInMs{ 0 };
	float gpuTimeToInvertInMs{ 0 };
	double *cpuMatrixElementsPntr;
	double *cpuLUMatrixElementsPntr;
	int *cpuPivotMatrixElementsPntr;
	double *cpuInvertedMatrixElementsPntr;
	std::vector<std::vector<double>> elements;
};

