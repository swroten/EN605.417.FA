#pragma once

#include <string>
#include <vector>

class Matrix
{
public:
	Matrix(std::vector<std::vector<double>> elements);
	~Matrix();
	int GetNumberOfElements();
	std::string ToString();
private:
	int GetNumberOfElementsInMatrix(std::vector<std::vector<double>> matrix);
	int rowDimension{ 0 };
	int columnDimension{ 0 };
	int numberOfElements{ 0 };
	double *cpuMatrixPntr;
	double *cpuInvertedMatrixPntr;
	std::vector<std::vector<double>> elements;
};

