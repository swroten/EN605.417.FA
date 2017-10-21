#pragma once

void PrintDeviceProperties();

void GetRandomNumbersForMatrix(double *cpuMatrix,
										 const int numberOfElements);

float GetInvertedMatrixCPU(double *cpuInvertedMatrix,
									const double *cpuLUMatrix,
									const int *cpuPivotMatrix,
									const int squareMatrixDimension);

float GetLUDecompositionMatrixCPU(double *cpuInvertedMatrix,
											 int *cpuPivotMatrix, 
											 const double *cpuMatrix,
											 const int numberOfElements,
											 const int squareMatrixDimension);

float GetInvertedMatrixGPU(double *cpuInvertedMatrix,
									const double *cpuLUMatrix,
									const int *cpuPivotMatrix,
									const int numberOfElements,
									const int squareMatrixDimension);

float GetLUDecompositionMatrixGPU(double *cpuInvertedMatrix,
											 int *cpuPivotMatrix,
											 const double *cpuMatrix, 
											 const int numberOfElements,
											 const int squareMatrixDimension);
