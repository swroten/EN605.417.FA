#pragma once

void PrintDeviceProperties();

float GetInvertedMatrixCPU(double *cpuInvertedMatrix,
									int *cpuPivotMatrix,
									const int squareMatrixDimension);

float GetLUDecompositionMatrixCPU(double *cpuInvertedMatrix,
											 int *cpuPivotMatrix, 
											 const double *cpuMatrix,
											 const int numberOfElements,
											 const int squareMatrixDimension);

float GetInvertedMatrixGPU(double *cpuInvertedMatrix,
									int *cpuPivotMatrix,
									const int squareMatrixDimension);

float GetLUDecompositionMatrixGPU(double *cpuInvertedMatrix,
											 int *cpuPivotMatrix,
											 const double *cpuMatrix, 
											 const int numberOfElements,
											 const int squareMatrixDimension);
