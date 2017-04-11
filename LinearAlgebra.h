#pragma once

#include <vector>
#include <math.h>

#include "Matrix.h"
#include "LinearAlgebraCUDA.cuh"


//Tests needed to perform linear algebra operations safely.
template<typename T> bool areMatricesMultiplicable(const Matrix<T>& x, const Matrix<T>& y)
{
	if (x.cols_ == y.rows_)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

template<typename T> bool areMatricesSameSize(const Matrix<T>& x, const Matrix<T>& y)
{
	//Test that verifies if two matrices have the same structure.
	if ((x.rows_ == y.rows_ && x.cols_ == y.cols_) || (x.rows_ == y.cols_ && x.cols_ == y.rows_))
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

template<typename T> bool areMatricesExactSameSize(const Matrix<T>& x, const Matrix<T>& y)
{
	//Test that verifies if two matrices have the same number of rows and columns.
	if (x.rows_ == y.rows_ && x.cols_ == y.cols_)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

//Linear algebra operations.
template<typename T> void matricesDotProduct(const Matrix<T>& x, const Matrix<T>& y, Matrix<T>* out, const int& parall, int resize = 1)
{
	//Dot product between two matrices, with optional CUDA implementation.
	if (resize == 1)
	{
		out->resize(x.rows_, y.cols_);
	}

	if (parall == 0)
	{

		for (int z = 0; z < y.cols_; z++)
		{
			for (int w = 0; w < x.rows_; w++)
			{
				float sum = 0.0f;

				for (int e = 0; e < x.cols_; e++)
				{
					sum += x.elements_[w * x.cols_ + e] * y.elements_[e * y.cols_ + z];
				}

				out->elements_[w * out->cols_ + z] = sum;
			}
		}

	}
	else if (parall == 1)
	{
		matricesDotProductGPU(x.elements_, x.rows_, x.cols_, y.elements_, y.rows_, y.cols_, out->elements_);
	}
	else if (parall == 2)
	{
		matricesDotProductGPUSH(x.elements_, x.rows_, x.cols_, y.elements_, y.rows_, y.cols_, out->elements_);
	}

}
		
template<typename T> void matricesDifference(const Matrix<T>& x, const Matrix<T>& y, Matrix<T>* out, const int& parall, int resize = 1)
{
	//Difference between two matrices, with optional CUDA implementation.
	if (resize == 1)
	{
		out->resize(x.rows_, x.cols_);
	}

	if (parall == 0)
	{

		for (int i = 0; i < (out->rows_ * out->cols_); i++)
		{
			out->elements_[i] = x.elements_[i] - y.elements_[i];
		}

	}
	else if (parall == 1)
	{
		vectorsDiffGPU(x.elements_, y.elements_, (x.rows_ * x.cols_), out->elements_);
	}
	else if (parall == 2)
	{
		vectorsDiffGPUSH(x.elements_, y.elements_, (x.rows_ * x.cols_), out->elements_);
	}

}
		
template<typename T> void outerProduct(const Matrix<T>& x, const Matrix<T>& y, Matrix<T>* out, const int& parall, int resize = 1)
{
	//Outer product between two matrices, with optional CUDA implementation.
	if (resize == 1)
	{
		out->resize(x.rows_, y.rows_);
	}

	if (parall == 0)
	{

		for (int w = 0; w < out->rows_; w++)
		{
			for (int z = 0; z < out->cols_; z++)
			{
				out->elements_[w * out->cols_ + z] = x.elements_[w] * y.elements_[z];
			}
		}

	}
	else if (parall == 1)
	{
		outerProdGPU(x.elements_, x.rows_, y.elements_, y.rows_, out->elements_);
	}
	else if (parall == 2)
	{
		outerProdGPUSH(x.elements_, x.rows_, y.elements_, y.rows_, out->elements_);
	}

}

template<typename T> void transposeMatrix(const Matrix<T>& x, Matrix<T>* out, const int& parall, int resize = 1)
{
	//Transpose a matrix, with optional CUDA implementation.
	if (resize == 1)
	{
		out->resize(x.cols_, x.rows_);
	}

	if (parall == 0)
	{

		for (int w = 0; w < x.rows_; w++)
		{
			for (int z = 0; z < x.cols_; z++)
			{
				out->elements_[z * out->cols_ + w] = x.elements_[w * x.cols_ + z];
			}
		}

	}
	else if (parall == 1)
	{
		transposeGPU(x.elements_, x.rows_, x.cols_, out->elements_);
	}
	else if (parall == 2)
	{
		transposeGPUSH(x.elements_, x.rows_, x.cols_, out->elements_);
	}
}



