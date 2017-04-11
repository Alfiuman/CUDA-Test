#include <Windows.h>

#include "LinearAlgebra.h"
#include "Matrix.h"

#define CPU		0
#define GLOBAL	1
#define SHARED	2

int main(int argc, char ** argv)
{
	int parall = SHARED;
	int numRuns = 100;

	Matrix<float> prima(5000, 1000);
	Matrix<float> seconda(1000, 600);
	Matrix<float> risultato(1, 1);

	prima.assign(7.0f);
	seconda.assign(3.0f);
	risultato.assign(2.0f);

	std::cout << "\n\nstart\n\n";

	long int before = GetTickCount();

	for (int i = 0; i < numRuns; i++)
	{
		matricesDotProduct(prima, seconda, &risultato, parall);
		//transposeMatrix(prima, &risultato, parall);
		//outerProduct(prima, seconda, &risultato, parall);
		//matricesDifference(prima, seconda, &risultato, parall);
	}

	long int after = GetTickCount();
	std::cout << " timeTick was " << after - before << "\n\n";


	for (int i = 0; i < risultato.rows_ * risultato.cols_; i++)
	{
		std::cout << risultato.elements_[i] << "   " << i << std::endl;
	}

	std::cin.get();
	return 0;
}









