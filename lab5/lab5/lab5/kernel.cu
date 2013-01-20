#include "defines.h"

extern "C" __global__ void Polynomial(float *result, float *a, float x, int n)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	__shared__ float resultPart[BLK_SZ+1];

	if(i < n + 1)
	{
		resultPart[i] = a[i] * pow(x, i); //obliczenie czesci wielomianu
	}
	__syncthreads();

	//sumowanie wszystkich czesci (w jednym watku)
	if(i == 0)
	{
		float w = 0.0f;
		for(int j = 0; j < n + 1; j++)
			w = w + resultPart[j];
		*result = w;
	}
	__syncthreads();
}
