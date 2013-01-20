#include "defines.h"

/*
extern "C" __global__ void IncVect(float *Tin, float *Tout) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if(x < N)
		Tout[x] = Tin[x] + 1;
}
*/

extern "C" __global__ void Polynomial(float *result, float *a, float x, int n)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < n + 1)
	{
		result[i] = a[i] * pow(x, i);
	}
}
