#include	<stdio.h>
#include	<assert.h> 
#include	<cuda.h>

#include <iostream>

#include 	"defines.h"

#define ALIGN_UP(offset, alignment)  (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

double PolynomialCPU(float *a, float x, int n)
{
	double r = 0.0f;

	float *result = new float[n+1];

	for(int i=0;i<n+1;i++)
	{
		result[i] = a[i] * pow(x, i);
	}

	for(int i=0;i<n+1;i++)
		r += double(result[i]);

	delete [] result;

	return r;
}

int main(int argc, char *argv[])
{
	int i;
	int n = 3;
	float x = 1;
	//float a[] = {2, 3, 4};
	float *a = new float[n+1];
	float *result = new float[n+1];

	for(int i=0;i<n+1;i++)
		a[i] = i*0.1;


	int blocks = (n+1) / BLK_SZ;
    if((n+1) % BLK_SZ)
		blocks++;

	CUdevice	hDevice;
	CUcontext	hContext;
	CUmodule	hModule;
	CUfunction	hFunction;

	CALL( cuInit(0) );
	CALL( cuDeviceGet(&hDevice, 0) ); 	
	CALL( cuCtxCreate(&hContext, 0, hDevice) );
	CALL( cuModuleLoad(&hModule, "kernel.cubin") );
	CALL( cuModuleGetFunction(&hFunction, hModule, "Polynomial") );


	//dane wejsciowe - kopiowanie
	CUdeviceptr DevA, DevResult;

	CALL( cuMemAlloc(&DevA, (n+1)*sizeof(float) ) );
	CALL( cuMemAlloc(&DevResult, (n+1)*sizeof(float) ) );

	CALL( cuMemcpyHtoD(DevA, a, (n+1)*sizeof(float)  ) );


	CALL( cuFuncSetBlockShape(hFunction, BLK_SZ, 1, 1) );


	//przekazanie parametrów do kernela
	int 	offset = 0;
	void   *ptr;

	ptr = (void*)(size_t)DevResult;
	ALIGN_UP(offset, __alignof(ptr));
	CALL( cuParamSetv(hFunction, offset, &ptr, sizeof(ptr)) );
	offset += sizeof(ptr);

	ptr = (void*)(size_t)DevA;
	ALIGN_UP(offset, __alignof(ptr));
	CALL( cuParamSetv(hFunction, offset, &ptr, sizeof(ptr)) );
	offset += sizeof(ptr);

	ALIGN_UP(offset, __alignof(float));
	CALL( cuParamSetf(hFunction, offset, x) );
	offset += sizeof(float);

	ALIGN_UP(offset, __alignof(int));
	CALL( cuParamSeti(hFunction, offset, n) );
	offset += sizeof(int);


	CALL( cuParamSetSize(hFunction, offset) );

	CALL( cuLaunchGrid(hFunction, blocks, 1) );


	//kopiowanie danych na hosta
	CALL( cuMemcpyDtoH((void *) result, DevResult, (n+1)*sizeof(float) ) );


	CALL( cuMemFree(DevA) );
	CALL( cuMemFree(DevResult) );


	double s = 0.0;
	for(int i=0;i<n+1;i++)
	{
		//std::cout << x << '\t' << a[i] << '\t' <<  result[i] << std::endl;
		s += double(result[i]);
	}
	std::cout << "GPU:\t" << s << std::endl;



	std::cout << "CPU:\t" << PolynomialCPU(a, x, n) << std::endl;


	puts("done");

	delete [] a;
	delete [] result;

	return 0;
}

