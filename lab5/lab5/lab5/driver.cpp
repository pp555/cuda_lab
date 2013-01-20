#include	<stdio.h>
#include	<assert.h> 
#include	<cuda.h>

#include <iostream>
#include <ctime>

#include 	"defines.h"

#define ALIGN_UP(offset, alignment)  (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

//funkcja liczaca wartosc wielomianu na CPU
double PolynomialCPU(float *a, float x, int n)
{
	float result = 0.0f;
	for(int i=0;i<n+1;i++)
	{
		result= (result * x) + a[n - i];
	}
	return result;
}

int main(int argc, char *argv[])
{
	srand(time(NULL));
	for(int k=0;k<4;k++)
	{
		int n = 30*(k+1);
		float x = ((float) rand()) / (float) RAND_MAX;
		float *a = new float[n+1];
		float resultGPU;

		for(int i = 0; i < n + 1; i++)
			a[i] = i * 0.5*((float) rand()) / (float) RAND_MAX;


		int blocks = (n + 1) / BLK_SZ;
		if((n + 1) % BLK_SZ)
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
		CALL( cuMemAlloc(&DevResult, sizeof(float) ) );

		CALL( cuMemcpyHtoD(DevA, a, (n+1)*sizeof(float)  ) );


		CALL( cuFuncSetBlockShape(hFunction, BLK_SZ, 1, 1) );


		//przekazanie parametrow do kernela
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


		//kopiowanie wyniku na hosta
		CALL( cuMemcpyDtoH((void *) &resultGPU, DevResult, sizeof(float) ) );


		//zwalnianie pamieci na urzadzeniu
		CALL( cuMemFree(DevA) );
		CALL( cuMemFree(DevResult) );


		//obliczenia na CPU
		float resultCPU = PolynomialCPU(a, x, n);


		std::cout << "GPU:\t" << resultGPU << std::endl;
		std::cout << "CPU:\t" << resultCPU << std::endl;
		std::cout << "roznica:\t" << fabs(resultGPU - resultCPU) << std::endl;
		
		delete [] a;
	}

	return 0;
}

