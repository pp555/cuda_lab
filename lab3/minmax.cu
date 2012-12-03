#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <iostream>
#include <cstdlib>


#define BLOCK_SIZE 64
#define N 64

using namespace std;

void displayLastError(const string &msg)
{
	cout << "Last Error (" << msg << "):\t" << cudaGetErrorString(cudaGetLastError()) << endl;
}

__global__ void minMaxCuda(float *array)
{
int nTotalThreads = blockDim.x;	// Total number of active threads
	while(nTotalThreads > 1)
	{
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.

		if (threadIdx.x < halfPoint)
		{
			float temp = array[threadIdx.x + halfPoint];

			if (temp < array[threadIdx.x]) array[threadIdx.x] = temp;

			
		}
		__syncthreads();

		nTotalThreads = (nTotalThreads >> 1);	// divide by two.

	}
}


int main(int argc, char *argv[])
{

    float *data = new float[N];
    for(int i=0;i<N;i++)
        data[i] = i;

    float *deviceData;
    float *deviceMax;
    size_t size = N*sizeof(float);
    cudaMalloc((void**)&deviceData, size);
	displayLastError("memory allocation");
    cudaMalloc((void**)&deviceMax, sizeof(float));
	displayLastError("memory allocation");

	cudaMemcpy(deviceData, data, size, cudaMemcpyHostToDevice);
	displayLastError("memory copying");

    int blocks = N / BLOCK_SIZE;
    if(N % BLOCK_SIZE)
		blocks++;

    minMaxCuda<<<blocks, BLOCK_SIZE>>>(deviceData);
	displayLastError("kernel");


    float max;
	cudaMemcpy(&max, deviceMax, sizeof(float), cudaMemcpyDeviceToHost);
	displayLastError("memory copying");


    cudaFree(deviceData);
	displayLastError("free");
    cudaFree(deviceMax);
	displayLastError("free");
    delete [] data;

    return 0;

}

