#!/usr/bin/python
import sys

print '''
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <iostream>
#include <cstdlib>


#define BLOCK_SIZE 512
#define N 100
#define MUL 9999
#define NUMBER_TYPE unsigned int

using namespace std;

void displayLastError(const string &msg)
{
	//cout << "Last Error (" << msg << "):\t" << cudaGetErrorString(cudaGetLastError()) << endl;
}

template<class numType>
__global__ void kernel(numType *array)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x < N)
    {
        numType num = array[x];
#pragma unroll''' , sys.argv[1] , '''
        for(int i = 0;i < MUL-1;i++)
            array[x] += (i%2==0)?num:0;
    }
}


int main(int argc, char *argv[])
{

    NUMBER_TYPE *data = new NUMBER_TYPE[N];
    for(int i=0;i<N;i++)
        data[i] = i;

    NUMBER_TYPE *deviceData;
    size_t size = N*sizeof(NUMBER_TYPE);
    cudaMalloc((void**)&deviceData, size);
	displayLastError("memory allocation");

	cudaMemcpy(deviceData, data, size, cudaMemcpyHostToDevice);
	displayLastError("memory copying");


    int blocks = N / BLOCK_SIZE;
    if(N % BLOCK_SIZE)
		blocks++;

    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
    for(int t=0;t<1000;t++)
        kernel<NUMBER_TYPE><<<blocks, BLOCK_SIZE>>>(deviceData);
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	displayLastError("kernel");

    float kernelTime = 0;
	cudaEventElapsedTime(&kernelTime, start, stop);
    cout << /*"kernel execution time:\t" <<*/ kernelTime/1000.0 << endl;

	cudaMemcpy(data, deviceData, size, cudaMemcpyDeviceToHost);
	displayLastError("memory copying");

    cudaFree(deviceData);
	displayLastError("free");

    //for(int i = 0;i<N;i++)
    //    cout << data[i] << endl;


    delete [] data;


	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    return 0;
}
'''
