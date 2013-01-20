#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <string>


#define BLOCK_SIZE 128
#define N 1000
#define LOOP 9999
#define NUMBER_TYPE unsigned int

using namespace std;

void displayLastError(const string &msg)
{
	cout << "Last Error (" << msg << "):\t" << cudaGetErrorString(cudaGetLastError()) << endl;
}

template<class numType>
__global__ void kernel(numType *array)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x < N)
    {
        numType num = array[x];
#pragma unroll 1
        for(long long int i = 1;i < LOOP;i*=2)
            array[x] = num*2+3*i/1000.0*i/1000.0;
    }
}

__global__ void kernel_unroll(NUMBER_TYPE *array)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x < N)
    {
        NUMBER_TYPE num = array[x];
#pragma unroll 100
        for(long long int i = 1;i < LOOP;i*=2)
            array[x] = num*2+3*i/1000.0*i/1000.0;
    }
}

__global__ void kernel_unroll2(NUMBER_TYPE *array)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(x < N)
    {
        NUMBER_TYPE num = array[x];
#pragma unroll 10
        for(long long int i = 1;i < LOOP;i*=2)
            array[x] = num*2+3*i/1000.0*i/1000.0;
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

	//bez unroll
	cudaEventRecord(start, 0);
    for(int t=0;t<100;t++)
        kernel<NUMBER_TYPE><<<blocks, BLOCK_SIZE>>>(deviceData);
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	displayLastError("kernel");

    float kernelTime = 0;
	cudaEventElapsedTime(&kernelTime, start, stop);
    cout << "kernel execution time:\t" << kernelTime << endl;
	
	//unroll 100
	cudaEventRecord(start, 0);
    for(int t=0;t<100;t++)
        kernel_unroll<<<blocks, BLOCK_SIZE>>>(deviceData);
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	displayLastError("kernel");

    kernelTime = 0;
	cudaEventElapsedTime(&kernelTime, start, stop);
    cout << "kernel execution time:\t" << kernelTime << endl;
	
	//unroll 10
	cudaEventRecord(start, 0);
    for(int t=0;t<100;t++)
        kernel_unroll2<<<blocks, BLOCK_SIZE>>>(deviceData);
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	displayLastError("kernel");

    kernelTime = 0;
	cudaEventElapsedTime(&kernelTime, start, stop);
    cout << "kernel execution time:\t" << kernelTime << endl;





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

