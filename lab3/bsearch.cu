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

__global__ void bSearchCuda(float *array, float search, int *index)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(array[x]==search)
        *index = x;
}


int main(int argc, char *argv[])
{

    float *data = new float[N];
    for(int i=0;i<N;i++)
        data[i] = i;

    float *deviceData;
    int *deviceIndex;
    size_t size = N*sizeof(float);
    cudaMalloc((void**)&deviceData, size);
	displayLastError("memory allocation");
    cudaMalloc((void**)&deviceIndex, sizeof(int));
	displayLastError("memory allocation");

	cudaMemcpy(deviceData, data, size, cudaMemcpyHostToDevice);
	displayLastError("memory copying");

    int index = -1;
	cudaMemcpy(deviceIndex, &index, sizeof(int), cudaMemcpyHostToDevice);




    int blocks = N / BLOCK_SIZE;
    if(N % BLOCK_SIZE)
		blocks++;

    bSearchCuda<<<blocks, BLOCK_SIZE>>>(deviceData, 100.0f, deviceIndex);
	displayLastError("kernel");


	cudaMemcpy(&index, deviceIndex, sizeof(int), cudaMemcpyDeviceToHost);
	displayLastError("memory copying");

    cout << index << endl;


    cudaFree(deviceData);
	displayLastError("free");
    cudaFree(deviceIndex);
	displayLastError("free");
    delete [] data;

    return 0;

}

