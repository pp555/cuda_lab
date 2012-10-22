#include <stdio.h>
#include <iostream>
#include <ctime>
#include <unistd.h>

using namespace std;

#define  N   		1000000
#define  BLOCK_SIZE	16
#define TIME_CHECK clock()/float(CLOCKS_PER_SEC)

float 	   hArray[N];
float     *dArray;
int 	   blocks;


void prologue(void) {
	memset(hArray, 0, sizeof(hArray));
	for(int i = 0; i < N; i++) {
		hArray[i] =  i + 1;
	}		
   	cudaMalloc((void**)&dArray, sizeof(hArray));
   	cudaMemcpy(dArray, hArray, sizeof(hArray), cudaMemcpyHostToDevice);
}

void epilogue(void) {
	cudaMemcpy(hArray, dArray, sizeof(hArray), cudaMemcpyDeviceToHost);
	cudaFree(dArray);
}


// Kernel
__global__ void pow3(float *A) {
	int x = blockDim.x * blockIdx.x + threadIdx.x; 

    if(x < N)
	    A[x] = A[x] * A[x] * A[x]; 
}

int main(int argc, char** argv)
{
    int	 devCnt;

    cudaGetDeviceCount(&devCnt);
    if(devCnt == 0) {
		perror("No CUDA devices available -- exiting.");
		return 1;
    }



	float gpu_start_time = 0;
	float gpu_post_prologue_time = 0;
	float gpu_post_computing_time = 0;
	float gpu_end_time = 0;
	float cpu_start_time = 0;
	float cpu_end_time = 0;

	gpu_start_time = TIME_CHECK;

    prologue();
    blocks = N / BLOCK_SIZE;
    if(N % BLOCK_SIZE)
		blocks++;

	gpu_post_prologue_time = TIME_CHECK;

    pow3<<<blocks, BLOCK_SIZE>>>(dArray);
    cudaThreadSynchronize();

	gpu_post_computing_time = TIME_CHECK;

    epilogue();

	gpu_end_time = TIME_CHECK;


sleep(1);

//cpu
	cpu_start_time = TIME_CHECK;
	for(long long i=0;i<N;i++)
		hArray[i] = hArray[i] * hArray[i] * hArray[i];
	cpu_end_time = TIME_CHECK;


	cout << gpu_start_time << endl;
	cout << gpu_post_prologue_time << endl;
	cout << gpu_post_computing_time << endl;
	cout << gpu_end_time << endl;

	cout << cpu_start_time << endl;
	cout << cpu_end_time << endl;

	cout << "prologue\t" <<  gpu_post_prologue_time - gpu_start_time << endl;
	cout << "counting\t" <<  gpu_post_computing_time - gpu_post_prologue_time << endl;
	cout << "epilogue\t" <<  gpu_end_time - gpu_post_computing_time << endl;


	cout << "cpu\t" <<  cpu_end_time - cpu_start_time << endl;
    

    return 0;
}
