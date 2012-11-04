#include <stdio.h>
#include <iostream>
#include <ctime>
#include <unistd.h>
#include <cmath>

#define  N		1000000
#define  BLOCK_SIZE	64
#define TIME_CHECK clock()/float(CLOCKS_PER_SEC)

using namespace std;

float 	   hArray[N];
float     *dArray;
int 	   blocks;


void prologue(void) {
   	cudaMalloc((void**)&dArray, N*sizeof(float));
}
void epilogue(void) {
	cudaMemcpy(hArray, dArray, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dArray);
}


// Kernel
__global__ void pi(float *arr) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;

	if(x < N)
	{
		double licznik = (x%2)?-1:1;
		double mianownik = 2*x+1;
		arr[x] = licznik/mianownik;
	}
}

int main(int argc, char** argv)
{
	float gpu_start_time = 0;
	float gpu_post_prologue_time = 0;
	float gpu_post_computing_time = 0;
	float gpu_end_time = 0;
	float cpu_start_time = 0;
	float cpu_end_time = 0;


	if(argc != 2)
		return -1;
    double eps = atof(argv[1]);
	double x0=1, x1=10000;
	double x2 = 1;
	cout << eps << endl;

//cpu
	cpu_start_time = TIME_CHECK;
	int i = 1;
	int mianownik = 1;
	while(abs(x2 - x1) > eps)
	{
		x1 = x2;
		int licznik = (i%2)?-1:1;
		i++;
		mianownik += 2;
		x2 = (float)licznik/(float)mianownik;
		x0 += x2;
	}
	x0 = 4 * x0;
	cpu_end_time = TIME_CHECK;
	printf("%.10f\n", x0);


//gpu
    int	 devCnt;
    cudaGetDeviceCount(&devCnt);
    if(devCnt == 0) {
		perror("No CUDA devices available -- exiting.");
		return 1;
    }
	gpu_start_time = TIME_CHECK;
    prologue();
    blocks = N / BLOCK_SIZE;
    if(N % BLOCK_SIZE)
		blocks++;

	gpu_post_prologue_time = TIME_CHECK;
    pi<<<blocks, BLOCK_SIZE>>>(dArray);
    cudaThreadSynchronize();

	gpu_post_computing_time = TIME_CHECK;
	epilogue();


	double sum = 0;
	for(int i=0;i<N;i++)
		sum += hArray[i];

	sum *= 4;
	gpu_end_time = TIME_CHECK;

	printf("%.10f\n", sum);

	cout << "prologue\t" <<  gpu_post_prologue_time - gpu_start_time << endl;
	cout << "counting\t" <<  gpu_post_computing_time - gpu_post_prologue_time << endl;
	cout << "epilogue\t" <<  gpu_end_time - gpu_post_computing_time << endl;
	cout << "cpu\t" <<  cpu_end_time - cpu_start_time << endl;

    return 0;
}
