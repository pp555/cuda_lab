#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>

#include <ctime>
#include <unistd.h>
#include <sys/time.h>


#define INPUT_BMP_FILE "lenabig.bmp"
#define OUTPUT_BMP_FILE_CPU "result-cpu.bmp"
#define OUTPUT_BMP_FILE_GPU "result-cuda.bmp"
#define FILTER_WINDOW_SIZE 9

#define BLOCK_SIZE 64

using namespace std;

//#pragma pack(push)
//#pragma pack(1)
struct FileHeader
{
    char id[2];
    unsigned int size;
    int sth;
    unsigned int offset;

} __attribute__ ((packed));

struct DibHeader
{
    unsigned int dib_size;
    unsigned int width;
    unsigned int height;
} __attribute__ ((packed));

struct Pixel
{
    unsigned char b;
    unsigned char g;
    unsigned char r;
} __attribute__ ((packed));
//#pragma pack(pop)

struct Times // execution times [ms]
{
	float cuda;
	float cudaOnlyComputation;
	double cpu;
} executionTimes;

void medianFilterCpu(Pixel *input, Pixel *output, int width, int height);
Pixel medianFilterCpuHelper(Pixel *image, int width, int height, int y, int x);
unsigned char selection(unsigned char window[FILTER_WINDOW_SIZE]);
void medianFilterGpu(Pixel *input, Pixel *output, int width, int height);
__global__ void medianFilterCuda(Pixel *input, Pixel *output, int width, int height);
__device__ unsigned char selectionCuda(unsigned char window[FILTER_WINDOW_SIZE]);
void displayLastError(const string &msg);

//get time (for CPU)
double get_timestamp()
{
	struct timeval now;
	gettimeofday(&now, NULL);
	unsigned long long timeMicroseconds = now.tv_usec + now.tv_sec*1000000;
	return (double)timeMicroseconds/1000.0;
}

int main(int argc, char *argv[])
{
	//if(argc < 2)
	//{
	//	cerr << "no input file specified" <<  endl;
	//	return 1;
	//}
	//opening input file
    ifstream bmpFile(/*argv[1]*/INPUT_BMP_FILE, ios::in | ios::binary);
    if(!bmpFile.is_open())
    {
        cerr << "file not opened" << endl;
        exit(0);
    }

    //reading file headers
    FileHeader header;
    bmpFile.read((char*)&header, sizeof(header));

    cout << "size:\t" << header.size << endl;
    cout << "offset:\t" << header.offset << endl;

    DibHeader dib;
    bmpFile.read((char*)&dib, sizeof(dib));
    cout << "DIB size:\t" << dib.dib_size << endl;
    cout << "width:\t" << dib.width << endl;
    cout << "height:\t" << dib.height << endl;

    bmpFile.seekg(header.offset, ios_base::beg);

    //reading image
    Pixel *image = new Pixel[dib.height*dib.width];
    for(int y = dib.height - 1; y >= 0 ; y--)
    {
        for(int x = 0; x < dib.width; x++)
            bmpFile.read((char*)&(image[y*dib.width + x]), sizeof(Pixel));
        bmpFile.seekg(dib.width%4, ios_base::cur);
    }

    int devCnt;
    cudaGetDeviceCount(&devCnt);
    if(devCnt == 0)
    {
		perror("No CUDA devices available -- exiting.");
		return 1;
    }
	
    Pixel *outputCpu = new Pixel[dib.width*dib.height];
    Pixel *outputGpu = new Pixel[dib.width*dib.height];

int num = 1;
Times avgTimes = {0.0, 0.0, 0.0};
for(int i=0;i<num;i++)
{

    medianFilterGpu(image, outputGpu, dib.width, dib.height);
    medianFilterCpu(image, outputCpu, dib.width, dib.height);

    avgTimes.cpu += executionTimes.cpu/(float)num;
    avgTimes.cuda += executionTimes.cuda/(float)num;
    avgTimes.cudaOnlyComputation += executionTimes.cudaOnlyComputation/(float)num;


}
	cout << "times:\n";
	cout << "CPU:\t\t" << avgTimes.cpu << endl;
	cout << "GPU:\t\t" << avgTimes.cuda << endl;
	cout << "GPU (computation):\t\t" << avgTimes.cudaOnlyComputation << endl;




    //saving result bmp
    ofstream bmpResultCpu(OUTPUT_BMP_FILE_CPU, ios::out | ios::binary);
    ofstream bmpResultGpu(OUTPUT_BMP_FILE_GPU, ios::out | ios::binary);
    char *buf = new char[header.offset];
    char zerosBuf[4] = {0};
    bmpFile.seekg(0, ios_base::beg);
    bmpFile.read(buf, header.offset);
    bmpResultCpu.write(buf, header.offset);
    bmpResultGpu.write(buf, header.offset);
	delete [] buf;

    for(int y = dib.height - 1; y >= 0 ; y--)
    {
        for(int x = 0; x < dib.width; x++)
        {
            bmpResultCpu.write((char*)&(outputCpu[y*dib.width + x]), sizeof(Pixel));
            bmpResultGpu.write((char*)&(outputGpu[y*dib.width + x]), sizeof(Pixel));
        }
        bmpResultCpu.write(zerosBuf, dib.width%4);
        bmpResultGpu.write(zerosBuf, dib.width%4);
    }
	
    bmpResultCpu.close();
    bmpResultGpu.close();
    bmpFile.close();

    cout << endl;

	delete [] image;
	delete [] outputCpu;
	delete [] outputGpu;

    return 0;
}

void medianFilterCpu(Pixel *input, Pixel *output, int width, int height)
{
	double start = get_timestamp();

    for(int y = height - 1; y >= 0 ; y--)
    {
        for(int x = 0; x < width; x++)
        {
			output[y*width+x] = medianFilterCpuHelper(input, width, height, y, x);
        }
    }
	
	executionTimes.cpu = get_timestamp() - start;
}

Pixel medianFilterCpuHelper(Pixel *image, int width, int height, int y, int x)
{
	if(x < 1 || y < 1 || x >= width || y >= height)
		return image[y*width+x];

	Pixel p;
	unsigned char window[FILTER_WINDOW_SIZE];

	//red
	
	window[0] = image[(y-1)%height*width+(x-1)%width].r;
	window[1] = image[(y-1)%height*width+(x)%width].r;
	window[2] = image[(y-1)%height*width+(x+1)%width].r;
	window[3] = image[(y)%height*width+(x-1)%width].r;
	window[4] = image[(y)%height*width+(x)%width].r;
	window[5] = image[(y)%height*width+(x+1)%width].r;
	window[6] = image[(y+1)%height*width+(x-1)%width].r;
	window[7] = image[(y+1)%height*width+(x)%width].r;
	window[8] = image[(y+1)%height*width+(x+1)%width].r;
	p.r = selection(window);
	
	//green
	window[0] = image[(y-1)%height*width+(x-1)%width].g;
	window[1] = image[(y-1)%height*width+(x)%width].g;
	window[2] = image[(y-1)%height*width+(x+1)%width].g;
	window[3] = image[(y)%height*width+(x-1)%width].g;
	window[4] = image[(y)%height*width+(x)%width].g;
	window[5] = image[(y)%height*width+(x+1)%width].g;
	window[6] = image[(y+1)%height*width+(x-1)%width].g;
	window[7] = image[(y+1)%height*width+(x)%width].g;
	window[8] = image[(y+1)%height*width+(x+1)%width].g;
	p.g = selection(window);
	
	//blue
	window[0] = image[(y-1)%height*width+(x-1)%width].b;
	window[1] = image[(y-1)%height*width+(x)%width].b;
	window[2] = image[(y-1)%height*width+(x+1)%width].b;
	window[3] = image[(y)%height*width+(x-1)%width].b;
	window[4] = image[(y)%height*width+(x)%width].b;
	window[5] = image[(y)%height*width+(x+1)%width].b;
	window[6] = image[(y+1)%height*width+(x-1)%width].b;
	window[7] = image[(y+1)%height*width+(x)%width].b;
	window[8] = image[(y+1)%height*width+(x+1)%width].b;
	p.b = selection(window);
	
	return p;
}

unsigned char selection(unsigned char window[FILTER_WINDOW_SIZE])
{
	//http://en.wikipedia.org/wiki/Selection_algorithm
	unsigned char minIndex, minValue;
	for(int i = 0; i < FILTER_WINDOW_SIZE / 2; i++)
	{
		minIndex = i;
		minValue = window[i];
		for(int j = i + 1; j < FILTER_WINDOW_SIZE; j++)
		{
			if(window[j] < minValue)
			{
				minIndex = j;
				minValue = window[j];
			}
		}
		window[minIndex] = window[i]; 
		window[i] = minValue;
	}
	return window[FILTER_WINDOW_SIZE / 2];
}

void medianFilterGpu(Pixel *input, Pixel *output, int width, int height)
{
	cudaEvent_t start, stop, startComp, stopComp;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&startComp);
	cudaEventCreate(&stopComp);
	cudaEventRecord(start, 0);

	size_t size = width * height * sizeof(Pixel);
	Pixel *deviceInputImage;
	cudaMalloc((void**)&deviceInputImage, size);
	displayLastError("input image memory allocation");
	cudaMemcpy(deviceInputImage, input, size, cudaMemcpyHostToDevice);
	displayLastError("input image memcpy");

	Pixel *deviceOutputImage;
	cudaMalloc((void**)&deviceOutputImage, size);
	displayLastError("output image memory allocation");

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(width/BLOCK_SIZE, height/BLOCK_SIZE);
	
	cudaEventRecord(startComp, 0);
	medianFilterCuda<<<dimGrid, dimBlock>>>(deviceInputImage, deviceOutputImage, width, height);
	cudaEventRecord(stopComp, 0);
	cudaEventSynchronize(stopComp);
	cudaThreadSynchronize();
	displayLastError("kernel");

	cudaMemcpy(output, deviceOutputImage, size, cudaMemcpyDeviceToHost);
	displayLastError("output image memcpy");

	cudaFree(deviceInputImage);
	displayLastError("freeing input image memory");
	cudaFree(deviceOutputImage);
	displayLastError("freeing output image memory");
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&executionTimes.cuda, start, stop);
	cudaEventElapsedTime(&executionTimes.cudaOnlyComputation, startComp, stopComp);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

//kernel
__global__ void medianFilterCuda(Pixel *input, Pixel *output, int width, int height)
{
	Pixel result;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;	
	
	unsigned char window[FILTER_WINDOW_SIZE];

	//red
	window[0] = input[((row-1)%height*width)+(col-1)%width].r;
	window[1] = input[((row-1)%height*width)+(col)%width].r;
	window[2] = input[((row-1)%height*width)+(col+1)%width].r;
	window[3] = input[((row)%height*width)+(col-1)%width].r;
	window[4] = input[((row)%height*width)+(col)%width].r;
	window[5] = input[((row)%height*width)+(col+1)%width].r;
	window[6] = input[((row+1)%height*width)+(col-1)%width].r;
	window[7] = input[((row+1)%height*width)+(col)%width].r;
	window[8] = input[((row+1)%height*width)+(col+1)%width].r;
	result.r = selectionCuda(window);
	
	//green
	window[0] = input[((row-1)%height*width)+(col-1)%width].g;
	window[1] = input[((row-1)%height*width)+(col)%width].g;
	window[2] = input[((row-1)%height*width)+(col+1)%width].g;
	window[3] = input[((row)%height*width)+(col-1)%width].g;
	window[4] = input[((row)%height*width)+(col)%width].g;
	window[5] = input[((row)%height*width)+(col+1)%width].g;
	window[6] = input[((row+1)%height*width)+(col-1)%width].g;
	window[7] = input[((row+1)%height*width)+(col)%width].g;
	window[8] = input[((row+1)%height*width)+(col+1)%width].g;
	result.g = selectionCuda(window);
	
	//blue
	window[0] = input[((row-1)%height*width)+(col-1)%width].b;
	window[1] = input[((row-1)%height*width)+(col)%width].b;
	window[2] = input[((row-1)%height*width)+(col+1)%width].b;
	window[3] = input[((row)%height*width)+(col-1)%width].b;
	window[4] = input[((row)%height*width)+(col)%width].b;
	window[5] = input[((row)%height*width)+(col+1)%width].b;
	window[6] = input[((row+1)%height*width)+(col-1)%width].b;
	window[7] = input[((row+1)%height*width)+(col)%width].b;
	window[8] = input[((row+1)%height*width)+(col+1)%width].b;
	result.b = selectionCuda(window);
	
	output[row * width + col] = result;
}

__device__ unsigned char selectionCuda(unsigned char window[FILTER_WINDOW_SIZE])
{
	//http://en.wikipedia.org/wiki/Selection_algorithm
	unsigned char minIndex, minValue;
	for(int i = 0; i < FILTER_WINDOW_SIZE / 2; i++)
	{
		minIndex = i;
		minValue = window[i];
		for(int j = i + 1; j < FILTER_WINDOW_SIZE; j++)
		{
			if(window[j] < minValue)
			{
				minIndex = j;
				minValue = window[j];
			}
		}
		window[minIndex] = window[i]; 
		window[i] = minValue;
	}
	return window[FILTER_WINDOW_SIZE / 2];
}

void displayLastError(const string &msg)
{
//	cout << "Last Error (" << msg << "):\t" << cudaGetErrorString(cudaGetLastError()) << endl;
}
