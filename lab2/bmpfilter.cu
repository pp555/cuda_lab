
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>

#define INPUT_BMP_FILE "lenabig.bmp"
#define OUTPUT_BMP_FILE "result.bmp"
#define FILTER_WINDOW_SIZE 9

#define BLOCK_SIZE 16

using namespace std;




void displayLastError(const string &msg)
{
	cout << "Last Error (" << msg << "):\t" << cudaGetErrorString(cudaGetLastError()) << endl;
}

#pragma pack(push)
#pragma pack(1)
struct FileHeader
{
    char id[2];
    unsigned int size;
    int sth;
    unsigned int offset;

};// __attribute__ ((packed));

struct DibHeader
{
    unsigned int dib_size;
    unsigned int width;
    unsigned int height;
};// __attribute__ ((packed));

struct Pixel
{
    unsigned char b;
    unsigned char g;
    unsigned char r;
};// __attribute__ ((packed));
#pragma pack(pop)



__global__ void medianFilterCuda(Pixel *input, Pixel *output, int width, int height);


void medianFilterGpu(Pixel *input, Pixel *output, int width, int height)
{

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
	medianFilterCuda<<<dimGrid, dimBlock>>>(deviceInputImage, deviceOutputImage, width, height);
	displayLastError("kernel");

	cudaMemcpy(output, deviceOutputImage, size, cudaMemcpyDeviceToHost);
	displayLastError("output image memcpy");


	cudaFree(deviceInputImage);
	displayLastError("freeing input image memory");
	cudaFree(deviceOutputImage);
	displayLastError("freeing output image memory");

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

	
	
	

    Pixel *output = new Pixel[dib.width*dib.height];
    for(long long i=0;i< dib.width*dib.height;i++)
{
    	output[i].r = 255;
    	output[i].g = 0;
    	output[i].b = 255;

}



    medianFilterGpu(image, output, dib.width, dib.height);






    //saving result bmp
    ofstream bmpResult(OUTPUT_BMP_FILE, ios::out | ios::binary);
    char *buf = new char[header.offset];
    char zerosBuf[4] = {0};
    bmpFile.seekg(0, ios_base::beg);
    bmpFile.read(buf, header.offset);
    bmpResult.write(buf, header.offset);

    for(int y = dib.height - 1; y >= 0 ; y--)
    {
        for(int x = 0; x < dib.width; x++)
        {
            bmpResult.write((char*)&(output[y*dib.width + x]), sizeof(Pixel));
        }
        bmpResult.write(zerosBuf, dib.width%4);
    }

    bmpResult.close();
    bmpFile.close();

    cout << endl;
    return 0;

}
