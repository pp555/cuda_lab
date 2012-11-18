#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdlib>

#define INPUT_BMP_FILE "lena.bmp"
#define OUTPUT_BMP_FILE "result.bmp"
#define FILTER_WINDOW_SIZE 9

#define BLOCK_SIZE 16

using namespace std;

void displayLastError(const string &msg)
{
	cout << "Last Error (" << msg << "):\t" << cudaGetErrorString(cudaGetLastError()) << endl;
}

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

struct Image
{
	int width;
	int height;
	Pixel *pixels;
};

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
	dim3 dimGrid(width, height);
	medianFilterCuda<<<dimGrid, dimBlock>>>(deviceInputImage, deviceOutputImage, width, height);
	displayLastError("kernel");

	cudaMemcpy(output, deviceOutputImage, size, cudaMemcpyDeviceToHost);
	displayLastError("output image memcpy");


	cudaFree(deviceInputImage);
	displayLastError("freeing input image memory");
	cudaFree(deviceOutputImage);
	displayLastError("freeing output image memory");

}

//kernel
__global__ void medianFilterCuda(Pixel *input, Pixel *output, int width, int height)
{
	Pixel result;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;	
	
	unsigned char window[FILTER_WINDOW_SIZE];
/*
	//red
	int sum = 0;
	window[0] = input[((row-1)%height*width)+(col-1)%width].r;
	window[1] = input[((row-1)%height*width)+(col)%width].r;
	window[2] = input[((row-1)%height*width)+(col+1)%width].r;
	window[3] = input[((row)%height*width)+(col-1)%width].r;
	window[4] = input[((row)%height*width)+(col)%width].r;
	window[5] = input[((row)%height*width)+(col+1)%width].r;
	window[6] = input[((row+1)%height*width)+(col-1)%width].r;
	window[7] = input[((row+1)%height*width)+(col)%width].r;
	window[8] = input[((row+1)%height*width)+(col+1)%width].r;
	for(int i=0;i<9;i++)
		sum+=window[i];
	result.r = sum/9;
	
	//green
	sum = 0;
	window[0] = input[((row-1)%height*width)+(col-1)%width].g;
	window[1] = input[((row-1)%height*width)+(col)%width].g;
	window[2] = input[((row-1)%height*width)+(col+1)%width].g;
	window[3] = input[((row)%height*width)+(col-1)%width].g;
	window[4] = input[((row)%height*width)+(col)%width].g;
	window[5] = input[((row)%height*width)+(col+1)%width].g;
	window[6] = input[((row+1)%height*width)+(col-1)%width].g;
	window[7] = input[((row+1)%height*width)+(col)%width].g;
	window[8] = input[((row+1)%height*width)+(col+1)%width].g;
	for(int i=0;i<9;i++)
		sum+=window[i];
	result.g = sum/9;
	
	//blue
	sum = 0;
	window[0] = input[((row-1)%height*width)+(col-1)%width].b;
	window[1] = input[((row-1)%height*width)+(col)%width].b;
	window[2] = input[((row-1)%height*width)+(col+1)%width].b;
	window[3] = input[((row)%height*width)+(col-1)%width].b;
	window[4] = input[((row)%height*width)+(col)%width].b;
	window[5] = input[((row)%height*width)+(col+1)%width].b;
	window[6] = input[((row+1)%height*width)+(col-1)%width].b;
	window[7] = input[((row+1)%height*width)+(col)%width].b;
	window[8] = input[((row+1)%height*width)+(col+1)%width].b;
	for(int i=0;i<9;i++)
		sum+=window[i];
	result.b = sum/9;
	*/
	
	
	result.r = input[(row*width)+col].r;
	result.g = input[(row*width)+col].g;
	result.b = input[(row*width)+col].b;
	
	
	output[row * width + col] = result;
}



int main(int argc, char *argv[])
{
	if(argc < 2)
	{
		cerr << "no input file specified" <<  endl;
		return 1;
	}
	//opening input file
    ifstream bmpFile(argv[1], ios::in | ios::binary);
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

