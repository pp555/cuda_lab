#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <assert.h>
#include <cuda.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cuda_texture_types.h>
#include <string>
#include <cuda_texture_types.h>
#define INPUT_BMP_FILE "lenabig.bmp"
#define OUTPUT_BMP_FILE "result-cuda2.bmp"
#define FILTER_WINDOW_SIZE 9
typedef float real;
#define BLOCK_SIZE 16

using namespace std;

void displayLastError(const string &msg)
{
	cout << "Last Error (" << msg << "):\t" << cudaGetErrorString(cudaGetLastError()) << endl;
}


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
struct Times // execution times [ms]
{
	float cuda;
	float cudaOnlyComputation;
	double cpu;
} executionTimes;
//#pragma pack(pop)
void PixelToFloatArray(Pixel *pixel, int w, int h, float *&texArr)
{
	texArr = new float[w*h*4];
	for(int i = 0; i<w*h; i++)
	{
		texArr[i*4]   = (float)(pixel[i].r) / 255.0f;
		texArr[i*4+1] = (float)(pixel[i].g) / 255.0f;
		texArr[i*4+2] = (float)(pixel[i].b) / 255.0f;
		texArr[i*4+3] = 0.0f;
	}
}
__global__ void medianFilterCudaTexture(cudaArray *input, Pixel *output, int width, int height, float pixelWidth);

size_t pitch;
texture<float4 , 2, cudaReadModeElementType> tex;
void medianFilterGpu(Pixel *input, Pixel *output, int width, int height)
{
	cudaEvent_t start, stop, startComp, stopComp;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&startComp);
	cudaEventCreate(&stopComp);
	cudaEventRecord(start, 0);
	
	float4 a;
	float *textureData;
	float *pitchPtr;
	float *cudaMemoryPtr;
	size_t pitch;
	size_t size = width*height;
	cudaArray *inArr;
	cudaArray *outArr;
	Pixel *outData;
	//zamiana pixeli na liniowa tablice float
	PixelToFloatArray(input, width, height, textureData);
	//format kanalow rgba (32 bity na skaldowa)
	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	displayLastError("channel format creation");
	//alokacja cudaArray na urzadzeniu (width x height x ilosc skladowych w formacie(4)
	cudaMallocArray(&inArr, &channelDesc, width, height, cudaArraySurfaceLoadStore);
	displayLastError("array alloc");
	//alokacja pamieci na pixele - wyjscie z urzadzenia
	cudaMalloc((void**)&outData, size*sizeof(Pixel));
	displayLastError("cuda malloc out");
	//kopiowanie pixeli do cudaArray (sizeof(float4) - 4 kanaly rgba)
	cudaMemcpyToArray(inArr, 0, 0, textureData, size*sizeof(float4), cudaMemcpyHostToDevice);
	displayLastError("array copy to device");
	//ustawienia stanu samplera
	//cudaAdressModeWrap - wyjscie poza wspolrzedne tekstury powoduja kopiowanie wartosci z
	//wspolrzednych brzegowych (tj. dla -0.6 wartosc jest czytana ze wspl 0.0, dla 1.2 z 1.0)
		tex.addressMode[0] = cudaAddressModeWrap;
		tex.addressMode[1] = cudaAddressModeWrap;
		tex.addressMode[2] = cudaAddressModeWrap;
		//filtr punktowy
		tex.filterMode = cudaFilterModePoint;
		tex.normalized = true;
	//powiazanie macierzy tekstury z samplerem
	cudaBindTextureToArray(&tex, inArr, &channelDesc);
	displayLastError("texture bind");
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(height/BLOCK_SIZE, height/BLOCK_SIZE);
	cudaEventRecord(startComp, 0);
	medianFilterCudaTexture<<<dimGrid, dimBlock>>>(inArr, outData, width, height, 1.0f/width);
	cudaEventRecord(stopComp, 0);
	cudaEventSynchronize(stopComp);
	displayLastError("kernel");
	cudaMemcpy(output, outData, size*sizeof(Pixel), cudaMemcpyDeviceToHost);
	displayLastError("copy out");
	cudaFreeArray(inArr);
	displayLastError("free array");
	cudaFree(outData);
	displayLastError("free output");
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&executionTimes.cuda, start, stop);
	cudaEventElapsedTime(&executionTimes.cudaOnlyComputation, startComp, stopComp);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
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
__global__ void medianFilterCudaTexture(cudaArray *input, Pixel *output, int width, int height, float pixelWidth)
{
	//tablica pomocnicza
	float pixelKernel[] =
	{
		-pixelWidth, -pixelWidth,
		-pixelWidth, 0,
		-pixelWidth, pixelWidth,
		0          , -pixelWidth,
		0          , 0,
		0          , pixelWidth,
		pixelWidth , -pixelWidth,
		pixelWidth , 0,
		pixelWidth , pixelWidth
	};
	unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
	//zamiana wspolrzednych obrazu na wspolrzedne tekstury
	float u = row / (float)width;
	float v = col / (float)height;

	//sobel w x
	//float4 sum;
	//sum.x = 0;
	//sum.y = 0;
	//sum.z = 0;
	//sum.w = 0;
	//float okno[] =
	//{
	//	-1, 0, 1,
	//	-2, 0, 2,
	//	-1, 0, 1
	//};
	//for(int i = 0; i<9; i++)
	//	{
	//		float4 a = tex2D(tex, u+pixelKernel[i*2], v+pixelKernel[i*2+1]);
	//		sum.x += a.x * okno[i];
	//		sum.y += a.y * okno[i];
	//		sum.z += a.z * okno[i];
	//		sum.w += a.w * okno[i];
	//	}
	//sum.x /= 9.0f;
	//sum.y /= 9.0f;
	//sum.z /= 9.0f;
	//sum.w /= 9.0f;

	//output[col * width + row].r = sum.x;
	//output[col * width + row].g = sum.y;
	//output[col * width + row].b = sum.z;

	//filtr medianowy
	unsigned char windowR[FILTER_WINDOW_SIZE];
	unsigned char windowG[FILTER_WINDOW_SIZE];
	unsigned char windowB[FILTER_WINDOW_SIZE];
	for(int i = 0; i<9; i++)
	{
		//odczytanie wartosci z rejestru teksturowego z podanego samplera
		float4 a = tex2D(tex, u+pixelKernel[i*2], v+pixelKernel[i*2+1]);
		//tworzenie tablic do szukania mediany
		windowR[i] = (int)255*a.x;
		windowG[i] = (int)255*a.y;
		windowB[i] = (int)255*a.z;
	}
	//szukanie mediany dla kazdej skladowej obrazu
	output[col * width + row].r = selectionCuda(windowR);
	output[col * width + row].g = selectionCuda(windowG);
	output[col * width + row].b = selectionCuda(windowB);
}

int main(int argc, char *argv[])
{
	//if(argc < 2)
	//{
	// cerr << "no input file specified" << endl;
	// return 1;
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
Times avgTimes = {0.0, 0.0, 0.0};
int num = 100;
for(int i=0;i<num;i++)
{
    medianFilterGpu(image, output, dib.width, dib.height);
avgTimes.cuda += executionTimes.cuda/(float)num;
avgTimes.cudaOnlyComputation += executionTimes.cudaOnlyComputation/(float)num;
}
	cout << "times:\n";
	cout << "GPU:\t\t" << avgTimes.cuda << endl;
	cout << "GPU (computation):\t\t" << avgTimes.cudaOnlyComputation << endl;

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
