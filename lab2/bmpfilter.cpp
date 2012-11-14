#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdlib>

#define INPUT_BMP_FILE "lena.bmp"
#define OUTPUT_BMP_FILE "result.bmp"
#define FILTER_WINDOW_SIZE 9

using namespace std;


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


Pixel medianFilter(Pixel **image, int width, int height, int y, int x)
{
	if(x < 1 || y < 1 || x >= width || y >= height)
		return image[y][x];

	Pixel p;
	unsigned char window[FILTER_WINDOW_SIZE];

	//red
	window[0] = image[(y-1)%height][(x-1)%width].r;
	window[1] = image[(y-1)%height][(x)%width].r;
	window[2] = image[(y-1)%height][(x+1)%width].r;
	window[3] = image[(y)%height][(x-1)%width].r;
	window[4] = image[(y)%height][(x)%width].r;
	window[5] = image[(y)%height][(x+1)%width].r;
	window[6] = image[(y+1)%height][(x-1)%width].r;
	window[7] = image[(y+1)%height][(x)%width].r;
	window[8] = image[(y+1)%height][(x+1)%width].r;
	p.r = selection(window);
	
	//green
	window[0] = image[(y-1)%height][(x-1)%width].g;
	window[1] = image[(y-1)%height][(x)%width].g;
	window[2] = image[(y-1)%height][(x+1)%width].g;
	window[3] = image[(y)%height][(x-1)%width].g;
	window[4] = image[(y)%height][(x)%width].g;
	window[5] = image[(y)%height][(x+1)%width].g;
	window[6] = image[(y+1)%height][(x-1)%width].g;
	window[7] = image[(y+1)%height][(x)%width].g;
	window[8] = image[(y+1)%height][(x+1)%width].g;
	p.g = selection(window);
	
	//blue
	window[0] = image[(y-1)%height][(x-1)%width].b;
	window[1] = image[(y-1)%height][(x)%width].b;
	window[2] = image[(y-1)%height][(x+1)%width].b;
	window[3] = image[(y)%height][(x-1)%width].b;
	window[4] = image[(y)%height][(x)%width].b;
	window[5] = image[(y)%height][(x+1)%width].b;
	window[6] = image[(y+1)%height][(x-1)%width].b;
	window[7] = image[(y+1)%height][(x)%width].b;
	window[8] = image[(y+1)%height][(x+1)%width].b;
	p.b = selection(window);
	
	return p;
}


int main()
{
	//opening input file
    ifstream bmpFile(INPUT_BMP_FILE, ios::in | ios::binary);
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
    Pixel **image = new Pixel*[dib.height];
    for(int y = dib.height - 1; y >= 0 ; y--)
    {
        image[y] = new Pixel[dib.width];
        for(int x = 0; x < dib.width; x++)
            bmpFile.read((char*)&(image[y][x]), sizeof(Pixel));
        bmpFile.seekg(dib.width%4, ios_base::cur);
    }
    
    //displaying image in console
//    cout << 'x' << '\t' << 'y' << '\t' << 'r' << '\t' << 'g' << '\t' << 'b' << '\t' << endl;
//    for(int y = 0; y < dib.height ; y++)
//    {
//        for(int x = 0; x < dib.width; x++)
//        {
//            cout << x << '\t' << y << '\t' << (int)image[y][x].r << '\t' << (int)image[y][x].g << '\t' << (int)image[y][x].b << '\t' << endl;
//        }
//    }

    //gray-scale
//    for(int y = 0; y < dib.height ; y++)
//    {
//        for(int x = 0; x < dib.width; x++)
//        {
//            char color = (image[y][x].r + image[y][x].g + image[y][x].b) / 3;
//            image[y][x].r = image[y][x].g = image[y][x].b = color;
//        }
//    }



    //median filter
    Pixel **newImage = new Pixel*[dib.height];
    
    for(int y = dib.height - 1; y >= 0 ; y--)
    {
        newImage[y] = new Pixel[dib.width];
        for(int x = 0; x < dib.width; x++)
        {
			newImage[y][x] = medianFilter(image, dib.width, dib.height, y, x);
        }

    }
    
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
            bmpResult.write((char*)&(newImage[y][x]), sizeof(Pixel));
        }
        bmpResult.write(zerosBuf, dib.width%4);
    }
    
    bmpResult.close();


    bmpFile.close();
    
    cout << endl;
    return 0;
    
}
