#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdlib>

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


int main()
{
    ifstream bmpFile("./lena.bmp", ios::in | ios::binary);
    if(!bmpFile.is_open())
    {
        cerr << "file not opened" << endl;
        exit(0);
    }
    
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
    
//    Pixel pixel;
    
    Pixel **image = new Pixel*[dib.height];
    
    for(int y = dib.height - 1; y >= 0 ; y--)
    {
        image[y] = new Pixel[dib.width];
        for(int x = 0; x < dib.width; x++)
        {
            cout << bmpFile.tellg() << endl;
//            bmpFile.read((char*)&pixel, sizeof(pixel));
            bmpFile.read((char*)&(image[y][x]), sizeof(Pixel));
//            cout << (int)image[y][x].r << '\t' << (int)image[y][x].g << '\t' << (int)image[y][x].b << '\t' << endl;
        }
        bmpFile.seekg(dib.width, ios_base::cur);
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    bmpFile.close();
    
    cout << endl;
    return 0;
    
}