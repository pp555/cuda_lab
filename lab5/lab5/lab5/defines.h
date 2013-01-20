//#define		N	128
#define		BLK_SZ	128
#define CALL(x)	{int r=x;\
		if(r!=0){fprintf(stderr,"%s returned %d in line %d -- exiting.\n",#x,r,__LINE__);\
		exit(0);}} 
