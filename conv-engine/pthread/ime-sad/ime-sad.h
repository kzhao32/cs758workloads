#ifndef __IME_SAD_H__
#define __IME_SAD_H__

#include <math.h>
#include <time.h>

#define DEBUG 1

#define SIZE 4																								//STENCIL SIZE - 4 x 4
#define IMAGE_MUL(NUM)     (int)ceil((double)((double)NUM / SIZE))
#define IMAGE_PADDING(NUM)  ( IMAGE_MUL(NUM) * SIZE ) 

#define INDEX_2D(I, J, K, ROW, COL, SIZE) ( (IMAGE_PADDING(COL) * (I + (K/SIZE))) \
																			    + ((K % SIZE) + J)				 							\
																			    )						


//Input Image generation  -- Generates image in multiple of stencil size only for sad
void genImage(int* inputImage, int rowx, int coly){
	int i, j, mem = 0;
	
	for(i = 0; i < IMAGE_PADDING(rowx); i++){
		
		for(j = 0; j < IMAGE_PADDING(coly); j++){
		
			if( (j > (coly - 1)) || (i > (rowx - 1)) ){
				
				*(inputImage + mem) = 0;															//Zero padding
			} else {
			
				*(inputImage + mem) = (rand() % 255);						//Values range from 0 to 255
			
			}
		
			mem++;
		}
	}

	//fprintf(stderr, "Generated a zero-padded image of size %d x %d\n", rowx, coly);
}


//print the image to a file
void printImage(int* inputImage, int rowx, int coly, FILE* fp){
	int i, j, mem = 0;

	for(i = 0; i < IMAGE_PADDING(rowx); i++){
		
		for(j = 0; j < IMAGE_PADDING(coly); j++){
			
				fprintf(fp, "%d\t", *(inputImage + mem));
				mem++;	

		}//end first for
		fprintf(fp, "\n");
	}

}

#endif
