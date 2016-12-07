#ifndef __IME_SAD_H__
#define __IME_SAD_H__

#include <math.h>
#include <time.h>

#define IMAGE_PADDING(NUM, SIZE)  (NUM + SIZE - 1)
#define ZERO_PADDING(NUM)  ((NUM - 1) / 2)

//Input Image generation
void genImage(int* inputImage, int rowx, int coly, int ker_size){
	int i, j, mem = 0;

	for(i = 0; i < IMAGE_PADDING(rowx, ker_size); i++){
		
		for(j = 0; j < IMAGE_PADDING(coly, ker_size); j++){
		
			if( (j < ZERO_PADDING(coly)) || (j > (coly + ZERO_PADDING(coly) - 1)) || 
					(i < ZERO_PADDING(rowx)) || (i > (rowx + ZERO_PADDING(rowx) - 1)) ){
				
				*(inputImage + mem) = 0;																									//Zero padding
			} else {
			
				*(inputImage + mem) = (rand() % 255);																				//Values range from 0 to 255
			
			}
		
			mem++;
		}
	}

	fprintf(stderr, "\nGenerated a zero-padded image of size %d x %d\n", rowx, coly);
}


//print the image to a file
void printImage(int* inputImage, int rowx, int coly, int ker_size, FILE* fp){
	int i, j, mem = 0;

	for(i = 0; i < IMAGE_PADDING(rowx, ker_size); i++){
		
		for(j = 0; j < IMAGE_PADDING(coly, ker_size); j++){
			
			if( (j < ZERO_PADDING(coly)) || (j > (coly + ZERO_PADDING(coly) - 1)) || 
					(i < ZERO_PADDING(rowx)) || (i > (rowx + ZERO_PADDING(rowx) - 1)) ){

				//do nothing -- Skip the padded zeros
				mem++;

			} else {
				
				fprintf(fp, "%d\t", *(inputImage + mem));
				mem++;	
			}

		}//end first for
		fprintf(fp, "\n");
	}

}

//Filter/Stencil Kernel generation -- For Gaussian blur etc.
void genKernel(int* kernel, int size){
	srand(time(NULL));
	int i, j, mem = 0;

	for(i = 0; i < size; i++){
		
		for(j = 0; j < size; j++){
			*(kernel + mem) = (rand() % 63);								//Values range from 0 to 255
			mem++;
		}
	}
	fprintf(stderr, "\nGenerated a stencil kernel of size %d x %d\n", size, size);
}

//print the stencil kernel to a file
void printKernel(int* kernel, int size, FILE* fp){

	int i, j, mem = 0;

	for(i = 0; i < size; i++){
		
		for(j = 0; j < size; j++){
			
			fprintf(fp, "%d\t", *(kernel + mem));
			mem++;
		}
		fprintf(fp, "\n");
	}
}




#endif
