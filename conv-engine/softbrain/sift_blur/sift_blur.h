#ifndef __SIFT_BLUR_H__
#define __SIFT_BLUR_H__

#include <math.h>
#include <time.h>

#define IMAGE_MUL(NUM,SIZE)     (int)ceil((double)((double)NUM / SIZE))
#define IMAGE_PADDING(NUM,SIZE)  ( IMAGE_MUL(NUM,SIZE) * SIZE ) 

#define INDEX_1D_HOR(I, J, K, COL, SIZE) ( (COL * I) \
																			    + ((K % SIZE) + J)				 							\
																			   )						

#define INDEX_1D_VER(I, J, K, ROW, SIZE) ( ((ROW * I) + (ROW * K)) \
																			    + J				 							\
																			   )						

//Image Structure
typedef struct __IMAGE{
	int rowx;
	int coly;
	int pad_rowx;
	int pad_coly;
	short * img_ptr;
}image;


//Get the updated padded image based on kernel size
void getPaddedRowCol(image* Image, int size, int dir){
	
		//Get the padded size for image based on the type of conv direction
		if(!dir){                                                 					//Horizantal
			Image->pad_coly = Image->coly + size - 1; 				// IMAGE COLS need to be multiple of kernel size (9,13,15)
			Image->pad_rowx = Image->rowx;																		//The number of rows may remain same
		}else{																															//Vertical
			Image->pad_rowx = Image->rowx + size - 1; 			// IMAGE ROWS need to be multiple of kernel size (9,13,15)
			Image->pad_coly = Image->coly;																	//The number of rows may remain same
		}
}


//Input Image generation  -- Generates image in multiple of stencil size 9, 13, 15 
void genImage(image* Image){
	int i, j, mem = 0;
	// srand(time(NULL));
	srand(1);

	for(i = 0; i < Image->pad_rowx; i++){
		
		for(j = 0; j < Image->pad_coly; j++){
		
			if( (j > (Image->coly - 1)) || (i > (Image->rowx - 1)) ){
				
				*(Image->img_ptr + mem) = 0;														//Zero padding
			} else {
			
				// *(Image->img_ptr + mem) = (rand() % 64);						//Values range from 0 to 255
				*(Image->img_ptr + mem) = (1 % 64);
			}
		
			mem++;
		}
	}
	fprintf(stderr, "Generated a zero-padded image of size %d x %d\n", Image->rowx, Image->coly);
}


//print the image to a file
void printImage(image* Image, FILE* fp){
	int i, j, mem = 0;

	for(i = 0; i < (Image->pad_rowx); i++){
		
		for(j = 0; j < (Image->pad_coly); j++){
			
				fprintf(fp, "%d\t", *(Image->img_ptr + mem));
				mem++;	

		}//end first for
		fprintf(fp, "\n");
	}

}


//1D Filter/Stencil Kernel generation -- For Gaussian blur etc.
void genKernel(short int* kernel, int size){
	// srand(time(NULL));
	srand(1);
	int i, mem = 0;

	for(i = 0; i < size; i++){
		
			*(kernel + mem) = (rand() % 64);								//Values range from 0 to 255 -- 8 bit co-eff
			*(kernel + mem) = (1 % 64);
			mem++;
		}
	
	fprintf(stderr, "\nGenerated a stencil kernel of size %d\n", size);
}

//print the stencil kernel to a file
void printKernel(short int* kernel, int size, FILE* fp){

	int i, mem = 0;

	for(i = 0; i < size; i++){
		
			fprintf(fp, "%d\t", *(kernel + mem));
			mem++;
	}

	fprintf(fp, "\n");
}





#endif
