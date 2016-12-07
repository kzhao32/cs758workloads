#ifndef __SIFT_DOG_H__
#define __SIFT_DOG_H__

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

#define INDEX_SUB()

//Image Structure
typedef struct __IMAGE{
	int rowx;
	int coly;
	int pad_rowx;
	int pad_coly;
	int* img_ptr;
}image;


//Get the updated padded image based on kernel size
void getPaddedRowCol(image* Image, int size, int dir){
	
		//Get the padded size for image based on the type of conv direction
		if(!dir){                                                 					//Horizantal
			Image->pad_coly = IMAGE_PADDING(Image->coly, size) + size - 1; 				// IMAGE COLS need to be multiple of kernel size (9,13,15)
			Image->pad_rowx = Image->rowx;																		//The number of rows may remain same
		}else{																															//Vertical
			Image->pad_rowx = IMAGE_PADDING(Image->rowx, size) + size - 1; 			// IMAGE ROWS need to be multiple of kernel size (9,13,15)
			Image->pad_coly = Image->coly;																	//The number of rows may remain same
		}
}


//Input Image generation  -- Generates image in multiple of stencil size 9, 13, 15 
void genImage(image* Image){
	int i, j, mem = 0;

	for(i = 0; i < Image->pad_rowx; i++){
		
		for(j = 0; j < Image->pad_coly; j++){
		
			if( (j > (Image->coly - 1)) || (i > (Image->rowx - 1)) ){
				
				*(Image->img_ptr + mem) = 0;														//Zero padding
			} else {
			
				*(Image->img_ptr + mem) = (rand() % 255);						//Values range from 0 to 255
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
void genKernel(int* kernel, int size){
	int i, mem = 0;

	for(i = 0; i < size; i++){
		
			*(kernel + mem) = (rand() % 255);								//Values range from 0 to 255 -- 8 bit co-eff
			mem++;
		}
	
	fprintf(stderr, "\nGenerated a stencil kernel of size %d\n", size);
}

//print the stencil kernel to a file
void printKernel(int* kernel, int size, FILE* fp){

	int i, mem = 0;

	for(i = 0; i < size; i++){
		
			fprintf(fp, "%d\t", *(kernel + mem));
			mem++;
	}

	fprintf(fp, "\n");
}


//SIFT blurring Kernel  -- 1D Horizantal Convultion
void blur_filter_hor(image* inputImage, int* kernel, image* outImage, int size){
	
	int i, j, k, index, index_out;
	fprintf(stderr, "\n----1D Horizantal Convolution----\n\n");
	int pixel_blur, blur_prod;
	int normalized_blur;

	//Iterate whole image -- Horizantal direction
	for(i = 0; i < (inputImage->pad_rowx); i++){      //Row-wise

		fprintf(stderr, "CALCULATING BLUR OF %dth ROW OF IMAGE\n", i);
		for(j = 0; j < ((inputImage->pad_coly) - (size - 1)); j++){      //Column-wise

			pixel_blur = 0, normalized_blur = 0;
			if(j < (inputImage->coly)){																		//Avoid unnecessary zero-computation

				for (k = 0; k < size; k++){
					blur_prod = 0;	
					index = INDEX_1D_HOR(i, j, k, inputImage->pad_coly, size); 

					blur_prod = (*(inputImage->img_ptr + index)) * (*(kernel + k)); //Product

					pixel_blur += blur_prod;
				}//inner for
			
				//Insert the blurred pixel to output image
				index_out = (outImage->coly * i) + j;
				if(pixel_blur > 255)
					normalized_blur = (pixel_blur % ((pixel_blur / 255) * 255));    //Normalizing
				*(outImage->img_ptr + index_out) = normalized_blur;
			}//end if

		}//first outer loop

	}//second outer loop  -- entire image done

}

//SIFT blurring Kernel  -- 1D Vertical Convultion
void blur_filter_ver(image* inputImage, int* kernel, image* outImage, int size){
	
	int i, j, k, index, index_out;
	fprintf(stderr, "\n----1D Vertical Convolution----\n\n");
	int pixel_blur, blur_prod;
	int normalized_blur;

	//Iterate whole image -- Horizantal direction
	for(j = 0; j < (inputImage->pad_coly); j++){      //Column-wise pixels

		fprintf(stderr, "CALCULATING BLUR OF %dth COLUMN OF IMAGE\n", j);
		for(i = 0; i < ((inputImage->pad_rowx) - (size - 1)); i++){      //Rows-wise pixels

			pixel_blur = 0, normalized_blur = 0;
			if(i < (inputImage->rowx)){																		//Avoid unnecessary zero-computation

				for (k = 0; k < size; k++){
					blur_prod = 0;	
					index = INDEX_1D_VER(i, j, k, inputImage->rowx, size); 

					blur_prod = (*(inputImage->img_ptr + index)) * (*(kernel + k)); //Product

					pixel_blur += blur_prod;
				}//inner for
			
				//Insert the blurred pixel to output image
				index_out = j + (outImage->rowx * i);
				if(pixel_blur)
					normalized_blur = (pixel_blur % ((pixel_blur / 255) * 255));    //Normalizing
				*(outImage->img_ptr + index_out) = normalized_blur; 
			}//end if

		}//first outer loop

	}//second outer loop  -- entire image done

}





#endif
