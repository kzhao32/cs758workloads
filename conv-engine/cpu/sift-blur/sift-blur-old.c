// SIFT - Scale Invariant Feature Transform Kernel//
// This code implements Gaussian Blurring using 1D horizantal and Vertical Convolution
// Kernel convolutes horizantal or vertical based on User input
// Stencil kernel size -- Granularity at which convolution takes place (9, 13, 15 tap)


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <assert.h>
#include "sift-blur.h"

#include "common.h"

#define SIZE_INT sizeof(int) 


static int DEBUG_FLAG = 0;

void usage(char** argv){
	fprintf(stderr, 
			"\nUsage:	%s -x [Image_Row_count] -y [Image_Column_Count] -s [Stencil 1D Size (9, 13, 15)] -d [Direction Horizantal(0)/vertical(1)] -o [Output file]\n", basename(argv[0]));
}


//*************SIFT_BLUR_HOR--KERNEL***************//

//SIFT blurring Kernel  -- 1D Horizantal Convultion
void blur_filter_hor(image* inputImage, int* kernel, image* outImage, int size){
	
	int i, j, k, index, index_out;
//	fprintf(stderr, "\n----1D Horizantal Convolution----\n\n");
	int pixel_blur, blur_prod;
	int normalized_blur;
	
	//Iterate whole image -- Horizantal direction
	for(i = 0; i < (inputImage->pad_rowx); i++){      //Row-wise

		//fprintf(stderr, "CALCULATING BLUR OF %dth ROW OF IMAGE\n", i);
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
				if(pixel_blur)
					normalized_blur = (pixel_blur % ((pixel_blur / 255) * 255));    //Normalizing
				*(outImage->img_ptr + index_out) = normalized_blur;
				
				printf("INDEX: %d, J: %d\n",index, j);
			
			}//end if

		}//first outer loop

	}//second outer loop  -- entire image done

}

//*************SIFT_BLUR_HOR--KERNEL***************//


//*************SIFT_BLUR_VER--KERNEL***************//

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

//*************SIFT_BLUR_VER--KERNEL***************//

//MAIN 
int main(int argc, char** argv){
		
		//Use defaults if user does not specify image and filter sizes
		int x = 3, y = 3, size = 9, dir = 0;
		char* outfile = "blur.out";

		//Parse the arguments
		int r;
		while((r = getopt(argc, argv, "x:y:s:d:o:h")) != -1)
		{
			switch(r) {
				case 'x':
					x = atoi(optarg);
					break;
				case 'y':
					y = atoi(optarg);
					break;	
				case 's':
					size = atoi(optarg);
					break;	
				case 'd':
					dir = atoi(optarg);
					break;	
				case 'o':
				  outfile = strdup(optarg); 
					break;
				case 'h':
				  usage(argv);
					exit(1);
					break;
				default:
					usage(argv);
			}
		}

		if(argc < 9){
			usage(argv);
			fprintf(stderr, "No size options given -- Defaulting to image size (3 x 3) and 1D Stencil Size 9 in Horizantal direction\n");
		}

		if((size != 9) && (size != 13) && (size != 15)){
			usage(argv);
			fprintf(stderr, "\nStencil Kernel needs to be of size -- 9 or 13 or 15\n");
			exit(1);
		}
	

		//Open a file for log output
		FILE* fp = fopen(outfile, "w");
		assert(fp != NULL);
		
		//Allocate Input Image structure
		image* inputImage = (image*)malloc(sizeof(image)); 	
		inputImage->rowx = x;
		inputImage->coly = y;

		//Get the padded size for image based on the type of conv direction
		getPaddedRowCol(inputImage, size, dir);

		//Generate Input image
		inputImage->img_ptr = (int*)malloc(SIZE_INT * (inputImage->pad_rowx) * (inputImage->pad_coly));	
		genImage(inputImage);
		
		fprintf(fp, "\nINPUT IMAGE MATRIX -- ZERO PADDED\n");
		printImage(inputImage, fp);

		//Output image
		image* outImage = (image*)malloc(sizeof(image));	
		outImage->rowx = x;
		outImage->coly = y;
		outImage->pad_rowx = x;
		outImage->pad_coly = y;
		outImage->img_ptr = (int*)malloc(SIZE_INT * (outImage->rowx) * (outImage->coly));	

		//Generate a Gaussian co-efficient kernel
		int* coeff_kernel = malloc(SIZE_INT * size);
		genKernel(coeff_kernel, size);

		fprintf(fp, "\n1D CO-EFFICIENT KERNEL\n");
		printKernel(coeff_kernel, size, fp);

		//Perform 1D Gaussin Blur at input image based on Convolution direction
		
		if(!dir){
			if(DEBUG_FLAG)	
				begin_roi();
			blur_filter_hor(inputImage, coeff_kernel, outImage, size);
		}else{
			if(DEBUG_FLAG)	
				begin_roi();
			blur_filter_ver(inputImage, coeff_kernel, outImage, size);
		}

		//Print the Output image
		fprintf(fp, "\nBlurred OUTPUT IMAGE MATRIX\n");
	 	printImage(outImage, fp);

		
		
		fclose(fp);
		free(inputImage->img_ptr);
		free(inputImage);
		free(outImage->img_ptr);
		free(outImage);
		free(coeff_kernel);
		
		return 0;

}
			
