// SIFT - Scale Invariant Feature Transform Kernel//
// This code calls the sift-blur on Original Image first with different Gaussian co-efficients
// And then Downsamples(by 2) the image and calls sift-blur again until a pyrmaid of images is formed
// For computation purpose we form pyramid size of 2 (2 images with different blur co-efficients) 
// And Difference of Gaussian(DoG) of 2 images of a pyramid are performed here
// Does NOT compute the difference in stencil-wise blocks

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <assert.h>

#include "sift_dog.h"
#include "sim.h"

#define SIZE_INT sizeof(int) 

static int DEBUG_FLAG = 0;

void usage(char** argv){
	fprintf(stderr, 
			"\nUsage:	%s -x [Image_Row_count] -y [Image_Column_Count] -s [Stencil 1D Size (9, 13, 15)] -d [Direction Horizantal(0)/vertical(1)] -o [Output file]\n", basename(argv[0]));
}


//*************SIFT_DOG--KERNEL***************//	
__attribute__ ((noinline))
void dog_filter(image* Image1, image* Image2, image* dogImage){

	//Do the MATRIX subtraction of 2 matrices
	int i, j, k, index;
	int pix1, pix2;

	//fprintf(stderr, "\n");
	//Iterate whole images
	for(i = 0; i < (Image1->rowx); i++){

		index = ((Image1->coly) * i); 
		//for(j = 0; j < (Image1->coly); j++){

			//index = ((Image1->coly) * i) + j; 

		for(k = 0; k < (Image1->coly); k++){
			
			pix1 = *(Image1->img_ptr + index + k);
			pix2 = *(Image2->img_ptr + index + k);

			*(dogImage->img_ptr + index + k) = abs(pix1 - pix2);

		}//first outer loop

	}//second outer loop

}
//*************SIFT_DOG--KERNEL***************//	

//MAIN 
int main(int argc, char** argv){
		
		//Use defaults if user does not specify image and filter sizes
		int x = 3, y = 3, size = 9, dir = 0;
		char* outfile = "dog.out";

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
		
		//Allocate Input Image structures for an images 
		
		//INPUT IMAGE 
		image* inputImage = (image*)malloc(sizeof(image)); 	
		inputImage->rowx = x;
		inputImage->coly = y;

		//Get the padded size for image based on the type of conv direction
		getPaddedRowCol(inputImage, size, dir);

		//Generate Input image
		inputImage->img_ptr = (int*)malloc(SIZE_INT * (inputImage->pad_rowx) * (inputImage->pad_coly));	
		srand(time(NULL));
		genImage(inputImage);
		
		fprintf(fp, "\nINPUT IMAGE MATRIX -- ZERO PADDED\n");
		printImage(inputImage, fp);

		//Blurred Output image-1 - with 1st gaussian kernel
		image* outImage1 = (image*)malloc(sizeof(image));	
		outImage1->rowx = x;
		outImage1->coly = y;
		outImage1->pad_rowx = x;
		outImage1->pad_coly = y;
		outImage1->img_ptr = (int*)malloc(SIZE_INT * (outImage1->rowx) * (outImage1->coly));	

		//Blurred Output image-2
		image* outImage2 = (image*)malloc(sizeof(image));	
		outImage2->rowx = x;
		outImage2->coly = y;
		outImage2->pad_rowx = x;
		outImage2->pad_coly = y;
		outImage2->img_ptr = (int*)malloc(SIZE_INT * (outImage2->rowx) * (outImage2->coly));	

		//Generate 1st Gaussian co-efficient kernel for image
		int* coeff_kernel1 = malloc(SIZE_INT * size);
		srand(time(NULL) * 20);
		genKernel(coeff_kernel1, size);

		fprintf(fp, "\n1D CO-EFFICIENT KERNEL-1\n");
		printKernel(coeff_kernel1, size, fp);

		//Generate 2nd Gaussian co-efficient kernel for image
		int* coeff_kernel2 = malloc(SIZE_INT * size);
		srand(time(NULL) * 40);
		genKernel(coeff_kernel2, size);

		fprintf(fp, "\n1D CO-EFFICIENT KERNEL-2\n");
		printKernel(coeff_kernel2, size, fp);


		//Perform 1D Gaussin Blur for each input image based on Convolution direction
		if(!dir){
			blur_filter_hor(inputImage, coeff_kernel1, outImage1, size);
			blur_filter_hor(inputImage, coeff_kernel2, outImage2, size);
		}else{
			blur_filter_ver(inputImage, coeff_kernel1, outImage1, size);
			blur_filter_ver(inputImage, coeff_kernel2, outImage2, size);
		}
		
		//Print the Output images
		fprintf(fp, "\nBlurred OUTPUT IMAGE-1 MATRIX\n");
	 	printImage(outImage1, fp);

		fprintf(fp, "\nBlurred OUTPUT IMAGE-2 MATRIX\n");
	 	printImage(outImage2, fp);

		//Image for DOG
		image* dogImage = (image*)malloc(sizeof(image));	
		dogImage->rowx = x;
		dogImage->coly = y;
		dogImage->pad_rowx = x;
		dogImage->pad_coly = y;
		dogImage->img_ptr = (int*)malloc(SIZE_INT * (dogImage->rowx) * (dogImage->coly));	

		//Now call SIF-DOG Kernel
		if(DEBUG_FLAG)	
            begin_roi();
		dog_filter(outImage1, outImage2, dogImage);	
		
		if(DEBUG_FLAG)	
            end_roi();

		//Print the final dog Image
		fprintf(fp, "\nDoG OUTPUT IMAGE MATRIX\n");
	 	printImage(dogImage, fp);

		fclose(fp);
		free(inputImage->img_ptr);
		free(inputImage);
		free(outImage1->img_ptr);
		free(outImage1);
		free(outImage2->img_ptr);
		free(outImage2);
		free(coeff_kernel1);
		free(coeff_kernel2);
		free(dogImage->img_ptr);
		free(dogImage);

		return 0;

}
			
