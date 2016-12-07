// SIFT - Scale Invariant Feature Transform Kernel//
// This code calls the sift-extrema on The generated DoG Images//
// Each Dog Image is the down-sampled version of original image//
// SIFT-EXTREMA computes maxima and minima of DoG images
// by comparing a pixel in each DoG image with 26 neighbors from 
// all 3 DoGs with stencil size 3

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <assert.h>

#include "sift_extrema.h"

static int DEBUG_FLAG = 0;

#define SIZE_INT sizeof(int) 

#define MIN(A, B) ( (A < B) ? 1 : 0 )
#define MAX(A, B) ( (A > B) ? 1	: 0 )

void usage(char** argv){
	fprintf(stderr, 
			"\nUsage:	%s -x [Image_Row_count] -y [Image_Column_Count] -d [Direction Horizantal(0)/vertical(1)] -o [Output file]\n", basename(argv[0]));
}


//*************SIFT_EXTREMA_HOR--KERNEL***************//	

void extrema_filter_hor(image* Image1, image* Image2, image* Image3, image* outImage){

	//Iterate the 2nd image -- All pixels need to be compared with its 8 beighbors 
	// Plus all the 18 neighbors from 2 other images
	int i, j, k, index = 0;
	int pixel, pix1, pix2;

	//Iterate whole image
	for(i = 0; i < (Image2->rowx); i++){

		for(j = 0; j < (Image2->coly); j++){

			//Get the pixel
			index = (Image2->coly * i) + j; 
			pixel = *(Image2->img_ptr + index);

			pix1 = *(Image1->img_ptr + index); // get the smae pixel from image 1
			pix2 = *(Image3->img_ptr + index); // get the same pixel from image 2

			for(k = 0; k < SIZE; k++){
					
		


			}//k for

		}//first outer loop

	}//second outer loop


}

//*************SIFT_EXTREMA_HOR--KERNEL***************//	


//*************SIFT_EXTREMA_VER--KERNEL***************//	

void extrema_filter_ver(image* Image1, image* Image2, image* Image3, image* outImage){

}

//*************SIFT_EXTREMA_VER--KERNEL***************//	

//MAIN 
int main(int argc, char** argv){
		
		//Use defaults if user does not specify image and filter sizes
		int x = 3, y = 3, dir = 0;
		char* outfile = "extrema.out";

		//Parse the arguments
		int r;
		while((r = getopt(argc, argv, "x:y:d:o:h")) != -1)
		{
			switch(r) {
				case 'x':
					x = atoi(optarg);
					break;
				case 'y':
					y = atoi(optarg);
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

		if(argc < 7){
			usage(argv);
			fprintf(stderr, "No size options given -- Defaulting to image size (3 x 3) and 1D Stencil Size 9 in Horizantal direction\n");
		}

		//Open a file for log output
		FILE* fp = fopen(outfile, "w");
		assert(fp != NULL);
	
		//Assumptions -- We have 3 images, each of them being the DoG of
		//images downsampled by 2 every step//
		//Here we generate these 3 images using randomizer and then apply extrema kernel 

		//Allocate DoG Image structures for an images 
		
		//DOG IMAGE-1//
		image* dogImage1 = (image*)malloc(sizeof(image)); 	
		dogImage1->rowx = x;
		dogImage1->coly = y;

		//Get the padded size for image based on the type of conv direction
		getPaddedRowCol(dogImage1, dir);

		//Generate DoG image-1
		dogImage1->img_ptr = (int*)malloc(SIZE_INT * (dogImage1->pad_rowx) * (dogImage1->pad_coly));	
		srand(time(NULL));
		genImage(dogImage1);
		
		fprintf(fp, "\nDOG IMAGE-1 MATRIX -- ZERO PADDED\n");
		printImage(dogImage1, fp);

		//DOG IMAGE-2//
		image* dogImage2 = (image*)malloc(sizeof(image)); 	
		dogImage2->rowx = x;
		dogImage2->coly = y;

		//Get the padded size for image based on the type of conv direction
		getPaddedRowCol(dogImage2, dir);

		//Generate DoG image-2
		dogImage2->img_ptr = (int*)malloc(SIZE_INT * (dogImage2->pad_rowx) * (dogImage2->pad_coly));	
		srand(time(NULL));
		genImage(dogImage2);
		
		fprintf(fp, "\nDOG IMAGE-3 MATRIX -- ZERO PADDED\n");
		printImage(dogImage2, fp);

		//DOG IMAGE-3//
		image* dogImage3 = (image*)malloc(sizeof(image)); 	
		dogImage3->rowx = x;
		dogImage3->coly = y;

		//Get the padded size for image based on the type of conv direction
		getPaddedRowCol(dogImage3, dir);

		//Generate DoG image-1
		dogImage3->img_ptr = (int*)malloc(SIZE_INT * (dogImage3->pad_rowx) * (dogImage3->pad_coly));	
		srand(time(NULL));
		genImage(dogImage3);
		
		fprintf(fp, "\nDOG IMAGE-3 MATRIX -- ZERO PADDED\n");
		printImage(dogImage3, fp);

		//Final SIFT-Extrema detection Output image
		image* outImage = (image*)malloc(sizeof(image));	
		outImage->rowx = x;
		outImage->coly = y;
		outImage->pad_rowx = x;
		outImage->pad_coly = y;
		outImage->img_ptr = (int*)malloc(SIZE_INT * (outImage->rowx) * (outImage->coly));	
		
		//Perform 1D Gaussin Blur for each input image based on Convolution direction
		if(!dir){
		//	extrema_filter_hor(dogImage1, dogImage2, dogImage3, outImage);
		}
		
		//Print the Output images
		fprintf(fp, "\nFinal OUTPUT IMAGE MATRIX\n");
	 	//printImage(outImage, fp);

		fclose(fp);
		free(dogImage1->img_ptr);
		free(dogImage1);
		free(dogImage2->img_ptr);
		free(dogImage2);
		free(dogImage3->img_ptr);
		free(dogImage3);
		free(outImage->img_ptr);
		free(outImage);
		
		return 0;

}
			
