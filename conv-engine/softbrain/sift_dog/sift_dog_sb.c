// TODO general padding before calling dog_filter

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

#include "sift_dog_sb.h"
#include "../../../common/include/sb_insts.h"

#define SIZE_INT sizeof(short int) 

static int DEBUG_FLAG = 0;

void usage(char** argv){
	fprintf(stderr, 
			"\nUsage:	%s -x [Image_Row_count] -y [Image_Column_Count] -s [Stencil 1D Size (9, 13, 15)] -d [Direction Horizantal(0)/vertical(1)] -o [Output file]\n", argv[0]);
}

//*************SIFT_DOG--KERNEL***************//	
__attribute__ ((noinline))
void dog_filter(image* Image1, image* Image2, image* dogImage){

	//Do the MATRIX subtraction of 2 matrices

	// version 1
	// int i;
	// int loops = (Image1->coly + 3) / 4;
	// int size_4short = 4 * sizeof(short);

	// SB_CONFIG(sift_dog_sb_config, sift_dog_sb_size);

	// // printf("Image1->pad_coly: %d\n", dogImage->pad_coly);

	// for(i = 0; i < (Image1->rowx); i++){
	// // i = 1;	{
	// 	int inIndex = (Image1->pad_coly * i); 
	// 	int outIndex = (dogImage->pad_coly * i);
		
	// 	SB_DMA_READ(Image1->img_ptr + inIndex,
	// 		size_4short,
	// 		size_4short,
	// 		loops,
	// 		P_sift_dog_sb_I);

	// 	SB_DMA_READ(Image2->img_ptr + inIndex,
	// 		size_4short,
	// 		size_4short,
	// 		loops,
	// 		P_sift_dog_sb_F);

	// 	SB_DMA_WRITE(P_sift_dog_sb_R,
	// 		size_4short,
	// 		size_4short,
	// 		loops,
	// 		dogImage->img_ptr + outIndex);

	// }//second outer loop

	// SB_WAIT_ALL();

	// version 2
	int i;
	int size_4short = 4 * sizeof(short);
	int multi = 4;
	int acc_size = multi * size_4short;
	int loops = (Image1->coly + 3) / 4 * Image1->rowx / multi;

	begin_roi();

	SB_CONFIG(sift_dog_sb_config, sift_dog_sb_size);

	// printf("Image1->pad_coly: %d\n", dogImage->pad_coly);

	// for(i = 0; i < (Image1->rowx); i++){
	// i = 1;	{
	{
		SB_DMA_READ(Image1->img_ptr,
			acc_size,
			acc_size,
			loops,
			P_sift_dog_sb_I);

		SB_DMA_READ(Image2->img_ptr,
			acc_size,
			acc_size,
			loops,
			P_sift_dog_sb_F);

		SB_DMA_WRITE(P_sift_dog_sb_R,
			acc_size,
			acc_size,
			loops,
			dogImage->img_ptr);

	}//second outer loop

	SB_WAIT_ALL();
	end_roi();

}
//*************SIFT_DOG--KERNEL***************//	

//MAIN 
int main(int argc, char** argv){
		//Use defaults if user does not specify image and filter sizes
		int x = 4, y = 4, size = 9, dir = 0;
		char* outfile = "dog.out";

		//Parse the arguments
		int r;
		while((r = getopt(argc, argv, "x:y:o:h")) != -1)
		{
			switch(r) {
				case 'x':
					x = atoi(optarg);
					break;
				case 'y':
					y = atoi(optarg);
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
			// fprintf(stderr, "No size options given -- Defaulting to image size (3 x 3) and 1D Stencil Size 9 in Horizantal direction\n");
			exit(1);
		}

		// if((size != 9) && (size != 13) && (size != 15)){
		// 	usage(argv);
		// 	fprintf(stderr, "\nStencil Kernel needs to be of size -- 9 or 13 or 15\n");
		// 	exit(1);
		// }
		
		//Open a file for log output
		FILE* fp = fopen(outfile, "w");
		assert(fp != NULL);
		
		//Allocate Input Image structures for an images 
		
		//INPUT IMAGE 
		image* inputImage = (image*)malloc(sizeof(image)); 	
		inputImage->rowx = x;
		inputImage->coly = y;

		//Get the padded size for image based on the type of conv direction
		int remainder = y % 4;
		printf("remainder: %d\n", remainder);
		getPaddedRowCol(inputImage, 5 - remainder, dir);

		//Generate Input image
		inputImage->img_ptr = (short int*)malloc(SIZE_INT * (inputImage->pad_rowx) * (inputImage->pad_coly));	
		srand(time(NULL));
		genImage(inputImage);
		
		fprintf(fp, "\nINPUT IMAGE MATRIX -- ZERO PADDED\n");
		printImage(inputImage, fp);

		//Generate Input image
		image* inputImage2 = (image*)malloc(sizeof(image)); 	
		inputImage2->rowx = x;
		inputImage2->coly = y;
		inputImage2->pad_rowx = inputImage->pad_rowx;
		inputImage2->pad_coly = inputImage->pad_coly;

		inputImage2->img_ptr = (short int*)malloc(SIZE_INT * (inputImage->pad_rowx) * (inputImage->pad_coly));	
		genImage(inputImage2);
		
		fprintf(fp, "\nINPUT IMAGE 2 MATRIX -- ZERO PADDED\n");
		printImage(inputImage2, fp);

		//Image for DOG
		image* dogImage = (image*)malloc(sizeof(image));	
		dogImage->rowx = x;
		dogImage->coly = y;
		dogImage->pad_rowx = inputImage->pad_rowx;
		dogImage->pad_coly = inputImage->pad_coly;
		dogImage->img_ptr = (short int*)malloc(SIZE_INT * (dogImage->rowx) * (dogImage->coly));	
		for (int i = 0; i < dogImage->pad_rowx; i++)
			for (int j = 0; j < dogImage->pad_coly; j++) {
				*(dogImage->img_ptr + i * dogImage->pad_coly + j) = 0;
			}

		//Now call SIF-DOG Kernel
    
		dog_filter(inputImage, inputImage2, dogImage);
    

		//Print the final dog Image
		fprintf(fp, "\nDoG OUTPUT IMAGE MATRIX\n");
	 	printImage(dogImage, fp);

		fclose(fp);
		free(inputImage->img_ptr);
		free(inputImage);
		free(inputImage2->img_ptr);
		free(inputImage2);
		free(dogImage->img_ptr);
		free(dogImage);

		return 0;

}
			
