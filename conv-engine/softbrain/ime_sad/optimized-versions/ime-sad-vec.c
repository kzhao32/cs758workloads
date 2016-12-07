// Interget Motion Estimation SAD Kernel for H.264 Video Encoder //
// 2D Convolution -- Abs Diff and Addition
// Abs diff b/w 2 images and addition of differences
// Stencil kernel size -- Granularity at which convolution takes place


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <assert.h>
#include <stdbool.h>

#include "ime-sad.h"
#include "common.h"

#define SIZE_INT sizeof(int) 


static int DEBUG_FLAG = 1;

void usage(char** argv){
	fprintf(stderr, 
			"\nUsage:	%s -x [Image_Row_count] -y [Image_Column_Count] -d [Image Ref diff 1/0] -o [Output file]\nKernel/Stencil is of fixed size - 4 x 4\n", basename(argv[0]));
}


//*************IME_SAD--KERNEL***************//

//SAD Calculation Kernel
int SAD_filter(int* inputImage, int* refImage, int* diffImage, int rowx, int coly){
	
	printf("Started Tracing\n");
	//Need to choose first 'size x size' blocks from either image
	int i = 0, j = 0, k, index;
	int final_sad = 0;
	int pix1, pix2;

	//fprintf(stderr, "\n");
	//Iterate whole image
	for(i = 0; i < IMAGE_PADDING(rowx); i=i++){

		int matrix_sad = 0;
		for(j = 0; j < IMAGE_PADDING(coly); j=j++){

			//Iterate through whole image block-wise(conv size) - 4 x 4 2D convolution
			//fprintf(stderr, "CALCULATE SAD - Block with i=%d, j=%d\n", i, j);
			
			int block_sad = 0;
			
			//manually unrolling the loop -- index incremented by size times 
			//for (k = 0; k < SIZE; k++){
		
				//index = INDEX_2D(i, j, k, rowx, coly, SIZE); 
				
				//1
				pix1 = *(inputImage + index);
				pix2 = *(refImage + index);
				*(diffImage + index) = abs(pix1 - pix2);		//diff
				block_sad += abs(pix1 - pix2);							//temp sum

				index++;
		/*
				//2
				pix1 = *(inputImage + index + 1);
				pix2 = *(refImage + index + 1);
				*(diffImage + index + 1) = abs(pix1 - pix2);		//diff
				block_sad += abs(pix1 - pix2);							//temp sum

				//3
				pix1 = *(inputImage + index + 2);
				pix2 = *(refImage + index + 2);
				*(diffImage + index + 2) = abs(pix1 - pix2);		//diff
				block_sad += abs(pix1 - pix2);							//temp sum

				//4
				pix1 = *(inputImage + index + 3);
				pix2 = *(refImage + index + 3);
				*(diffImage + index + 3) = abs(pix1 - pix2);		//diff
				block_sad += abs(pix1 - pix2);							//temp sum
		*/
				
			//}//inner for
		
			matrix_sad += block_sad;
		
		}//first outer loop

		final_sad += matrix_sad;

	}//second outer loop

	return final_sad;
}

//*************IME_SAD--KERNEL***************//

//MAIN 
int main(int argc, char** argv){
		
		//Use defaults if user does not specify image and filter sizes
		int rowx = 3, coly = 3, diff = 1;
		char* outfile = "sad.out";

		//Parse the arguments
		int r;
		while((r = getopt(argc, argv, "x:y:d:o:h")) != -1)
		{
			switch(r) {
				case 'x':
					rowx = atoi(optarg);
					break;
				case 'y':
					coly = atoi(optarg);
					break;	
				case 'd':
					diff = atoi(optarg);
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

		if(argc != 7){
			usage(argv);
			fprintf(stderr, "\nNo size options given -- Defaulting to image size (3 x 3)\n");
		}


		//Get the padded size for image
		int img_rowx = IMAGE_PADDING(rowx); // IMAGE SIZE needs to be multiple of kernel for sad
		int img_coly = IMAGE_PADDING(coly);
	
		//Open a file for log output
		FILE* fp = fopen(outfile, "w");
		assert(fp != NULL);

		//Generate Image
		srand(time(NULL));
		int* inputImage = (int*)malloc(SIZE_INT * img_rowx * img_coly);	
		genImage(inputImage, rowx, coly);
		
		fprintf(fp, "\nINPUT IMAGE MATRIX\n");
		printImage(inputImage, rowx, coly, fp);

		//Get a reference Image -- May need to get from user
		srand(time(NULL) + (diff * 20)); 																//To avoid same image generation
		int* refImage = (int*)malloc(SIZE_INT * img_rowx * img_coly);	
		genImage(refImage, rowx, coly);
		
		fprintf(fp, "\nREF IMAGE MATRIX\n");
		printImage(refImage, rowx, coly, fp);

		//Temporary Diff image
		int* diffImage = (int*)malloc(SIZE_INT * img_rowx * img_coly);	
		
		//Find the SAD of the images at kernel(convolution) size -- ker_size
		if(DEBUG_FLAG)	
			DETAILED_SIM_START();
		int sad_out = SAD_filter(inputImage, refImage, diffImage, rowx, coly);
		if(DEBUG_FLAG)
			DETAILED_SIM_STOP();

		fprintf(fp, "\nDIFF IMAGE MATRIX\n");
		printImage(diffImage, rowx, coly, fp);
		
		if(!sad_out){
			fprintf(stderr, "\nSAD OUTPUT VALUE=%d: IMAGES ARE SAME\n\n", sad_out);
			fprintf(fp, "\nSAD OUTPUT VALUE=%d: IMAGES ARE SAME\n", sad_out);
		}else{
			fprintf(stderr, "\nSAD OUTPUT VALUE=%d: IMAGES ARE DIFFERENT\n\n", sad_out);
			fprintf(fp, "\nSAD OUTPUT VALUE=%d: IMAGES ARE DIFFERENT\n", sad_out);
		}
	
		fclose(fp);
		free(inputImage);
		free(refImage);
		free(diffImage);

		return 0;
}
			
