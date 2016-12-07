// this method uses a mask
// it reads data at the same position 4 times.

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

#include "ime_sad.h"
// #include "/u/y/f/yfl/private/mpu/softbrain-workloads/common/include/sim_timing.h"
#include "../../../common/include/sb_insts.h"
#include "ime_sad_sb.h"
#include "sim.h"

#define SIZE_INT sizeof(short int) 

static int DEBUG_FLAG = 0;

void usage(char** argv){
	fprintf(stderr, 
			"\nUsage:	%s -x [Image_Row_count] -y [Image_Column_Count] -d [Image Ref diff 1/0] -o [Output file]\nKernel/Stencil is of fixed size - 4 x 4\n", argv[0]);
}


//*************IME_SAD SOFTBRAIN KERNEL***************//

//SAD Calculation Kernel
int SAD_filter(short int* inputImage, short int* refImage, short int* diffImage, int rowx, int coly, int img_rowx, int img_coly){
	
	//Need to choose first 'size x size' blocks from either image
	int i = 0, j = 0, k, l;
	int final_sad = 0;
	int size_short = sizeof(short);
	int size_4short = 4 * size_short;
	short * ker = (short *)malloc(size_4short * 8);
	for (i = 0; i < 32; i++)
		ker[i] = 0;
	for (i = 0; i < 4; i++)
		for (j = 0; j < 4; j++)
			ker[i * 8 + i + j] = 1;

	for (i = 0; i < 4; i++) {
		for (j = 0; j < 8; j++)
			printf("%d ", ker[i * 8 + j]);
		printf("\n");
	}

	SB_CONFIG(ime_sad_sb_config, ime_sad_sb_size);

	int scr_addr = 0;
	
	SB_DMA_SCRATCH_LOAD(ker, 0, 8 * size_4short, 1, scr_addr);
	SB_WAIT_SCR_WR();

	long long * buffer = (long long *)malloc(size_4short * img_rowx * img_coly);
	
	for (i = 0; i < rowx - 3; i++) {
	// i=0; {
		int index = i * img_coly;
		int loops = (coly) / 4;
		// printf("loops: %d\n", loops);
		for (j = 0; j < loops; j++) {
		// j = 0;{
			int ind = index + j * 4;
			int l = 4;
			for (k = 0; k < 4; k++) {
				SB_CONST(P_ime_sad_sb_C, 0, 1);

				SB_DMA_READ(inputImage + ind,
					size_short * img_coly,
					2 * size_4short,
					l,
					P_ime_sad_sb_I);

				SB_DMA_READ(refImage + ind,
					size_short * img_coly,
					2 * size_4short,
					l,
					P_ime_sad_sb_F);

				SB_SCR_PORT_STREAM(scr_addr + k * 2 * size_4short,
					0,
					2 * size_4short,
					l,
					P_ime_sad_sb_K);

				SB_RECURRENCE(P_ime_sad_sb_R,
					P_ime_sad_sb_C,
					l - 1);

				
				SB_DMA_WRITE(P_ime_sad_sb_R,
					size_4short,
					size_4short,
					1,
					buffer + ind + k);

			}
			// SB_DMA_WRITE_SHF16(P_ime_sad_sb_R,
			// 	1 * size_4short,
			// 	1 * size_4short,
			// 	l / 4,
			// 	diffImage + ind);
		}
	}

	SB_WAIT_ALL();

	for (i = 0; i < rowx - 3; i++)
		for (j = 0; j < coly - 3; j++) {
			int ind = i * img_coly + j;
			final_sad += (short)(*(buffer + ind));
			*(diffImage + ind) = (short)(*(buffer + ind));
		}

	free(ker);
	free(buffer);

  return final_sad;
}

//*************IME_SAD--KERNEL***************//

//MAIN 
int main(int argc, char** argv){
		
		//Use defaults if user does not specify image and filter sizes
		int rowx = 4, coly = 4, diff = 1;
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

		if(argc != 9 || rowx < 4){
			usage(argv);
			fprintf(stderr, "\nNo size options given -- Defaulting to image size (4 x 4)\n");
			exit(1);
		}

		//Get the padded size for image
		// int img_rowx = IMAGE_PADDING(rowx); // IMAGE SIZE needs to be multiple of kernel for sad
		// int img_coly = IMAGE_PADDING(coly);
		int img_coly = coly;
		int img_rowx = rowx;
		int remainder = (coly) % 4;
		if (remainder > 0)
			img_coly = coly + 4 - remainder;

		// padding 4 more short integers for each row
		// img_coly += 4;
	
		//Open a file for log output
		FILE* fp = fopen(outfile, "w");
		assert(fp != NULL);

		//Generate Image
		srand(time(NULL));
		short int* inputImage = (short int*)malloc(SIZE_INT * img_rowx * img_coly);	
		genImage(inputImage, rowx, coly, img_rowx, img_coly);
		
		fprintf(fp, "\nINPUT IMAGE MATRIX\n");
		printImage(inputImage, img_rowx, img_coly, fp);

		//Get a reference Image -- May need to get from user
		srand(time(NULL) + (diff * 20)); 																//To avoid same image generation
		short int* refImage = (short int*)malloc(SIZE_INT * img_rowx * img_coly);	
		genImage(refImage, rowx, coly, img_rowx, img_coly);
		
		fprintf(fp, "\nREF IMAGE MATRIX\n");
		printImage(refImage, img_rowx, img_coly, fp);

		//Temporary Diff image
		short int* diffImage = (short int*)malloc(SIZE_INT * img_rowx * img_coly);
		int i, j;
		for (i = 0; i < img_rowx; i++) {
			for (j = 0; j < img_coly; j++) {
				diffImage[i * img_coly + j] = -1;
			}
		}

		//Find the SAD of the images at kernel(convolution) size -- ker_size
		if(DEBUG_FLAG)	
			begin_roi();
		int sad_out = SAD_filter(inputImage, refImage, diffImage, rowx, coly, img_rowx, img_coly);
		if(DEBUG_FLAG)
			end_roi();

		fprintf(fp, "\nDIFF IMAGE MATRIX\n");
		printImage(diffImage, img_rowx, img_coly, fp);

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
