// this version adopts shifting
// the column size has to be multiples of 16

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
#include "../sift_dog/sift_dog_sb.h"
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
	int i, j, k, l;
	int final_sad = 0;
	int size_short = sizeof(short);
	int size_4short = 4 * size_short;

	short int* diffImagebuffer = (short int *)malloc(size_short * img_rowx * img_coly);
	long long * shift_buffer;
	memset(diffImagebuffer, 0, size_short * img_rowx * img_coly);

	// first dfg file to do sift_dog (absolute difference)
	{
		// begin_roi();
		int multi = 4;
		int acc_size = multi * size_4short;
		int loops = img_coly / 4 * img_rowx / multi;
		SB_CONFIG(sift_dog_sb_config, sift_dog_sb_size);

		SB_DMA_READ(inputImage,
			acc_size,
			acc_size,
			loops,
			P_sift_dog_sb_I);

		SB_DMA_READ(refImage,
			acc_size,
			acc_size,
			loops,
			P_sift_dog_sb_F);

		SB_DMA_WRITE(P_sift_dog_sb_R,
			acc_size,
			acc_size,
			loops,
			diffImage);

		SB_WAIT_ALL();
		// end_roi();
	}

	{
		shift_buffer = (long long *)malloc(size_4short * 8 * 4);
		memset(shift_buffer, 0, size_4short * 32);

		for (k = 0; k < 4; k++) {
			for (j = 0; j < 4; j++) {
				shift_buffer[k * 8 + j * 2] = k * 16;
				shift_buffer[k * 8 + j * 2 + 1] = 64 - k * 16;
			}
		}

		// for (i = 0; i < 32; i++)
		// 	printf("%d ", shift_buffer[i]);
		// printf("\n");
	
		SB_CONFIG(ime_sad_sb_config, ime_sad_sb_size);
		

		int loops = (img_coly) / 4;
		int index = 0;
		int ind = 0;
		l = loops * (rowx - 3);
		// printf("l: %d\n", l);
		//  int scr_addr = 0;
		//
		//	SB_DMA_SCRATCH_LOAD(shift_buffer, 0, 8 * size_4short, 1, scr_addr);
		//	SB_WAIT_SCR_WR();

		//	SB_SCR_PORT_STREAM(scr_addr,
		//	 	0,
		//	 	8 * size_4short * 4,
		//	 	(rowx - 3) * loops,
		//	 	P_ime_sad_sb_S);
		// printf("%d %d\n", coly, img_coly);
		begin_roi();
		SB_DMA_WRITE_SHF16(P_ime_sad_sb_O,
			size_4short,
			size_4short,
			l,
			diffImagebuffer);

		for (i = 0; i < 4; i++) {
			// printf("yes\n"); //???????????????
			SB_DMA_READ(shift_buffer + i * 8, // shift_buffer is a pointer to long long array
				0,
				2 * size_4short * 4,
				l,
				P_ime_sad_sb_S);

			SB_DMA_READ(diffImage, // diffImage is a pointer to short int array
				size_4short,
				2 * size_4short,
				l,
				P_ime_sad_sb_Ia);

			SB_DMA_READ(diffImage + img_coly,
				size_4short,
				2 * size_4short,
				l,
				P_ime_sad_sb_Ib);

			SB_DMA_READ(diffImage + img_coly + img_coly,
				size_4short,
				2 * size_4short,
				l,
				P_ime_sad_sb_Ic);

			SB_DMA_READ(diffImage + img_coly + img_coly + img_coly,
				size_4short,
				2 * size_4short,
				l,
				P_ime_sad_sb_Id);
		}
		SB_WAIT_ALL(); // reason 1: wait_all not waiting
		end_roi();
		
		for (i = 0; i < 4; i++)
			for (j = 0; j < l; j++)
				diffImage[j * 4 + i] = diffImagebuffer[i * l + j];

		for (i = 0; i < rowx - 3; i++) {
			ind = i * img_coly;
			for (j = coly - 3; j < img_coly; j++) {
				diffImage[ind + j] = 0;
			}
		}

		for (i = rowx - 3; i < img_rowx; i++) {
			ind = i * img_coly;
			for (j = 0; j < img_coly; j++) {
				diffImage[ind + j] = 0;
			}
		}

		final_sad = 0;
		for (i = 0; i < rowx - 3; i++) {
			ind = i * img_coly;
			for (j = 0; j < coly - 3; j++) {
				final_sad += diffImage[ind + j];
			}
		}
	}
	
	free(shift_buffer);
	
	free(diffImagebuffer);
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
		int img_coly = coly;
		int img_rowx = rowx + 1; // to avoid read overflow
		int remainder = (coly) % 4;
		if (remainder > 0) // pad column to multiples of 4
			img_coly = coly + 4 - remainder;

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
		memset(diffImage, 0, SIZE_INT * img_rowx * img_coly);

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
