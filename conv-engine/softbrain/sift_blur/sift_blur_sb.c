// TODO down sampling

// this version uses scratch pad for both horizontal and vertical convolution
// the size of stencil can be either 9 or 13. 15 is not yet supported
// the number of rows and columns of input image must be multiples of 4

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

#include "sim.h"
#include "sift_blur.h"

#include "sift_blur_ver_sb.h"
#include "sift_blur_hor_sb.h"
#include "../../../common/include/sb_insts.h"

#define SIZE_INT sizeof(short) 

static int DEBUG_FLAG = 0;
int tsize = 16;

void usage(char** argv){
	fprintf(stderr, 
			"\nUsage:	%s -x [Image_Row_count] -y [Image_Column_Count] -s [Stencil 1D Size (9, 13)] -d [Direction Horizantal(0)/vertical(1)] -o [Output file]\n", argv[0]);
}

//*************Down_Sample by 2--KERNEL***************//
void down_sample(image* inputImage){

	int i, k;
	int pixel, index;
	
	//Iterate whole image -- as 16 wide vecor 
	for(i = 0; i < (inputImage->rowx); i++){      //Row-wise

		//for(j = 0; j < (inputImage->coly); j=j+16){

			index = (inputImage->coly * i);

			for(k = 0; k < (inputImage->coly); k++){

				//1
				pixel = *(inputImage->img_ptr + index + k);
				*(inputImage->img_ptr + index + k) = pixel / 2;
			}

	}// i for

}

//*************SIFT_BLUR_HOR--KERNEL***************//
//SIFT blurring Kernel  -- 1D Horizantal Convultion
__attribute__ ((noinline))
void blur_filter_hor(image* inputImage, short int* kernel, image* outImage, int size){

	int i, j, k;

	SB_CONFIG(sift_blur_hor_sb_config, sift_blur_hor_sb_size);
	int size_short = sizeof(short);
	int size_4short = 4 * size_short;
	// short int * short_buffer = (short *)malloc(inputImage->pad_rowx * inputImage->pad_coly * size_short);

	//concatenate all kernels in one line
	short int * k1 = (short int *)malloc(sizeof(short int) * tsize * 4);
	memset(k1, 0, size_short * 4 * tsize);
	short int * outBuffer = (short int *)malloc(size_short * outImage->pad_rowx * outImage->pad_coly);
	memset(outBuffer, 0, size_short * outImage->pad_rowx * outImage->pad_coly);
	for (i = 0; i < tsize * 4; i++) {
		k1[i] = 0;
	}
	
	for (i = 0; i < size; i++) {
		// kernel 1 & 2
		k1[i + tsize + 1] = k1[i] = kernel[i];
		// kernel 3 & 4
		k1[i + 2 * tsize + 2] = k1[i + 3 * tsize + 3] = kernel[i]; 
	}
	// for (int i = 0; i < tsize * 4; i++) {
	// 	printf("%d ", k1[i]);
	// }
	// printf("\n");

	// load k1 to scratchpad
	int scr_addr = 0;
	
//	SB_DMA_SCRATCH_LOAD(k1, 0, tsize * size_4short, 1, scr_addr);
//	SB_WAIT_ALL();

  int loops = (inputImage->pad_coly + 3) / 4;

	int inIndex = 0; 
	int outIndex = 0;
	int l = loops * inputImage->rowx;

 	// SB_CONST(P_sift_blur_hor_sb_C, 0, l*loops*inputImage->rowx);

//	SB_SCR_PORT_STREAM(scr_addr,// + tsize * 2 * size_short,
//		0,
//		4 * size_4short * 2,
//		// 1 * loops * inputImage->rowx - 4,
//		l,
//		P_sift_blur_hor_sb_F);
//
//	SB_SCR_PORT_STREAM(scr_addr + tsize * 2 * size_short,
//		0,
//		4 * size_4short * 2,
//		// 1 * loops * inputImage->rowx - 4,
//		l,
//		P_sift_blur_hor_sb_F);

	begin_roi();
	SB_DMA_READ(k1,
			0,
			4 * size_4short * 2,
			l,
			P_sift_blur_hor_sb_F);
// printf("la: %d\n", l);
	SB_DMA_READ(k1 + tsize * 2,
			0,
			4 * size_4short * 2,
			l,
			P_sift_blur_hor_sb_F);
	// printf("lb: %d\n", l);
	SB_DMA_WRITE_SHF16(P_sift_blur_hor_sb_R,
			1 * size_4short,
			1 * size_4short,
			l,
			outBuffer);
// printf("lc: %d\n", l);
	SB_DMA_READ(inputImage->img_ptr,
		size_4short,
		4 * size_4short,
		l,
		P_sift_blur_hor_sb_I);
// printf("ld: %d\n", l);
	SB_DMA_READ(inputImage->img_ptr,
		size_4short,
		4 * size_4short,
		l,
		P_sift_blur_hor_sb_I);
// printf("le: %d\n", l);
	SB_WAIT_ALL();

	end_roi();

	int ind = 0, x, y;
	int glen = (inputImage->pad_coly - inputImage->coly) / 2;
	for (i = 0; i < outImage->rowx; i++) {
		for (j = 0; j < outImage->pad_coly; j++) {
			x = ((j >> 1) & 1) * (outImage->rowx >> 1) + (i >> 1);
			y = (i & 1) * (outImage->pad_coly >> 1 + glen) + ((j >> 2) << 1 ) + (j & 1);
			outImage->img_ptr[ind + j] = outBuffer[x * outImage->pad_coly + y];
		}
		ind += outImage->pad_coly;
	}

	ind = 0;
	for (i = 0; i < outImage->rowx; i++, ind += outImage->pad_coly) {
		for (j = outImage->coly; j < outImage->pad_coly; j++) {
			outImage->img_ptr[ind + j] = 0;
		}
	}

	free(k1);
	free(outBuffer);
}

//*************SIFT_BLUR_HOR--KERNEL***************//


//*************SIFT_BLUR_VER--KERNEL***************//
//SIFT blurring Kernel  -- 1D Vertical Convultion
__attribute__ ((noinline))
void blur_filter_ver(image* inputImage, short int* kernel, image* outImage, int size){
	
	int i, k;
	int size_short = sizeof(short);
	int size_4short = 4 * size_short;
	short int * k1 = (short *)malloc(size_4short * tsize);
	memset(k1, 0, size_4short * tsize);

	// construct a new kernel
	for (i = 0; i < size * 4; i++) {
		k1[i] = kernel[i / 4];
	}
	for (i = size * 4; i < tsize * 4; i++) {
		k1[i] = 0;
	}

	int j;
	k = 0;
	// printf("inputImage->rowx: %d\n", inputImage->rowx);
	// printf("inputImage->coly: %d\n", inputImage->coly);
	// printf("inputImage->pad_rowx: %d\n", inputImage->pad_rowx);
	// printf("inputImage->pad_coly: %d\n", inputImage->pad_coly);
	// for (i = 0; i < inputImage->pad_rowx; i++) {
	// 	for (j = 0; j < inputImage->pad_coly; j++) {
	// 		printf("%d ", inputImage->img_ptr[k]);
	// 		k++;
	// 	}
	// 	printf("\n");
	// }

	SB_CONFIG(sift_blur_ver_sb_config, sift_blur_ver_sb_size);
	
	
	int loops = (inputImage->coly + 3) / 4;
	int stride = inputImage->pad_coly;
	int l = loops * inputImage->rowx;

	// calculate the first half of kernel
	begin_roi();
	SB_CONST(P_sift_blur_ver_sb_C, 0, l);

	int tmp = 0;
	SB_DMA_READ(inputImage->img_ptr + tmp,
		size_4short,
		size_4short,
		l,
		P_sift_blur_ver_sb_I0);
	
	tmp += stride;
	SB_DMA_READ(inputImage->img_ptr + tmp,
		size_4short,
		size_4short,
		l,
		P_sift_blur_ver_sb_I1);
	
	tmp += stride;
	SB_DMA_READ(inputImage->img_ptr + tmp,
		size_4short,
		size_4short,
		l,
		P_sift_blur_ver_sb_I2);
	
	tmp += stride;
	SB_DMA_READ(inputImage->img_ptr + tmp,
		size_4short,
		size_4short,
		l,
		P_sift_blur_ver_sb_I3);
	
	tmp += stride;
	SB_DMA_READ(inputImage->img_ptr + tmp,
		size_4short,
		size_4short,
		l,
		P_sift_blur_ver_sb_I4);
	
	tmp += stride;
	SB_DMA_READ(inputImage->img_ptr + tmp,
		size_4short,
		size_4short,
		l,
		P_sift_blur_ver_sb_I5);
	
	tmp += stride;
	SB_DMA_READ(inputImage->img_ptr + tmp,
		size_4short,
		size_4short,
		l,
		P_sift_blur_ver_sb_I6);
	
	tmp += stride;
	SB_DMA_READ(inputImage->img_ptr + tmp,
		size_4short,
		size_4short,
		l,
		P_sift_blur_ver_sb_I7);

	SB_DMA_READ(k1,
		0,
		8 * size_4short,
		l,
		P_sift_blur_ver_sb_F);

	SB_DMA_WRITE(P_sift_blur_ver_sb_R,
		size_4short,
		size_4short,
		l,
		outImage->img_ptr);

	SB_WAIT_ALL();

	// calculate the second half of the kernel
	SB_DMA_READ(outImage->img_ptr,
		size_4short,
		size_4short,
		l,
		P_sift_blur_ver_sb_C);

	tmp = 8 * stride;
	SB_DMA_READ(inputImage->img_ptr + tmp,
		size_4short,
		size_4short,
		l,
		P_sift_blur_ver_sb_I0);
	
	tmp += stride;
	SB_DMA_READ(inputImage->img_ptr + tmp,
		size_4short,
		size_4short,
		l,
		P_sift_blur_ver_sb_I1);
	
	tmp += stride;
	SB_DMA_READ(inputImage->img_ptr + tmp,
		size_4short,
		size_4short,
		l,
		P_sift_blur_ver_sb_I2);
	
	tmp += stride;
	SB_DMA_READ(inputImage->img_ptr + tmp,
		size_4short,
		size_4short,
		l,
		P_sift_blur_ver_sb_I3);
	
	tmp += stride;
	SB_DMA_READ(inputImage->img_ptr + tmp,
		size_4short,
		size_4short,
		l,
		P_sift_blur_ver_sb_I4);
	
	tmp += stride;
	SB_DMA_READ(inputImage->img_ptr + tmp,
		size_4short,
		size_4short,
		l,
		P_sift_blur_ver_sb_I5);
	
	tmp += stride;
	SB_DMA_READ(inputImage->img_ptr + tmp,
		size_4short,
		size_4short,
		l,
		P_sift_blur_ver_sb_I6);
	
	tmp += stride;
	SB_DMA_READ(inputImage->img_ptr + tmp,
		size_4short,
		size_4short,
		l,
		P_sift_blur_ver_sb_I7);

	SB_DMA_READ(k1 + 8 * 4,
		0,
		8 * size_4short,
		l,
		P_sift_blur_ver_sb_F);

	SB_DMA_WRITE(P_sift_blur_ver_sb_R,
		size_4short,
		size_4short,
		l,
		outImage->img_ptr);

	SB_WAIT_ALL();
	end_roi();

	free(k1);
}

//*************SIFT_BLUR_VER--KERNEL***************//

//MAIN 
int main(int argc, char** argv){
		
		//Use defaults if user does not specify image and filter sizes
		int x = 4, y = 4, size = 9, dir = 0;
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

		if (dir == 0) {
			if((size != 9) && (size != 13) ) {
				usage(argv);
				fprintf(stderr, "\nHorizontal stencil kernel needs to be of size -- 9 or 13\n");
				exit(1);
			}
		} else {
			if((size != 9) && (size != 13) && (size != 15)) {
				usage(argv);
				fprintf(stderr, "\nVertical stencil kernel needs to be of size -- 9 or 13 or 15\n");
				exit(1);
			}
		}

		//Open a file for log output
		FILE* fp = fopen(outfile, "w");
		assert(fp != NULL);
		
		//Allocate Input Image structure
		image* inputImage = (image*)malloc(sizeof(image));
		inputImage->rowx = x;
		inputImage->coly = y;

		//Get the padded size for image based on the type of conv direction
		int remainder, to_pad;
		if (dir == 0) {
			inputImage->pad_rowx = inputImage->rowx + 1;
			remainder = y % 4;
			if (remainder == 0) {
				to_pad = 0;
			} else {
				to_pad = 4 - remainder;
			}
			inputImage->pad_coly = inputImage->coly + tsize + to_pad;
		} else {
			remainder = y % 4;
			if (remainder == 0) {
				to_pad = 0;
			} else {
				to_pad = 4 - remainder;
			}
			inputImage->pad_coly = inputImage->coly + to_pad;
			inputImage->pad_rowx = inputImage->rowx + tsize - 1;
		}

		// getPaddedRowCol(inputImage, tsize + to_pad, dir); // tsize is 16, padding each row so that the number of columns is multiples of 4

		//Generate Input image
		inputImage->img_ptr = (short int*)malloc(SIZE_INT * (inputImage->pad_rowx) * (inputImage->pad_coly));
		memset(inputImage->img_ptr, 0, SIZE_INT * (inputImage->pad_rowx) * (inputImage->pad_coly));
		genImage(inputImage);
		
		fprintf(fp, "\nINPUT IMAGE MATRIX -- ZERO PADDED\n");
		printImage(inputImage, fp);

		//Output image
		image* outImage = (image*)malloc(sizeof(image));	
		outImage->rowx = x;
		outImage->coly = y;
		outImage->pad_rowx = inputImage->pad_rowx;
		outImage->pad_coly = inputImage->pad_coly;
		outImage->img_ptr = (short int*)malloc(SIZE_INT * (outImage->pad_rowx) * (outImage->pad_coly));
		memset(outImage->img_ptr, 0, SIZE_INT * (outImage->pad_rowx) * (outImage->pad_coly));

		//Generate a Gaussian co-efficient kernel
		short int* coeff_kernel = (short int *)malloc(SIZE_INT * tsize);
		memset(coeff_kernel, 0, SIZE_INT * tsize);
		genKernel(coeff_kernel, size);

		// padding zero to coeff_kernel
		int i;
		for (i = size; i < tsize; i++)
			coeff_kernel[i] = 0;

		fprintf(fp, "\n1D CO-EFFICIENT KERNEL -- ZERO PADDED\n");
		printKernel(coeff_kernel, tsize, fp);

		//Perform 1D Gaussin Blur at input image based on Convolution direction
		if(!dir){
			if(DEBUG_FLAG)	
				begin_roi();
		
			// down_sample(inputImage);
			blur_filter_hor(inputImage, coeff_kernel, outImage, size);
			if(DEBUG_FLAG)
				end_roi();
		} else {
			if(DEBUG_FLAG)	
				begin_roi();
		
			// down_sample(inputImage);
			blur_filter_ver(inputImage, coeff_kernel, outImage, size);
			if(DEBUG_FLAG)
				end_roi();
		}
		
		/*else{
			if(DEBUG_FLAG)	
				begin_roi();
			blur_filter_ver(inputImage, coeff_kernel, outImage, size);
		}
		*/

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
