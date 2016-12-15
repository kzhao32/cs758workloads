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
//#include <pthread.h>
#include <omp.h>
#include "hwtimer.h"
#include "sift-blur.h"

#include "sim.h"
#include "sift-blur.h"

#define SIZE_INT sizeof(int) 

static int DEBUG_FLAG = 0;
int NumProcs = 1;

void usage(char** argv){
	fprintf(stderr, 
			"\nUsage:	%s -x [Image_Row_count] -y [Image_Column_Count] -s [Stencil 1D Size (9, 13, 15)] -d [Direction Horizantal(0)/vertical(1)] -o [Output file] -p [numProcs]\n", basename(argv[0]));
}

struct BLUR_pthread_arg_struct {
    int threadId;
    image* inputImage;
    int* kernel;
    image* outImage;
    int size;
};
    
//*************Down_Sample by 2--KERNEL***************//
void down_sample(image* inputImage){

	int i, j, k;
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

			/*
			//2
			pixel = *(inputImage->img_ptr + index + 1);
			*(inputImage->img_ptr + index + 1) = pixel / 2;

			//3
			pixel = *(inputImage->img_ptr + index + 2);
			*(inputImage->img_ptr + index + 2) = pixel / 2;

			//4
			pixel = *(inputImage->img_ptr + index + 3);
			*(inputImage->img_ptr + index + 3) = pixel / 2;

			//5
			pixel = *(inputImage->img_ptr + index + 4);
			*(inputImage->img_ptr + index + 4) = pixel / 2;

			//6
			pixel = *(inputImage->img_ptr + index + 5);
			*(inputImage->img_ptr + index + 5) = pixel / 2;

			//7
			pixel = *(inputImage->img_ptr + index + 6);
			*(inputImage->img_ptr + index + 6) = pixel / 2;\

			//8
			pixel = *(inputImage->img_ptr + index + 7);
			*(inputImage->img_ptr + index + 7) = pixel / 2;

			//9
			pixel = *(inputImage->img_ptr + index + 8);
			*(inputImage->img_ptr + index + 8) = pixel / 2;

			//10
			pixel = *(inputImage->img_ptr + index + 9);
			*(inputImage->img_ptr + index + 9) = pixel / 2;

			//11
			pixel = *(inputImage->img_ptr + index + 10);
			*(inputImage->img_ptr + index + 10) = pixel / 2;
			
			//12
			pixel = *(inputImage->img_ptr + index + 11);
			*(inputImage->img_ptr + index + 11) = pixel / 2;

			//13
			pixel = *(inputImage->img_ptr + index + 12);
			*(inputImage->img_ptr + index + 12) = pixel / 2;
			
			//14
			pixel = *(inputImage->img_ptr + index + 13);
			*(inputImage->img_ptr + index + 13) = pixel / 2;

			//15
			pixel = *(inputImage->img_ptr + index + 14);
			*(inputImage->img_ptr + index + 14) = pixel / 2;
			
			//16
			pixel = *(inputImage->img_ptr + index + 15);
			*(inputImage->img_ptr + index + 15) = pixel / 2;

		}//j for
*/

	}// i for

}


//index = INDEX_1D_HOR(i, j, k, inputImage->pad_coly, size); 
//pixel_blur += blur_prod;


//*************SIFT_BLUR_HOR--KERNEL***************//
//SIFT blurring Kernel  -- 1D Horizantal Convultion
__attribute__ ((noinline))
void blur_filter_hor(image* inputImage, int* kernel, image* outImage, int size){
//void* blur_filter_hor(void* arg){
	int i, j, k, vec, index;
	int pixel_blur, blur_prod;
	int normalized_blur;
	
	//Iterate whole image -- Horizantal direction
    #pragma omp parallel for \
        shared(inputImage,kernel,outImage) \
        private(i,pixel_blur,index,k,vec,blur_prod)
	for(i = 0; i < (inputImage->pad_rowx); i++){      //Row-wise
	//for(i = (inputImage->pad_rowx) / NumProcs * threadId; i < (inputImage->pad_rowx) / NumProcs * (threadId + 1); i++){      //Row-wise
    	//Vectorizing for 16 wide vector
		//for(j = 0; j < (inputImage->coly); j=j+16){      //Column-wise -- 16 element
			pixel_blur = 0, normalized_blur = 0;
			index = (inputImage->coly * i);							//Image index
			
			//iterate over the kernel size 9 
			for (k = 0; k < size; k++){
				
				//Now multiply the each image pixel with kernel element
				for (vec = 0; vec < (inputImage->coly); vec++){
					blur_prod = (*(inputImage->img_ptr + index + vec + k)) * kernel[k]; //Product
					pixel_blur = blur_prod + (*(outImage->img_ptr + index + vec));
					
					//if(pixel_blur > 255)
						//normalized_blur = (pixel_blur % ((pixel_blur / 255) * 255));    //Normalizing
					
					*(outImage->img_ptr + index + vec) = pixel_blur;
				}//vector for
			
			}//kernel for -- 9 iters	
		
		//}//Next 16 pixels along the same row	
			
	}//row loop
}

//*************SIFT_BLUR_HOR--KERNEL***************//


//*************SIFT_BLUR_VER--KERNEL***************//

//SIFT blurring Kernel  -- 1D Vertical Convultion
void blur_filter_ver(image* inputImage, int* kernel, image* outImage, int size){
//void* blur_filter_ver(void* arg){
	int i, j, k, index, index_out;
	//fprintf(stderr, "\n----1D Vertical Convolution----\n\n");
	int pixel_blur, blur_prod;
	int normalized_blur;

	//Iterate whole image -- Horizantal direction
    #pragma omp parallel for \
        shared(inputImage,kernel,outImage) \
        private(j,i,pixel_blur,normalized_blur,k,index,blur_prod,index_out)
	for(j = 0; j < (inputImage->pad_coly); j++){      //Column-wise pixels
	//for(j = (inputImage->pad_coly) / NumProcs * threadId; j < (inputImage->pad_coly) / NumProcs * (threadId + 1); j++){      //Column-wise pixels

		//fprintf(stderr, "CALCULATING BLUR OF %dth COLUMN OF IMAGE\n", j);
		for(i = 0; i < ((inputImage->pad_rowx) - (size - 1)); i++){      //Rows-wise pixels

			pixel_blur = 0, normalized_blur = 0;
			if(i < (inputImage->rowx)){																		//Avoid unnecessary zero-computation

				for (k = 0; k < size; k++){
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
		while((r = getopt(argc, argv, "x:y:s:d:o:p:h")) != -1)
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
				case 'p':
					NumProcs = atoi(optarg);
					break;
				case 'h':
				  usage(argv);
					exit(1);
					break;
				default:
					usage(argv);
			}
		}

        if (DEBUG_FLAG != 0) {
            if(argc < 9){
                usage(argv);
                fprintf(stderr, "No size options given -- Defaulting to image size (3 x 3) and 1D Stencil Size 9 in Horizantal direction\n");
            }
        }

		if((size != 9) && (size != 13) && (size != 15)){
			usage(argv);
			fprintf(stderr, "\nStencil Kernel needs to be of size -- 9 or 13 or 15\n");
			exit(1);
		}
        
        printf("%s -x %d -y %d -p %d ", basename(argv[0]), x, y, NumProcs);
        omp_set_num_threads(NumProcs);
        
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
                            
            int threadIndex;
            
			if(DEBUG_FLAG)
				begin_roi();
            
            //Find the SAD of the images at kernel(convolution) size -- ker_size
            hwtimer_t timer;
            initTimer(&timer);
            startTimer(&timer); // Start the time measurment here before the algorithm starts
			
            down_sample(inputImage);
			blur_filter_hor(inputImage, coeff_kernel, outImage, size);
            
            stopTimer(&timer); // End the time measuremnt here since the algorithm ended

			if(DEBUG_FLAG)	
				end_roi();
            
            printf("%lld\n", getTimerNs(&timer));
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
			
