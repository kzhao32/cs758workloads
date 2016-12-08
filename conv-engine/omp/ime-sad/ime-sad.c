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
//#include <pthread.h>
#include <omp.h>
#include "hwtimer.h"

#include "ime-sad.h"

#define SIZE_INT sizeof(int) 

static int DEBUG_FLAG = 0;
int NumProcs = 1;
int* sad_outs; 

void usage(char** argv){
	fprintf(stderr, 
			"\nUsage:	%s -x [Image_Row_count] -y [Image_Column_Count] -d [Image Ref diff 1/0] -o [Output file] -p [numProcs]\nKernel/Stencil is of fixed size - 4 x 4\n", basename(argv[0]));
}

//*************IME_SAD--KERNEL***************//

//SAD Calculation Kernel
int SAD_filter(int* inputImage, int* refImage, int* diffImage, int rowx, int coly){
//void* SAD_filter(void* arg){
    
	//Need to choose first 'size x size' blocks from either image
	int i = 0, j = 0, k, index;
	int final_sad = 0;
	int pix1, pix2;
    int matrix_sad;
    int block_sad;

	//fprintf(stderr, "\n");
	//Iterate whole image
    #pragma omp parallel for \
        shared(inputImage,refImage,diffImage,rowx,coly) \
        private(i,j,k,index,pix1,pix2,matrix_sad,block_sad)
    for(i = 0; i < IMAGE_PADDING(rowx); i+=SIZE){
	//for(i = IMAGE_PADDING(rowx) / NumProcs * threadId; i < IMAGE_PADDING(rowx) && i < IMAGE_PADDING(rowx) / NumProcs * (threadId + 1); i+=SIZE){

		matrix_sad = 0;
		for(j = 0; j < IMAGE_PADDING(coly); j+=SIZE){

			//Iterate through whole image block-wise(conv size) - 4 x 4 2D convolution
			//fprintf(stderr, "CALCULATE SAD - Block with i=%d, j=%d\n", i, j);
			
			block_sad = 0;
			
			index = INDEX_2D(i, j, k, rowx, coly, SIZE); 
			for (k = 0; k < (SIZE * SIZE); k++){
			
				index = INDEX_2D(i, j, k, rowx, coly, SIZE); 
				pix1 = *(inputImage + index);
				pix2 = *(refImage + index);

				//*(diffImage + index) = abs(pix1 - pix2);

				block_sad += abs(pix1 - pix2);
			}//inner for
			matrix_sad += block_sad;
		}//first outer loop
		//final_sad += matrix_sad;
        sad_outs[i] = matrix_sad;
	}//second outer loop
    
    for(i = 0; i < IMAGE_PADDING(rowx); i+=SIZE){
        final_sad += sad_outs[i];
    }
    return final_sad;
}

//*************IME_SAD--KERNEL***************//

//MAIN 
int main(int argc, char** argv){
		
		//Use defaults if user does not specify image and filter sizes
		int rowx = 16, coly = 16, diff = 1;
		char* outfile = "sad.out";

		//Parse the arguments
		int r;
		while((r = getopt(argc, argv, "x:y:d:o:p:h")) != -1)
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
            if(argc != 7){
                usage(argv);
                fprintf(stderr, "\nNo size options given -- Defaulting to image size (16 x 16)\n");
            }
        }
        
        printf("%s -x %d -y %d -p %d ", basename(argv[0]), rowx, coly, NumProcs);
        omp_set_num_threads(NumProcs);

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
		srand(time(NULL) + (diff * 20)); //To avoid same image generation
		int* refImage = (int*)malloc(SIZE_INT * img_rowx * img_coly);	
		genImage(refImage, rowx, coly);
		
		fprintf(fp, "\nREF IMAGE MATRIX\n");
		printImage(refImage, rowx, coly, fp);

		//Temporary Diff image
		int* diffImage = (int*)malloc(SIZE_INT * img_rowx * img_coly);
        
		int threadIndex;
        // Initialize array of thread structures
        pthread_t* threads = (pthread_t*) malloc(sizeof(pthread_t) * NumProcs);
        if (threads == NULL) {
            printf("Could not malloc pthread_t\n");
            return EXIT_FAILURE;
        }
        
        sad_outs = (int*) malloc(IMAGE_PADDING(rowx) * sizeof(int));// new int[NumProcs];
        
		//Find the SAD of the images at kernel(convolution) size -- ker_size
        hwtimer_t timer;
        initTimer(&timer);
        startTimer(&timer); // Start the time measurment here before the algorithm starts
		int sad_out = SAD_filter(inputImage, refImage, diffImage, rowx, coly);

        stopTimer(&timer); // End the time measuremnt here since the algorithm ended

        if (DEBUG_FLAG != 0) {
            fprintf(fp, "\nDIFF IMAGE MATRIX\n");
            printImage(diffImage, rowx, coly, fp);
            
            if(!sad_out){
                fprintf(stderr, "\nSAD OUTPUT VALUE=%d: IMAGES ARE SAME\n\n", sad_out);
                fprintf(fp, "\nSAD OUTPUT VALUE=%d: IMAGES ARE SAME\n", sad_out);
            }else{
                fprintf(stderr, "\nSAD OUTPUT VALUE=%d: IMAGES ARE DIFFERENT\n\n", sad_out);
                fprintf(fp, "\nSAD OUTPUT VALUE=%d: IMAGES ARE SAME\n", sad_out);
            }
        }

		fclose(fp);
		free(inputImage);
		free(refImage);
		free(diffImage);
    
        //printf("Total Execution time: %lld ns\n", getTimerNs(&timer));
        printf("%lld\n", getTimerNs(&timer));

		return 0;
}
			
