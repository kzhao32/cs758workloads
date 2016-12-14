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
#include <pthread.h>
#include "hwtimer.h"

#include "sift-dog.h"
#include "sim.h"

#define SIZE_INT sizeof(int) 

static int DEBUG_FLAG = 0;
int NumProcs = 1;

void usage(char** argv){
	fprintf(stderr, 
			"\nUsage:	%s -x [Image_Row_count] -y [Image_Column_Count] -s [Stencil 1D Size (9, 13, 15)] -d [Direction Horizantal(0)/vertical(1)] -o [Output file]\n", basename(argv[0]));
}

struct DOG_pthread_arg_struct {
    int threadId;
    image* Image1;
    image* Image2;
    image* dogImage;
};

//*************SIFT_DOG--KERNEL***************//	
__attribute__ ((noinline))
//void dog_filter(image* Image1, image* Image2, image* dogImage){
void* dog_filter(void* arg) {
    struct DOG_pthread_arg_struct* args = arg;
    int threadId = args->threadId;
    image* Image1 = args->Image1;
    image* Image2 = args->Image2;
    image* dogImage = args->dogImage;
    
	//Do the MATRIX subtraction of 2 matrices
	int i, j, k, index;
	int pix1, pix2;

	//fprintf(stderr, "\n");
	//Iterate whole images
	//for(i = 0; i < (Image1->rowx); i++){
    for(i = (Image1->rowx) / NumProcs * threadId; i < (Image1->rowx) && i < (Image1->rowx) / NumProcs * (threadId + 1); i++){
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

            if((size != 9) && (size != 13) && (size != 15)){
                usage(argv);
                fprintf(stderr, "\nStencil Kernel needs to be of size -- 9 or 13 or 15\n");
                exit(1);
            }
        }
        
        printf("%s -x %d -y %d -p %d ", basename(argv[0]), x, y, NumProcs);

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

        int threadIndex;
        // Initialize array of thread structures
        pthread_t* threads = (pthread_t*) malloc(sizeof(pthread_t) * NumProcs);
        if (threads == NULL) {
            printf("Could not malloc pthread_t\n");
            return EXIT_FAILURE;
        }
        
        struct DOG_pthread_arg_struct* pthread_args = malloc(NumProcs * sizeof(struct DOG_pthread_arg_struct));
        for (threadIndex = 0; threadIndex < NumProcs; ++threadIndex) {
            pthread_args[threadIndex].threadId = threadIndex;
            pthread_args[threadIndex].Image1 = outImage1;
            pthread_args[threadIndex].Image2 = outImage2;
            pthread_args[threadIndex].dogImage = dogImage;
        }   
        
		//Now call SIF-DOG Kernel
		if(DEBUG_FLAG)
            begin_roi();
        
        hwtimer_t timer;
        initTimer(&timer);
        startTimer(&timer); // Start the time measurment here before the algorithm starts
		//dog_filter(outImage1, outImage2, dogImage);
        for (threadIndex = 0; threadIndex < NumProcs; ++threadIndex) {
            // ************************************************************
            // pthread_create takes 4 parameters
            //  p1: threads(output)
            //  p2: thread attribute
            //  p3: start routine, where new thread begins
            //  p4: arguments to the thread
            // ************************************************************
            if (pthread_create(&threads[threadIndex], 
                               NULL, 
                               &dog_filter,
                               (void*) &pthread_args[threadIndex])) {
                printf("Could not create thread %d\n", threadIndex);
                return EXIT_FAILURE;
            }
        }
        
        // Wait for each of the threads to terminate with join
        for (threadIndex = 0; threadIndex < NumProcs; ++threadIndex) {
            if (pthread_join(threads[threadIndex], NULL)) {
                printf("Could not join thread\n");
                return -1;
            }
        }
		stopTimer(&timer); // End the time measuremnt here since the algorithm ended

		if(DEBUG_FLAG)
            end_roi();

		//Print the final dog Image
		fprintf(fp, "\nDoG OUTPUT IMAGE MATRIX\n");
        //dogImage[0];
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

        //printf("Total Execution time: %lld ns\n", getTimerNs(&timer));
        printf("%lld\n", getTimerNs(&timer));
        
		return 0;

}
