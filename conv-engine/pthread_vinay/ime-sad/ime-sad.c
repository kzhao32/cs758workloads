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
#include <pthread.h>

#include "../include/ime-sad.h"
#include "../include/stddefines.h"

#define SIZE_INT sizeof(int) 


//static int DEBUG_FLAG = 0;


//thread specific structure
typedef struct{
  int* inputImage;
  int* refImage;
  int* diffImage;
  int rowx;
  int coly;
  int num_per_thread;
} thread_args_t;


void usage(char** argv){
	fprintf(stderr, 
			"\nUsage:	%s -n [num threads] -x [Image_Row_count] -y [Image_Column_Count] -d [Image Ref diff 1/0] -o [Output file]\nKernel/Stencil is of fixed size - 4 x 4\n", basename(argv[0]));
}


//*************IME_SAD--KERNEL***************//

//SAD Calculation Kernel
void* SAD_filter(void* args){

  int* inputImage;
  int* refImage;
  int* diffImage;

  thread_args_t* thread_args = (thread_args_t*)args; 

  inputImage = thread_args->inputImage;
  refImage = thread_args->refImage;
  diffImage = thread_args->diffImage;

	//Need to choose first 'size x size' blocks from either image
	int i = 0, j = 0, k, index = 0;
	int final_sad = 0;
	int pix1, pix2;

 	//fprintf(stderr, "\n");
	//Iterate whole image
	for(i = 0; i < thread_args->rowx; i++){

		int matrix_sad = 0;
		for(j = thread_args->coly; j < (thread_args->coly + thread_args->num_per_thread); j++){

			//Iterate through whole image block-wise(conv size) - 4 x 4 2D convolution
			//fprintf(stderr, "CALCULATE SAD - Block with i=%d, j=%d\n", i, j);
			
			int block_sad = 0;
			
			for (k = 0; k < (SIZE * SIZE); k++){
			
			  index = INDEX_2D(i, j, k, thread_args->rowx, thread_args->coly, SIZE); 

				pix1 = *(inputImage + index);
				pix2 = *(refImage + index);

				*(diffImage + index) = abs(pix1 - pix2);

				block_sad += abs(pix1 - pix2);
			}//inner for
		
			matrix_sad += block_sad;

		}//first outer loop

		final_sad += matrix_sad;

	}//second outer loop

  uint64_t* ret_val = (uint64_t*)malloc(sizeof(uint64_t));

  *ret_val = (uint64_t) final_sad;
	//return final_sad;

  pthread_exit((void*)ret_val);

}

//*************IME_SAD--KERNEL***************//

//MAIN 
int main(int argc, char** argv){
		
		//Use defaults if user does not specify image and filter sizes
		int rowx = 16, coly = 16, diff = 1;
    int i = 0;
		char* outfile = "sad.out";

    int num_threads = 1;
    

		//Parse the arguments
		int r;
		while((r = getopt(argc, argv, "n:x:y:d:o:h")) != -1)
		{
			switch(r) {
        case 'n':
          num_threads = atoi(optarg);
          printf("Executing IME_SAD with %d threads\n",num_threads);
          break;
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

		if(argc != 9){
			usage(argv);
			fprintf(stderr, "\nNo size options given -- Defaulting to image size (16 x 16)\n");
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

    printf("Starting Pthread IME-SAD\n");

    //pthread related
    pthread_t* pid;
    pthread_attr_t attr;
    thread_args_t* args;
    
    //intialize
    pthread_attr_init(&attr);
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);

    //Allocated threads
    CHECK_ERROR( (pid = (pthread_t*)malloc(sizeof(pthread_t) * num_threads)) == NULL);

    //alolocate thread related structures
    CHECK_ERROR( (args = (thread_args_t*)malloc(sizeof(thread_args_t) * num_threads)) == NULL);

    //int total_elems = img_rowx * img_coly; 
    int total_elems = img_coly; 

    int num_per_thread = total_elems / num_threads; 

    printf("Num cols per thread:%d\n", num_per_thread);

    //starting col
    int thread_col = 0;

    //Portions of image to pthreads
    for(i = 0; i < num_threads; i++){
    
      args[i].inputImage = inputImage;
      args[i].refImage = refImage;
      args[i].diffImage = diffImage;
      args[i].rowx = img_rowx;
      args[i].coly = thread_col;
      args[i].num_per_thread = num_per_thread;

      printf("Start col: %d, thread_id: %d\n", thread_col, i);
      //update
      thread_col += num_per_thread;
    

      pthread_create(&(pid[i]), &attr, SAD_filter, (void*)(&(args[i])));
    }
    
    //return val
    int final_sad = 0;
    void* retval;

	  //pthread_join
    for(i = 0; i < num_threads; i++){
      pthread_join(pid[i], &retval);
    }

    final_sad += *((int*)retval);

		fprintf(fp, "\nDIFF IMAGE MATRIX\n");
		printImage(diffImage, rowx, coly, fp);
		
		if(!final_sad){
			fprintf(stderr, "\nSAD OUTPUT VALUE=%d: IMAGES ARE SAME\n\n", final_sad);
			fprintf(fp, "\nSAD OUTPUT VALUE=%d: IMAGES ARE SAME\n", final_sad);
		}else{
			fprintf(stderr, "\nSAD OUTPUT VALUE=%d: IMAGES ARE DIFFERENT\n\n", final_sad);
			fprintf(fp, "\nSAD OUTPUT VALUE=%d: IMAGES ARE DIFFERENT\n", final_sad);
		}
	
		fclose(fp);
		free(inputImage);
		free(refImage);
		free(diffImage);

		return 0;
}
			
