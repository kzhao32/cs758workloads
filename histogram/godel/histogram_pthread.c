/* Copyright (c) 2007-2009, Stanford University
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of Stanford University nor the names of its 
*       contributors may be used to endorse or promote products derived from 
*       this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY STANFORD UNIVERSITY ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL STANFORD UNIVERSITY BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/ 

#include <stdio.h>
#include <strings.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <ctype.h>
//#include <omp.h>
#include <pthread.h>
#include "hwtimer.h"

#include "../include/stddefines.h"

#define IMG_DATA_OFFSET_POS 10
#define BITS_PER_PIXEL_POS 28

int swap;      // to indicate if we need to swap byte order of header information
int numThreads = 1;

/* test_endianess
 *
 */
void test_endianess() {
   unsigned int num = 0x12345678;
   char *low = (char *)(&(num));
   if (*low ==  0x78) {
      //dprintf("No need to swap\n");
      swap = 0;
   }
   else if (*low == 0x12) {
      //dprintf("Need to swap\n");
      swap = 1;
   }
   else {
      printf("Error: Invalid value found in memory\n");
      exit(1);
   } 
}

/* swap_bytes
 *
 */
void swap_bytes(char *bytes, int num_bytes) {
   int i;
   char tmp;
   
   for (i = 0; i < num_bytes/2; i++) {
      dprintf("Swapping %d and %d\n", bytes[i], bytes[num_bytes - i - 1]);
      tmp = bytes[i];
      bytes[i] = bytes[num_bytes - i - 1];
      bytes[num_bytes - i - 1] = tmp;   
   }
}

struct histogram_pthread_arg_struct {
    int threadId;
    int red[256];
    int green[256];
    int blue[256];
};

char *fdata;
struct stat finfo;
unsigned short *data_pos;

void* histogramKernel(void* arg) {
	struct histogram_pthread_arg_struct* args = (histogram_pthread_arg_struct*) arg;
    int threadId = args->threadId;
    int* red = args->red;
    int* green = args->green;
    int* blue = args->blue;
	
	int i;
	unsigned char *val;
	
	for (i=*data_pos + finfo.st_size / numThreads * threadId; 
		i < finfo.st_size && i < *data_pos + finfo.st_size / numThreads * (threadId + 1); 
		i += 3) {      
		val = (unsigned char *)&(fdata[i]);
		blue[*val]++;

		val = (unsigned char *)&(fdata[i+1]);
		green[*val]++;

		val = (unsigned char *)&(fdata[i+2]);
		red[*val]++;   
	}
}
   
int main(int argc, char *argv[]) {
      
   int i;
   int fd;
   char * fname;
	int red[256];
	int green[256];
	int blue[256];

   // Make sure a filename is specified
   if (argv[1] == NULL) {
      printf("USAGE: %s <bitmap filename>\n", argv[0]);
      exit(1);
   }
   
   if (argc > 2) {
        //omp_set_num_threads(atoi(argv[2]));
        numThreads = atoi(argv[2]);
    }
   
   fname = argv[1];
   
   // Read in the file
   CHECK_ERROR((fd = open(fname, O_RDONLY)) < 0);
   // Get the file info (for file length)
   CHECK_ERROR(fstat(fd, &finfo) < 0);
   // Memory map the file
   CHECK_ERROR((fdata = (char*) mmap(0, finfo.st_size + 1, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0)) == NULL);
   
   if ((fdata[0] != 'B') || (fdata[1] != 'M')) {
      printf("File is not a valid bitmap file. Exiting\n");
      exit(1);
   }
   
   test_endianess();    // will set the variable "swap"
   
   unsigned short *bitsperpixel = (unsigned short *)(&(fdata[BITS_PER_PIXEL_POS]));
   if (swap) {
      swap_bytes((char *)(bitsperpixel), sizeof(*bitsperpixel));
   }
   if (*bitsperpixel != 24) {    // ensure its 3 bytes per pixel
      printf("Error: Invalid bitmap format - ");
      printf("This application only accepts 24-bit pictures. Exiting\n");
      exit(1);
   }
   
   data_pos = (unsigned short *)(&(fdata[IMG_DATA_OFFSET_POS]));
   if (swap) {
      swap_bytes((char *)(data_pos), sizeof(*data_pos));
   }
   
   int imgdata_bytes = (int)finfo.st_size - (int)(*(data_pos));
   //printf("This file has %d bytes of image data, %d pixels\n", imgdata_bytes, imgdata_bytes / 3);
                                                            
   //printf("Starting sequential histogram\n");                                                            

   
   memset(&(red[0]), 0, sizeof(int) * 256);
   memset(&(green[0]), 0, sizeof(int) * 256);
   memset(&(blue[0]), 0, sizeof(int) * 256);
   
   int threadIndex;
	// Initialize array of thread structures
	pthread_t* threads = (pthread_t*) malloc(sizeof(pthread_t) * numThreads);
	if (threads == NULL) {
		printf("Could not malloc pthread_t\n");
		return EXIT_FAILURE;
	}
		
		
	struct histogram_pthread_arg_struct* pthread_args = (struct histogram_pthread_arg_struct*) malloc(numThreads * sizeof(struct histogram_pthread_arg_struct));
	for (threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
		pthread_args[threadIndex].threadId = threadIndex;
		memset(&(pthread_args[threadIndex].red[0]), 0, sizeof(int) * 256);
		memset(&(pthread_args[threadIndex].green[0]), 0, sizeof(int) * 256);
		memset(&(pthread_args[threadIndex].blue[0]), 0, sizeof(int) * 256);
	}  
	
   //printf("*data_pos = %d; finfo.st_size = %d\n", *data_pos, finfo.st_size);
   //zsim_identify_main_thread();
   //zsim_roi_begin();
   hwtimer_t timer;
   initTimer(&timer);
   startTimer(&timer); // Start the time measurment here before the algorithm starts
   
   //histogramKernel();
   for (threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
		// ************************************************************
		// pthread_create takes 4 parameters
		//  p1: threads(output)
		//  p2: thread attribute
		//  p3: start routine, where new thread begins
		//  p4: arguments to the thread
		// ************************************************************
		if (pthread_create(&threads[threadIndex], 
						   NULL, 
						   &histogramKernel,
						   (void*) &pthread_args[threadIndex])) {
			printf("Could not create thread %d\n", threadIndex);
			return EXIT_FAILURE;
		}
	}
	
	// Wait for each of the threads to terminate with join
	for (threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
		if (pthread_join(threads[threadIndex], NULL)) {
			printf("Could not join thread\n");
			return -1;
		}
	}
	
	for (threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
		for (int index = 0; index < 256; ++index) {
			red[index] += pthread_args[threadIndex].red[index];
			green[index] += pthread_args[threadIndex].green[index];
			blue[index] += pthread_args[threadIndex].blue[index];
		}
	}
   
   stopTimer(&timer);
   //zsim_roi_end();
   /*
   dprintf("\n\nBlue\n");
   dprintf("----------\n\n");
   for (i = 0; i < 256; i++) {
      dprintf("%d - %d\n", i, blue[i]);        
   }
   
   dprintf("\n\nGreen\n");
   dprintf("----------\n\n");
   for (i = 0; i < 256; i++) {
      dprintf("%d - %d\n", i, green[i]);        
   }
   
   dprintf("\n\nRed\n");
   dprintf("----------\n\n");
   for (i = 0; i < 256; i++) {
      dprintf("%d - %d\n", i, red[i]);        
   }

   CHECK_ERROR(munmap(fdata, finfo.st_size + 1) < 0);
   CHECK_ERROR(close(fd) < 0);
   */
   printf("Total Execution time: %lld ns\n", getTimerNs(&timer));
   return 0;
}
