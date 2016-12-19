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
#include <pthread.h>

#include "../include/stddefines.h"
#include "mpu.h"
#include "mpu_types.h"
#include "zsim_hooks.h"

#define IMG_DATA_OFFSET_POS 10
#define BITS_PER_PIXEL_POS 28
#define NUM_THREADS 128
#define MAX_VAULTS 16

int swap; // to indicate if we need to swap byte order of header information

typedef struct {
   unsigned char *data;
   long length;
   int red[256];
   int green[256];
   int blue[256];
} args_t;

/* test_endianess
 *
 */
void test_endianess() {
   unsigned int num = 0x12345678;
   char *low = (char *)(&(num));
   if (*low ==  0x78) {
      dprintf("No need to swap\n");
      swap = 0;
   }
   else if (*low == 0x12) {
      dprintf("Need to swap\n");
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

/* calc_hist
 * Function that computes the histogram for the region
 * assigned to each thread
 */
int calc_hist(unsigned char* args) {
   
  args_t *arguments = (args_t *)args;
  unsigned char *val;
  int i;
  
  int* red = arguments->red;
  int* green = arguments->green;
  int* blue = arguments->blue;
  
  //printf("Starting at %ld, doing %ld bytes\n", arguments->data_pos, arguments->length);
  for (i=0; i < arguments->length; i+=3) {
    val = &(arguments->data[i]);
    //printf("blue %d\n", *val);
    blue[*val]++;
    
    val = &(arguments->data[i+1]);
    green[*val]++;
    
    val = &(arguments->data[i+2]);
    red[*val]++;   
  }
  
  //for (i = 0; i < 10; i++) {
  //   dprintf("%d - %d\n", i, blue[i]);        
  //}

  return 0;
}

int initArgs(unsigned char* file_data, 
             int            length, 
             int            thread_id) {

  int i;
  int num_pages_args = ((sizeof(args_t) % MPU_PAGE_SIZE) == 0) ?
                        (sizeof(args_t) / MPU_PAGE_SIZE) :
                       ((sizeof(args_t) / MPU_PAGE_SIZE) + 1);
  //Allocate arguments in the vault where most of the 
  //data pointed to by the arguments (if any) resides
  int args_vault_id = thread_id % MAX_VAULTS;
  int args_page_id = MPU_CreatePages(num_pages_args, MPU_ALLOC_CONTIGUOUS, args_vault_id);
  args_t* args = (args_t *)MPU_GetPageAddress(args_page_id);

  args->data = file_data;
  args->length = length;
  for (i = 0; i < 256; i++) {
    args->red[i] = 0;
    args->green[i] = 0;
    args->blue[i] = 0;
  }

  return args_page_id;
}

int main(int argc, char *argv[]) {
     
  int i, j;
  int fd;
  unsigned char *fdata;
  struct stat finfo;
  char * fname;
  int red[256];
  int green[256];
  int blue[256];

  // Make sure a filename is specified
  if (argv[1] == NULL) {
     printf("USAGE: %s <bitmap filename>\n", argv[0]);
     exit(1);
  }
  
  fname = argv[1];
  
  // Read in the file
  CHECK_ERROR((fd = open(fname, O_RDONLY)) < 0);
  // Get the file info (for file length)
  CHECK_ERROR(fstat(fd, &finfo) < 0);
  // Memory map the file
  CHECK_ERROR((fdata = mmap(0, finfo.st_size + 1, 
     PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0)) == NULL);
   
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
  
  unsigned short *data_pos = (unsigned short *)(&(fdata[IMG_DATA_OFFSET_POS]));
  if (swap) {
     swap_bytes((char *)(data_pos), sizeof(*data_pos));
  }
  
  int imgdata_bytes = (int)finfo.st_size - (int)(*(data_pos));
  int num_pixels = ((int)finfo.st_size - (int)(*(data_pos))) / 3;
  printf("This file has %d bytes of image data, %d pixels\n", imgdata_bytes, num_pixels);

  memset(&(red[0]), 0, sizeof(int) * 256);
  memset(&(green[0]), 0, sizeof(int) * 256);
  memset(&(blue[0]), 0, sizeof(int) * 256);
  
  //Do Data Sharding

  int pixels_copied = 0;
  int req_pixels = num_pixels / NUM_THREADS;
  int excess_pixels = num_pixels % NUM_THREADS;

  //Address and size of each shard that holds the image data
  unsigned char** data_address = (unsigned char **)malloc(NUM_THREADS*sizeof(unsigned char *));
  int* data_page_id = (int *)malloc(NUM_THREADS * sizeof(int));
  int* data_length = (int *)malloc(NUM_THREADS*sizeof(int));
  int vault_id = 0;

  int copy_length;
  unsigned char* data_begin;
  int num_pages_required;
  int page_id;

  //Read through the file, shard data and assign to the vaults
  long curr_pos = (long)(*data_pos);
  for(i = 0; i < NUM_THREADS; i++) {
    copy_length = req_pixels;

    if (excess_pixels > 0) {
       copy_length++;
       excess_pixels--;            
    }
   
    printf("Shard %d size %d\n", i, copy_length);
    data_begin = fdata + curr_pos + pixels_copied*3;
	  pixels_copied += copy_length;

    //Allocate MPU vault space for file data
    //Add 1 extra page in case division results in a fractional number
    num_pages_required = (copy_length*3 / MPU_PAGE_SIZE) + 1;
    page_id = MPU_CreatePages(num_pages_required, MPU_ALLOC_CONTIGUOUS, vault_id);
    data_address[i] = (unsigned char *)MPU_GetPageAddress(page_id);
    data_page_id[i] = page_id;
    data_length[i] = copy_length*3;
    
    //Copy to vault
    memcpy(data_address[i], data_begin, data_length[i]);
   
    //Assign to vaults in round robin fashion
    vault_id = (vault_id + 1) % MAX_VAULTS;
  }

  //Now, prepare to launch MPU kernels

  /* Create Context and Load Kernels */
  MPUContext ctx = MPU_CreateContext();
  MPUOpcode hist_kernel = MPU_OPCODE_INVALID;
  hist_kernel = MPU_LoadKernel((void *)&calc_hist);
	
  //Allocate mailboxes and argument storage
  int** mboxArray = (int **)malloc(sizeof(int *) * NUM_THREADS);
  int* argPageID = (int *)malloc(sizeof(int) * NUM_THREADS);
  args_t** arg = (args_t **)malloc(sizeof(args_t *) * NUM_THREADS);

  //Must call this before ROI begin
  zsim_identify_main_thread();

  zsim_roi_begin();
  
  for(i=0; i<NUM_THREADS; i++) {
    argPageID[i] = initArgs(data_address[i], data_length[i], i);
    arg[i] = (args_t *)MPU_GetPageAddress(argPageID[i]);
    mboxArray[i] = (int *)MPU_Enqueue(ctx, hist_kernel, (unsigned char *)arg[i]);
    assert(mboxArray[i] != NULL);
  }

  MPU_Wait(ctx);
 
  //Reduce results on host
  for (i = 0; i < NUM_THREADS; i++) {
    for (j = 0; j < 256; j++) {
      red[j] += arg[i]->red[j];
      green[j] += arg[i]->green[j];
      blue[j] += arg[i]->blue[j];
    }
  }

  zsim_roi_end();
  
  /* Free all mailboxes and input arguments */
  for(i = 0; i < NUM_THREADS; i++ ) {
    MPU_FreeMailbox(ctx, mboxArray[i]);
    MPU_FreePage(argPageID[i]);
  }
  free(mboxArray);
  free(argPageID);
  free(arg);
  
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
  
  for(i = 0; i < NUM_THREADS; i++ ) {
    MPU_FreePage(data_page_id[i]);
  }
  free(data_page_id);
  free(data_address);
  free(data_length);
  
  return 0;
}
