#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <assert.h>

#include "sift_dog.h"
#include "sim.h"

#include "sift_dog_sb.h"
#include "../../../common/include/sb_insts.h"

#define SIZE_INT sizeof(short int) 

//MAIN 
int main(int argc, char** argv){

		SB_CONFIG(sift_dog_sb_config, sift_dog_sb_size);

		long long buffer = 0;
		long long I, F;
		I = 1;
		F = 5;

		int waiting = 0;

		if (waiting) {
			begin_roi();
		}	

		SB_DMA_READ(&I,
			0,
			sizeof(long long),
			1,
			P_sift_dog_sb_I);

		SB_DMA_READ(&F,
			0,
			sizeof(long long),
			1,
			P_sift_dog_sb_F);

		SB_DMA_WRITE(P_sift_dog_sb_R,
			0,
			sizeof(long long),
			1,
			&buffer);

		SB_WAIT_ALL();

		if (waiting)
			end_roi();

		printf("buffer: %lld\n", buffer);

		return 0;
}