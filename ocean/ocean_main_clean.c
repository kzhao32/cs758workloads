#include <stdio.h>
#include <stdlib.h>
#include "sys/mman.h"

extern void sequential_ocean (int** grid[2], int xdim, int ydim, int timesteps);
extern void pthread_ocean (int numProcs, int** grid[2], int xdim, int ydim, int timesteps);

int main(int argc, char* argv[]) {
    int xdim,ydim,timesteps;
	int NumProcs = 1;
    int** grid[2];
    int i,j,t;
    
    /********************Get the arguments correctly (start) **************************/
    /* 
    Five input Arguments to the program
    1. number of processes/threads
    2. X Dimension of the grid
    3. Y dimension of the grid
    4. number of timesteps the algorithm is to be performed
	5. Optionally the number of threads
    */
    if (argc < 4) {
        printf("The Arguments you entered are wrong.\n");
        printf("%s <x-dim> <y-dim> <timesteps> [numProcesors]\n", argv[0]);
        return EXIT_FAILURE;
    } else {
		xdim = atoi(argv[1]);
        ydim = atoi(argv[2]);
        timesteps = atoi(argv[3]);
		if (argc > 4) {
			NumProcs = atoi(argv[4]);
		}
    }
    ///////////////////////Get the arguments correctly (end) //////////////////////////

    /*********************create the grid as required (start) ************************/
    /*
    The grid needs to be allocated as per the input arguments and randomly initialized.
    Remember during allocation that we want to guarantee a contiguous block, hence the
    nasty pointer math.

    // TA NOTE: If you dislike this scheme and want to say, allocate a single contiguous
    // 1D array, and do the pointer arithmetic yourself during the iteration, you may do so.

    To test your code for correctness please comment this section of random initialization.
    */
    grid[0] = (int**) malloc(ydim*sizeof(int*));
    grid[1] = (int**) malloc(ydim*sizeof(int*));
    int* temp = (int*) malloc(xdim*ydim*sizeof(int));
    int* other_temp = (int*) malloc(xdim*ydim*sizeof(int));
    if (!grid[0] || !grid[1] || !grid[2] || !temp || !other_temp) {
		printf("\terror: malloc failed\n");
		return EXIT_FAILURE;
	}
    // Force xdim to be a multiple of 64 bytes.
    for (i=0; i<ydim; i++) {
        grid[0][i] = &temp[i*xdim];
        grid[1][i] = &other_temp[i*xdim];
    }
    for (i=0; i<ydim; i++) {
        for (j=0; j<xdim; j++) {
			
            if (i == 0 || j == 0 || i == ydim - 1 || j == xdim - 1) {
                grid[0][i][j] = 1000;
            } else {
                grid[0][i][j] = 500;
            }
            //grid[0][i][j] = rand();
            
            grid[1][i][j] = grid[0][i][j];
        }
    }
	printf("%dx%d,%d", xdim, ydim, NumProcs);
    ///////////////////////create the grid as required (end) //////////////////////////

    /********************* run ocean *************************************************/
	if (NumProcs == 1) {
		sequential_ocean(grid, xdim, ydim, timesteps);
	} else {
		pthread_ocean(NumProcs, grid, xdim, ydim, timesteps);
	}
	return EXIT_SUCCESS; 
} // end program early, since doing correctness comparison in gem5 is extremely slow

/*
	//sequential_ocean(grid, xdim, ydim, timesteps); // skip sequential since gem5 is slow

    //stopTimer(&timer); // End the time measuremnt here since the algorithm ended

    //Do the time calcuclation
    //long long sequential_time = getTimerNs(&timer);
	//printf("%dx%d,%d,%lld,", xdim, ydim, NumProcs, getTimerNs(&timer));
	
    // reset matrix for omp implementation //////////////////////////////////////
    
    //******************** run omp_ocean and display time ************************

   // startTimer(&timer); // Start the time measurment here before the algorithm starts

	//pthread_ocean(NumProcs, grid, xdim, ydim, timesteps);

    //stopTimer(&timer); // End the time measuremnt here since the algorithm ended

    //Do the time calcuclation

    //printf("%lld\n", getTimerNs(&timer));

	//printf("printing Sequential results\n");
	//printGrid(grid[2], xdim, ydim);
	//printf("printing mpi results\n");
	//printGrid(grid[0], xdim, ydim);
    //********************* show that the result matches ************************
    int numFailures = 0;
    for (i = 1; i < ydim-1; i++) {
        for (j = 1; j < xdim-1; j++) {
			if (grid[0][i][j] != grid[2][i][j]) {
				//printf("Test FAILED! grid[0][i][j] = %d; grid[2][i][j] = %d\n", grid[0][i][j], grid[2][i][j]);
				++numFailures;
			}
        }
    }
    
    // Free the memory we allocated for grid
    free(temp);
    free(other_temp);
	free(yet_another_temp);
    free(grid[0]);
    free(grid[1]);
	free(grid[2]);
    if (numFailures == 0) {
		printf("\tTest passed\n");
		return EXIT_SUCCESS;
	} else {
		printf("\tTest FAILED! numFailures = %d\n", numFailures);
		return EXIT_FAILURE;
	}
}
*/
