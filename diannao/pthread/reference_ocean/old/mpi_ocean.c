#include <stdio.h>
#include <mpi.h>

void mpi_ocean (int **grid[2], int xdim, int ydim, int timesteps, int myRank, int NumProcs) {
    /********************************************************
     * algorithm
     *
     * Two grids are passed in through the grid argument.
     * Each grid itself is an int**, which is a pointer to
     * an array of pointers (see the initialization code).
     *
     * Iterate over the grid[0], performing the Ocean
     * algorithm (described in wiki). Write the result to
     * the other grid. Then, swap the grids for the next
     * iteration.
     ******************************************************/
	
	int timeStepIndex;
	int xIndex, yIndex;
	MPI_Status status;
	
	// send entire grid twice so that every process will have the borders in both their grids
	int numberOfTimesToSendGrid;
	for (numberOfTimesToSendGrid = 0; numberOfTimesToSendGrid < 2; ++numberOfTimesToSendGrid) {
		if (myRank == 0) {
				int threadIndex;
				for (threadIndex = 1; threadIndex < NumProcs; ++threadIndex) {
					MPI_Send(grid[numberOfTimesToSendGrid][0], xdim * ydim, MPI_INT, threadIndex, numberOfTimesToSendGrid, MPI_COMM_WORLD);
				}
		} else {
			MPI_Recv(grid[timeStepIndex][0], xdim * ydim, MPI_INT, 0, timeStepIndex, MPI_COMM_WORLD, &status);
		}
	}
	
	for (timeStepIndex = 0; timeStepIndex < timesteps; ++timeStepIndex) {
		if (myRank == 0) {
			int threadIndex;
			for (threadIndex = 1; threadIndex < NumProcs; ++threadIndex) {
				MPI_Send(grid[0][0], xdim * ydim, MPI_INT, threadIndex, 99, MPI_COMM_WORLD);
			}
		} else {
			MPI_Recv(grid[0][0], xdim * ydim, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);
		}
		
		for (xIndex = 1 + myRank * ((xdim - 1) / NumProcs); xIndex < 1 + (myRank+1) * ((xdim - 1) / NumProcs); ++xIndex) {
			for (yIndex = 1; yIndex < ydim - 1; ++yIndex) {
				// start from 1 and end at dim-1 because borders dont change
				//grid[1][xIndex][yIndex] = 0;
				grid[1][xIndex][yIndex] = 
					(
						grid[0][xIndex    ][yIndex    ] + // center
						grid[0][xIndex    ][yIndex - 1] + // top
						grid[0][xIndex    ][yIndex + 1] + // bottom
						grid[0][xIndex - 1][yIndex    ] + // left
						grid[0][xIndex + 1][yIndex    ]   // right
					) / 5;
			}
		}
		
		if (myRank == 0) {
			int threadIndex;
			for (threadIndex = 1; threadIndex < NumProcs; ++threadIndex) {
				MPI_Recv(grid[1][1 + threadIndex * ((xdim-1) / NumProcs)], ydim * ((xdim-1) / NumProcs), MPI_INT, threadIndex, 99, MPI_COMM_WORLD, &status);
			}
			int** temp = grid[0];
			grid[0] = grid[1];
			grid[1] = temp;
		} else {
			MPI_Send(grid[1][1 + myRank * ((xdim-1) / NumProcs)], ydim * ((xdim-1) / NumProcs), MPI_INT, 0, 99, MPI_COMM_WORLD);
		}
	}
}

