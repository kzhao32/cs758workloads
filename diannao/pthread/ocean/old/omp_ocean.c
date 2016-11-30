#include <stdio.h>
#include <omp.h>

void omp_ocean (int **grid[2], int xdim, int ydim, int timesteps)
{
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
	int yIndex, xIndex;
	for (timeStepIndex = 0; timeStepIndex < timesteps; ++timeStepIndex) {
		#pragma omp parallel for \
		  shared(grid,xdim,ydim) \
		  private(xIndex,yIndex) \
		  schedule(dynamic)
		// modify line above to switch between dynamic, static, and different chunk sizes
		for (xIndex = 1; xIndex < xdim-1; ++xIndex) {
			for (yIndex = 1; yIndex < ydim-1; ++yIndex) {
				// start from 1 and end at dim-1 because borders dont change
				grid[1][xIndex][yIndex] = 
					(
						grid[0][xIndex    ][yIndex    ] + // center
						grid[0][xIndex    ][yIndex - 1] + // top
						grid[0][xIndex    ][yIndex + 1] + // bottom
						grid[0][xIndex - 1][yIndex    ] + // left
						grid[0][xIndex + 1][yIndex    ]   // right
					) / 5;								  // average
			}
		}
		// JOIN occurs here, with an implicit barrier
		// end of Parallel
		// swap the grid by switching the pointers
		int** temp = grid[0];
		grid[0] = grid[1];
		grid[1] = temp;
	}
}

