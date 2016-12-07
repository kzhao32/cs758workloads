#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>

int NumProcs;
int **grid[2];
int timesteps, xdim, ydim;

pthread_mutex_t   SyncLock; /* mutex */
pthread_cond_t    SyncCV; /* condition variable */
int               SyncCount; /* number of processors at the barrier so far */
pthread_mutex_t   ThreadLock; /* mutex */

/* Ad-hoc Barrier Code. This function will cause all the threads to synchronize
 * with each other.  What happens is that the mutex lock is used to control
 * access to the condition variable & SyncCount variable.  SyncCount is
 * initialized to 0 in the beginning, and when barrier is called by each
 * thread, SyncCount is incremented.  When it reaches NumProcs, the
 * number of threads/processors, a conditional broadcast is sent out which
 * wakes up all the threads.  The bad part is that each thread will then
 * contend over the mutex lock, SyncLock, and will be released sequentially.
 *
 * see man for further descriptions about cond_broadcast, cond_wait, etc. 
 *
 * Barrier locks could also be implemented in many other ways, using
 * semaphores, and other sync. functions
 */
void Barrier()
{
	int ret;

	pthread_mutex_lock(&SyncLock); /* Get the thread lock */
	SyncCount++;
	if(SyncCount == NumProcs) {
		ret = pthread_cond_broadcast(&SyncCV);
		SyncCount = 0;	
		assert(ret == 0);
	} else {
		ret = pthread_cond_wait(&SyncCV, &SyncLock); 
		assert(ret == 0);
	}
	pthread_mutex_unlock(&SyncLock);
}

/* The function which is called once the thread is allocated */
void* pthreadKernel(void* arg) {
	/* each thread has a private version of local variables */
	int threadId = (long) arg;
	//printf("hello world, im thread %d, im going to start at %d and end %d\n", threadId, 1 + threadId * ((xdim-1) / NumProcs), 1 + (threadId + 1) * ((xdim-1) / NumProcs));
	int timeStepIndex, xIndex, yIndex;

	/////////////////////// Execute Job /////////////////////////////////
	for (timeStepIndex = 0; timeStepIndex < timesteps; ++timeStepIndex) {
		for (xIndex = 1 + threadId * ((xdim-1) / NumProcs); xIndex < 1 + (threadId + 1) * ((xdim-1) / NumProcs); ++xIndex) {
			for (yIndex = 1; yIndex < ydim-1; ++yIndex) {
				// start from 1 and end at dim-1 because borders dont change
				//grid[1][xIndex][yIndex] = 0;
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
		Barrier();
		// end of Parallel
		// swap the grid by switching the pointers
		if (threadId == 0) {
			int** temp = grid[0];
			grid[0] = grid[1];
			grid[1] = temp;
		}
		Barrier();
	}
}

int pthread_ocean (int myNumProcs, int **myGrid[2], int myXdim, int myYdim, int myTimesteps) {
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
	
	// move data pointers to shared global pointers, so every thread has access
	NumProcs = myNumProcs;
	grid[0] = myGrid[0];
	grid[1] = myGrid[1];
	xdim = myXdim;
	ydim = myYdim;
	timesteps = myTimesteps;
	
	pthread_attr_t attr;
	int ret;
	long threadIndex;
	
	/* Initialize array of thread structures */
	pthread_t* threads = (pthread_t*) malloc(sizeof(pthread_t) * NumProcs);
	if (threads == NULL) {
		printf("Could not malloc pthread_t\n");
		return EXIT_FAILURE;
	}
	
	/* Init condition variables  and locks */
	ret = pthread_cond_init(&SyncCV, NULL);
	assert(ret == 0);
	ret = pthread_mutex_init(&SyncLock, NULL);
	assert(ret == 0);
	SyncCount = 0;

	for (threadIndex = 0; threadIndex < NumProcs; ++threadIndex) {
		/* ************************************************************
		* pthread_create takes 4 parameters
		*  p1: threads(output)
		*  p2: thread attribute
		*  p3: start routine, where new thread begins
		*  p4: arguments to the thread
		* ************************************************************ */
		if (pthread_create(&threads[threadIndex], NULL, pthreadKernel, (void*) threadIndex)) {
			printf("Could not create thread %d\n", threadIndex);
			return EXIT_FAILURE;
		}
	}
	
	/* Wait for each of the threads to terminate with join */
	for (threadIndex = 0; threadIndex < NumProcs; ++threadIndex) {
		if (pthread_join(threads[threadIndex], NULL)) {
			printf("Could not join thread\n");
			return -1;
		}
	}
			
	// swap back when done so myGrid[0] is results and myGrid[1] is scratch
	myGrid[0] = grid[0];
	myGrid[1] = grid[1];
	return EXIT_SUCCESS;
}

