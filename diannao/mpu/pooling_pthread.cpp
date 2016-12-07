#include <iostream>
#include <string>
#include "dnn.hpp"
#include "hwtimer.h"
#include <assert.h>

using namespace std;

#define AVG 1

//Problem Size
#ifndef Ny //if Ny is undefined, then assume nothing is defined
    #define Ny 32
    #define Nx 32

    #define Kx 4
    #define Ky 4
    //#define Ni 100  //Input Layers == Ouptut Layers
    #define Ni 128
#endif

//slide increment
#ifndef Sy
    #define Sy Ky
    #define Sx Kx
#endif

#ifndef Tii //Tiling Sizes:
    #define Tii 64
    #define Ti  16
    #define Ty  16
    #define Tx  16
#endif

#define NYPAD (Ny+Ky)
#define NXPAD (Nx+Kx)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)


//Arrays:
//VTYPE  neuron_i[NYPAD][NXPAD][Ni];
//VTYPE  neuron_n[NYSCL][NXSCL][Ni];
//VTYPE neuron_n2[NYSCL][NXSCL][Ni];

VTYPE  (*neuron_i)[NYPAD][NXPAD][Ni];
VTYPE  (*neuron_n)[NYSCL][NXSCL][Ni];
VTYPE (*neuron_n2)[NYSCL][NXSCL][Ni];

int NumProcs = 1;
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

void fill_pooling(VTYPE (&neuron_i)[NYPAD][NXPAD][Ni]) {
    int total=0;
    for(int yy = 0; yy < NYPAD; ++yy) {
        for(int xx = 0; xx < NXPAD; ++xx) {      
            for(int ni = 0; ni < Ni; ++ni) {
                neuron_i[yy][xx][ni] = total+1;
            }
        }
    }
}

int pooling_layer_blocked_pthread(int threadId, 
                                  VTYPE (&neuron_i)[NYPAD][NXPAD][Ni],
                                  VTYPE (&neuron_n)[NYSCL][NXSCL][Ni]) {
    
    //VTYPE (&neuron_i)[NYPAD][NXPAD][Ni] = *neuron_i;
    //VTYPE (&neuron_n)[NYSCL][NXSCL][Ni] = *neuron_n;
    
    int c=0;

    int ii;
    int i;
    int ky;
    int kx;
    VTYPE value[Ni]={0};
    //for (int yy = 0; yy < Ny; yy += Ty) {
    for (int yy = Ny/NumProcs*threadId; yy < Ny && yy < Ny/NumProcs*(threadId+1); yy += Ty) {
        for (int xx = 0; xx < Nx; xx += Tx) {
            for (int iii = 0; iii < Ni; iii += Tii) {
                // — Original code — (excluding ii loop)
                int yout = yy/Sy;
                for (int y = yy; y < yy + Ty; y += Sy) {
                    //int xout = xx/Sx;
                    // if moving parallel for here, then update xout accordingly
                    // cant move parallel for loop here because xout should be set before future iterations start
                    for (int x = xx; x < xx + Tx; x += Sx) {
                    //for (int x = xx + (Tx/NumProcs)*Sx*threadId; x < xx + Tx && x < xx + (Tx/NumProcs)*Sx*(threadId+1); x += Sx) {
                        //#pragma omp parallel for \
                            shared(neuron_i,neuron_n,yout,y,xout,x) \
                            private(ii,i,ky,kx,value)
                        int xout = xx/Sx + (x-xx)/Sx;
                        for (ii = iii; ii < iii + Tii; ii += Ti) {
                        //for (ii = iii + (Tii/NumProcs)*threadId; ii < iii + Tii && ii < iii + (Tii/NumProcs)*(threadId+1); ii += Ti) {
                            for (i = ii; i < ii + Ti; i++) {
                                value[i] = 0;
                            }
                            
                            for (ky = 0; ky < Ky; ky++) {
                                for (kx = 0; kx < Kx; kx++) {
                                    //c++;
                                    for (i = ii; i < ii + Ti; i++) {
                                        #ifdef AVG
                                            value[i] += neuron_i[ky + y][kx + x][i];
                                        #else
                                            value[i] = max(value[i], neuron_i[ky + y][kx + x][i]);
                                        #endif
                                    }
                                }
                            }

                            for (i = ii; i < ii + Ti; i++) {
                                #ifdef AVG
                                    neuron_n[yout][xout][i] = value[i] / (Kx * Ky);
                                #else
                                    neuron_n[yout][xout][i] = value[i];
                                #endif
                            }
                        }
                        //xout++;
                    }
                    yout++;
                }
            }
        }
    }
    return c;
}

void* pooling_layer_blocked_thread_wrapper(void* arg) {
    int threadId = *(int*) arg;
    delete (int*)arg;
    pooling_layer_blocked_pthread(threadId, *neuron_i, *neuron_n);
}

int pooling_layer_blocked(VTYPE (&neuron_i)[NYPAD][NXPAD][Ni],
                          VTYPE (&neuron_n)[NYSCL][NXSCL][Ni]) {
    int c=0;

    VTYPE value[Ni]={0};
    for (int yy = 0; yy < Ny; yy += Ty) {
        for (int xx = 0; xx < Nx; xx += Tx) {
            for (int iii = 0; iii < Ni; iii += Tii) {
                // — Original code — (excluding ii loop)
                int yout = yy/Sy;
                for (int y = yy; y < yy + Ty; y += Sy) {
                    int xout = xx/Sx;
                    for (int x = xx; x < xx + Tx; x += Sx) {

                        for (int ii = iii; ii < iii + Tii; ii += Ti) {
                            for (int i = ii; i < ii + Ti; i++) {
                                value[i] = 0;
                            }

                            for (int ky = 0; ky < Ky; ky++) {
                                for (int kx = 0; kx < Kx; kx++) {
                                    //c++;
                                    for (int i = ii; i < ii + Ti; i++) {
                                        #ifdef AVG
                                            value[i] += neuron_i[ky + y][kx + x][i];
                                        #else
                                            value[i] = max(value[i], neuron_i[ky + y][kx + x][i]);
                                        #endif
                                    }
                                }
                            }

                            for (int i = ii; i < ii + Ti; i++) {
                                #ifdef AVG
                                    neuron_n[yout][xout][i] = value[i] / (Kx * Ky);
                                #else
                                    neuron_n[yout][xout][i] = value[i];
                                #endif
                            }
                        }
                        xout++;
                    }
                    yout++;
                }
            }
        }
    }
    return c;
}

void pooling_layer_pthread(int threadId, 
                           VTYPE (&neuron_i)[NYPAD][NXPAD][Ni],
                           VTYPE (&neuron_n)[NYSCL][NXSCL][Ni]) {
    VTYPE value[Ni]={0};
    int x;
    int ii;
    int i;
    int ky;
    int kx;
    // — Original code —
    int yout = 0;
    int xout = 0;
    //for (int y = 0; y < Ny; y += Sy) {
    for (int y = Ny/NumProcs*threadId; y < Ny && y < Ny/NumProcs*(threadId+1); y += Sy) {
        yout = y / Sy;   
        for (int x = 0; x < Nx; x += Sx) {
        //for (x = (Nx/NumProcs)*threadId; x < Nx && x < (Nx/NumProcs)*(threadId+1); x += Sx) {
            xout = x / Sx;
            for (int i = 0; i < Ni; i++) {
                value[i]=0;
            }
            //Barrier();
            for (ky = 0; ky < Ky; ky++) {
                for (kx = 0; kx < Kx; kx++) {
                    for (i = 0; i < Ni; i++) {
                        #ifdef AVG
                            value[i] += neuron_i[ky + y][kx + x][i];
                        #else
                            value[i] = max(value[i], neuron_i[ky + y][kx + x][i]);
                        #endif
                    }
                }
            }
            //Barrier();
            //if (threadId == 0) {
                for (i = 0; i < Ni; i++) {
                    #ifdef AVG
                        neuron_n[yout][xout][i] = value[i] / (Kx * Ky);
                    #else
                        neuron_n[yout][xout][i] = value[i];
                    #endif
                }
                //xout++;
            //}
            //Barrier();
        }
        //yout++;
    }
}

void* pooling_layer_thread_wrapper(void* arg) {
    int threadId = *(int*) arg;
    delete (int*)arg;
    pooling_layer_pthread(threadId, *neuron_i, *neuron_n);
}

void pooling_layer(VTYPE (&neuron_i)[NYPAD][NXPAD][Ni],
                   VTYPE (&neuron_n)[NYSCL][NXSCL][Ni]) {
                   VTYPE value[Ni]={0};
    // — Original code —
    int yout = 0;
    for (int y = 0; y < Ny; y += Sy) {
        int xout = 0;
        for (int x = 0; x < Nx; x += Sx) {
            for (int i = 0; i < Ni; i++) {
                value[i]=0;
            }

            for (int ky = 0; ky < Ky; ky++) {
                for (int kx = 0; kx < Kx; kx++) {
                    for (int i = 0; i < Ni; i++) {
                        #ifdef AVG
                            value[i] += neuron_i[ky + y][kx + x][i];
                        #else
                            value[i] = max(value[i], neuron_i[ky + y][kx + x][i]);
                        #endif
                    }
                }
            }

            for (int i = 0; i < Ni; i++) {
                #ifdef AVG
                    neuron_n[yout][xout][i] = value[i] / (Kx * Ky);
                #else
                    neuron_n[yout][xout][i] = value[i];
                #endif
            }
            xout++;
        }
        yout++;
    }
}

int main(int argc, char** argv) {

    neuron_i  = (VTYPE (*)[NYPAD][NXPAD][Ni])malloc(NYPAD*NXPAD*Ni*sizeof(VTYPE));
    neuron_n  = (VTYPE (*)[NYSCL][NXSCL][Ni])malloc(NYSCL*NXSCL*Ni*sizeof(VTYPE));
    neuron_n2 = (VTYPE (*)[NYSCL][NXSCL][Ni])malloc(NYSCL*NXSCL*Ni*sizeof(VTYPE));

    fill_pooling(*neuron_i);
    
    if (argc==2 || argc==3) {
        //omp_set_num_threads(atoi(argv[2]));
        NumProcs = atoi(argv[2]);
    }
    
    // Initialize array of thread structures
    pthread_t* threads = (pthread_t*) malloc(sizeof(pthread_t) * NumProcs);
    if (threads == NULL) {
        printf("Could not malloc pthread_t\n");
        return EXIT_FAILURE;
    }
    
    // Init condition variables  and locks
    int ret;
    int threadIndex;
    ret = pthread_cond_init(&SyncCV, NULL);
    assert(ret == 0);
    ret = pthread_mutex_init(&SyncLock, NULL);
    assert(ret == 0);
    SyncCount = 0;
    
    hwtimer_t timer;
    initTimer(&timer);

    begin_roi();
    startTimer(&timer); // Start the time measurment here before the algorithm starts

    if(argc==4) {

        //cout << "Did nothing\n";

        //  } else if(argc==2 && string(argv[1])=="perf") {
    } else if(argc==3) {

        //pooling_layer_blocked();
        for (threadIndex = 0; threadIndex < NumProcs; ++threadIndex) {
            // ************************************************************
            // pthread_create takes 4 parameters
            //  p1: threads(output)
            //  p2: thread attribute
            //  p3: start routine, where new thread begins
            //  p4: arguments to the thread
            // ************************************************************
            if (atoi(argv[1]) == 0) {
                if (pthread_create(&threads[threadIndex], 
                    NULL, 
                    &pooling_layer_thread_wrapper, 
                    new int(threadIndex))) {
                    printf("Could not create thread %d\n", threadIndex);
                    return EXIT_FAILURE;
                }
            } else {
                if (pthread_create(&threads[threadIndex], 
                    NULL, 
                    &pooling_layer_blocked_thread_wrapper, 
                    new int(threadIndex))) {
                    printf("Could not create thread %d\n", threadIndex);
                    return EXIT_FAILURE;
                }
            }
        }
        
        // Wait for each of the threads to terminate with join
        for (threadIndex = 0; threadIndex < NumProcs; ++threadIndex) {
            if (pthread_join(threads[threadIndex], NULL)) {
                printf("Could not join thread\n");
                return -1;
            }
        }
        cout << "Perf Run Complete\n";
    } else {
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
                &pooling_layer_blocked_thread_wrapper, 
                new int(threadIndex))) {
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
        
        //int calc = pooling_layer_blocked();
        //if(calc > 0) {
        //    cout << "calc: " << calc << "\n";
        //}
        pooling_layer(*neuron_i,*neuron_n2);

        compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n2,NYSCL*NXSCL*Ni);
        cout << "adds: " << NYSCL*NXSCL*Ni*Ky*Kx <<  "\n";
        cout << "argc:" << argc << "\n";
        //  cout << "mult-block:  " << calc.first   << " sigmoid-block: " << calc.second  << "\n";
        //  cout << "mult-orig:  "  << calc2.first  << " sigmoid-orig:  " << calc2.second << "\n";
        //
        //  int n_outputs= Ny/Sy * Nx/Sx * Nn;
        //  cout << "mult-correct: " << n_outputs*Ni*Kx*Ky
        //       << " sigmoid-correct: "  << n_outputs << "\n";
    }
    stopTimer(&timer); // End the time measuremnt here since the algorithm ended
    end_roi();
    printf("Total Execution time: %lld ns\n", getTimerNs(&timer));
}
