#include <iostream>
#include "dnn.hpp"
#include <omp.h>
#include <assert.h>
#include "hwtimer.h"

using namespace std;

// Problem Size
//#define Nn 100  // Number of Output Layers
//#define Ni 200  // Number of Input  Layers

#ifndef Nn
#define Nn 128  // Number of Output Layers
#define Ni 224  // Number of Input  Layers
#endif

#ifndef Tii
// Tiling Sizes
#define Tnn 32  
#define Tii 32
//#define Tn 5
//#define Ti 25
#define Tn 16
#define Ti 16
#endif

//Arrays:
VTYPE synapse[Nn][Ni];
VTYPE neuron_i[Ni];
VTYPE neuron_n[Nn],    neuron_n2[Nn];
int calc = 0;

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

void fill_classifier(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni]) {
    for(int n = 0; n < Nn; ++n) {
        for(int i = 0; i < Ni; ++i) {
            synapse[n][i] = n*Ni+i;
        }
    }
    for(int i = 0; i < Ni; ++i) {
        neuron_i[i] = i;
    }
}
       
void* classifier_layer_blocked_pthread(void* arg) {
    int threadId = *(int*) arg;
    delete (int*)arg;
    int total_calc=0;
    VTYPE sum[Nn]={0};
    //for (int nnn = 0; nnn < Nn; nnn += Tnn) { // tiling for output neurons;
    for (int nnn = Nn/NumProcs*threadId; nnn < Nn && nnn < Nn/NumProcs*(threadId+1); nnn += Tnn) { // tiling for output neurons;
        for (int iii = 0; iii < Ni; iii += Tii) { // tiling for input neurons;
            for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
                //for (int n = nn; n < nn + Tn; n++) {
                //    cout << "i-n" << n << " " << nn+Tn << "\n";
                //    sum[n] = 0;
                //}
                for (int ii = iii; ii < iii + Tii; ii += Ti) {
                    //total_calc++;

                    // — Original code —
                    for (int n = nn; n < nn + Tn; n++) {
                        VTYPE sum_sc=0;
                        for (int i = ii; i < ii + Ti; i++) {
                            sum_sc += (synapse[n][i] * neuron_i[i])>>1;
                            //sum_sc += synapse[n][i] * i;
                        }
                        sum[n]+=sum_sc>>1;
                    }
                }
            }
        }
        for (int nn = nnn; nn < nnn + Tnn; nn++) {
            neuron_n[nn] = sigmoid(sum[nn]);
        }
    }
    //calc += total_calc;
}

int classifier_layer_blocked(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], 
                              VTYPE (&neuron_n)[Nn]) {
    int total_calc=0;
    VTYPE sum[Nn]={0};
    for (int nnn = 0; nnn < Nn; nnn += Tnn) { // tiling for output neurons;
        for (int iii = 0; iii < Ni; iii += Tii) { // tiling for input neurons;
            for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
                //for (int n = nn; n < nn + Tn; n++) {
                //    cout << "i-n" << n << " " << nn+Tn << "\n";
                //    sum[n] = 0;
                //}
                for (int ii = iii; ii < iii + Tii; ii += Ti) {
                    //total_calc++;

                    // — Original code —
                    for (int n = nn; n < nn + Tn; n++) {
                        VTYPE sum_sc=0;
                        for (int i = ii; i < ii + Ti; i++) {
                            sum_sc += (synapse[n][i] * neuron_i[i])>>1;
                            //sum_sc += synapse[n][i] * i;
                        }
                        sum[n]+=sum_sc>>1;
                    }
                }
            }
        }
        for (int nn = nnn; nn < nnn + Tnn; nn++) {
            neuron_n[nn] = sigmoid(sum[nn]);
        }
    }
    return total_calc;
}

void* classifier_layer_pthread(void* arg) {
    int threadId = *(int*) arg;
    delete (int*)arg;
    //int total_calc=0;
    // — Original code —
    //for (int n = 0; n < Nn; n++) {
    for (int n = (Nn/NumProcs)*threadId; n < Nn && n < (Nn/NumProcs)*(threadId+1); n++) {
        VTYPE temp=0;
        for (int i = 0; i < Ni; i++) {
            temp += (synapse[n][i] * neuron_i[i])>>1;
        }
        neuron_n[n] = sigmoid(temp);
        //    total_calc++;
    }
    //return total_calc;
}

int classifier_layer(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], VTYPE (&neuron_n)[Nn]) {
    int total_calc=0;
    // — Original code —
    for (int n = 0; n < Nn; n++) {
        VTYPE temp=0;
        for (int i = 0; i < Ni; i++) {
            temp += (synapse[n][i] * neuron_i[i])>>1;
        }
        neuron_n[n] = sigmoid(temp);
        //    total_calc++;
    }
    return total_calc;
}

int main(int argc, char** argv) {
    fill_classifier(synapse,neuron_i);
    
    if (argc==2 || argc==3) {
        //omp_set_num_threads(argv[2]);
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
        // } else if(argc==2 && string(argv[1])=="perf") {
    } else if(argc==3) {
        //int calc = classifier_layer_blocked(synapse,neuron_i,neuron_n);
        for (threadIndex = 0; threadIndex < NumProcs; ++threadIndex) {
            // ************************************************************
            // pthread_create takes 4 parameters
            //  p1: threads(output)
            //  p2: thread attribute
            //  p3: start routine, where new thread begins
            //  p4: arguments to the thread
            // ************************************************************
            if (atoi(argv[1]) % 2 == 0) {
                if (pthread_create(&threads[threadIndex], 
                    NULL, 
                    &classifier_layer_pthread, 
                    new int(threadIndex))) {
                    printf("Could not create thread %d\n", threadIndex);
                    return EXIT_FAILURE;
                }
            } else {
                if (pthread_create(&threads[threadIndex], 
                    NULL, 
                    &classifier_layer_blocked_pthread, 
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
        
        if (atoi(argv[1]) >= 2) {
            classifier_layer(synapse,neuron_i,neuron_n2);
            compare(neuron_n,neuron_n2,Nn);
            cout << "mults: " << Nn*Ni <<  " sigmoids: " << Nn << "\n";
        }
        //int calc = classifier_layer_blocked();
        //if(calc > 0) {
        //    cout << "calc: " << calc << "\n";
        //}
        //cout << "Perf Run Complete\n";
    } else {
        cout << "incorrect usage\n";
    }
    stopTimer(&timer); // End the time measuremnt here since the algorithm ended
    end_roi();
    printf("Total Execution time: %lld ns\n", getTimerNs(&timer));
}

