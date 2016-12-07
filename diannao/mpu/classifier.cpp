#include <iostream>
#include <assert.h>
#include <string.h>
#include "dnn.hpp"


#include <omp.h>
//#include "hwtimer.h"

#include "mpu.h"
#include "mpu_types.h"
#include "zsim_hooks.h"
#include "../include/stddefines.h"

using namespace std;

// Problem Size
//#define Nn 100  // Number of Output Layers
//#define Ni 200  // Number of Input  Layers


#define MAX_VAULTS 16

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
//VTYPE synapse[Nn][Ni];
//VTYPE neuron_i[Ni];
//VTYPE neuron_n[Nn],    neuron_n2[Nn];
int calc = 0;

static int NumProcs = 128;

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



typedef struct {
  VTYPE* synapse;
  VTYPE* neuron_i;
  VTYPE* neuron_n;
} args_t;


void fill_classifier_synapse(VTYPE* synapse) {
    for(int n = 0; n < Nn; ++n) {
        for(int i = 0; i < Ni; ++i) {
            synapse[n * Ni + i] = n*Ni+i;
        }
    }
}

//global vars initialization
void initialize_global_vars(VTYPE* neuron_i){
    
  for(int i = 0; i < Ni; ++i) {
        neuron_i[i] = i;
  }
}
       
//init args for arg structures
int initArgs(unsigned char* data, 
             int            length, 
             int            thread_id,
             int            offset_neuron_n) {
    
  int i;
  int num_pages_args = (sizeof(args_t) - 1) / MPU_PAGE_SIZE + 1;

  //Allocate arguments in the vault where most of the 
  //data pointed to by the arguments (if any) resides
    
  int args_vault_id = thread_id % MAX_VAULTS;
  int args_page_id = MPU_CreatePages(num_pages_args, MPU_ALLOC_CONTIGUOUS, args_vault_id);
  args_t* args = (args_t *)MPU_GetPageAddress(args_page_id);
    
  args->synapse = (VTYPE*)data;

  args->neuron_n = (VTYPE*)(data + offset_neuron_n);
   
  return args_page_id;
} 

#if 0
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
#endif

////MPU specific code
void* classifier_layer_MPU(void* arg) {
//    int threadId = *(int*) arg;
//    delete (int*)arg;
//    //int total_calc=0;
//    // — Original code —
//    //for (int n = 0; n < Nn; n++) {
//    for (int n = (Nn/NumProcs)*threadId; n < Nn && n < (Nn/NumProcs)*(threadId+1); n++) {
//        VTYPE temp=0;
//        for (int i = 0; i < Ni; i++) {
//            temp += (synapse[n][i] * neuron_i[i])>>1;
//        }
//        neuron_n[n] = sigmoid(temp);
//        //    total_calc++;
//    }
//    //return total_calc;
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
    
    if (argc==2 || argc==3) {
        //omp_set_num_threads(argv[2]);
        NumProcs = atoi(argv[2]);
    }
    
    //MPU related stuff
   
    VTYPE* synapse = (VTYPE*)malloc(sizeof(VTYPE) * Ni * Nn);
    VTYPE* neuron_n = (VTYPE*)malloc(sizeof(VTYPE) * Nn);
    VTYPE* global_neuron_i;

    //Initialize data
    fill_classifier_synapse(synapse);

    //Allocate sharded data on each vault
    int synapse_size_per_thread  = Nn * Ni / NumProcs * sizeof(VTYPE);
    int neuron_n_size_per_thread = Nn / NumProcs * sizeof(VTYPE);
   
    int neuron_i_size_per_thread = Ni * sizeof(VTYPE);

    int num_bytes_required_per_thread = synapse_size_per_thread + neuron_n_size_per_thread;
    int num_pages_required_per_thread = (num_bytes_required_per_thread - 1) / MPU_PAGE_SIZE + 1;
  
    //Allocating global variables
    
    int num_global_bytes_required = sizeof(VTYPE) * Ni;
    int num_global_pages_required = (num_global_bytes_required - 1) / MPU_PAGE_SIZE + 1;
    int global_page_id = MPU_CreatePages(num_global_pages_required, MPU_ALLOC_CONTIGUOUS, 0);
    int* global_vars = (int*)MPU_GetPageAddress(global_page_id);
   
    global_neuron_i = (VTYPE*)global_vars; 

    initialize_global_vars(global_neuron_i);

    //Allocate MPU vault space
    unsigned char** data_address = (unsigned char**) malloc(NumProcs*sizeof(unsigned char*));
    int* data_page_id = (int*) malloc(NumProcs * sizeof(int));
    int* data_length = (int*) malloc(NumProcs * sizeof(int));

    for (int threadIndex = 0; threadIndex < NumProcs; ++threadIndex) {
        int page_id = MPU_CreatePages(num_pages_required_per_thread, MPU_ALLOC_CONTIGUOUS, threadIndex % MAX_VAULTS);
        data_address[threadIndex] = (unsigned char*) MPU_GetPageAddress(page_id);
        data_page_id[threadIndex] = page_id;
        //data_length[] = num_bytes_required_per_thread; 

        //Copy to vault
        memcpy(data_address[threadIndex],                           synapse,  synapse_size_per_thread);
        memcpy(data_address[threadIndex] + synapse_size_per_thread, global_neuron_i, neuron_i_size_per_thread);
        //memcpy(data_address + synapse_size*sizeof(VTYPE) + neuron_i_size*sizeof(VTYPE), threadIndex, sizeof(int)  );
    }
    
    //Launch MPU kernels
    MPUContext ctx = MPU_CreateContext();
    MPUOpcode classifier_kernel = MPU_OPCODE_INVALID;
    classifier_kernel = MPU_LoadKernel((void*) &classifier_layer_MPU);

    //Allocate mailboxes and argument storage
    int** mboxArray = (int **) malloc(sizeof(int*) * NumProcs);
    int* argPageID = (int*) malloc(sizeof(int) * NumProcs);
    args_t** arg_per_thread = (args_t**) malloc(sizeof(args_t*) * NumProcs);

    //Must call this before ROI begin
    zsim_identify_main_thread();

    zsim_roi_begin();

    for (int threadIndex = 0; threadIndex < NumProcs; ++threadIndex) {
        argPageID[threadIndex] = initArgs(data_address[threadIndex], data_length[threadIndex], threadIndex, synapse_size_per_thread);
        arg_per_thread[threadIndex] = (args_t*) MPU_GetPageAddress(argPageID[threadIndex]);
        mboxArray[threadIndex] = (int*) MPU_Enqueue(ctx, classifier_kernel, (unsigned char*)arg_per_thread[threadIndex]);
        assert(mboxArray[threadIndex] != NULL);
    }

    MPU_Wait(ctx);


    //Copy results back to host
    for (int threadIndex = 0; threadIndex < NumProcs; ++threadIndex) {
        memcpy(neuron_n + neuron_n_size_per_thread * threadIndex, arg_per_thread[threadIndex]->neuron_n, neuron_n_size_per_thread);
    }

    zsim_roi_end();

    //Free mailboxes and args
    for(int threadIndex = 0; threadIndex < NumProcs; ++threadIndex) {
      MPU_FreeMailbox(ctx, mboxArray[threadIndex]);
      MPU_FreePage(argPageID[threadIndex]);
    }
   
    free(mboxArray);
    free(argPageID);
    free(arg_per_thread);


    //if(argc==4) {
    //    // } else if(argc==2 && string(argv[1])=="perf") {
    //} else if(argc==3) {
    //    //int calc = classifier_layer_blocked(synapse,neuron_i,neuron_n);
    //    for (threadIndex = 0; threadIndex < NumProcs; ++threadIndex) {
    //        // ************************************************************
    //        // pthread_create takes 4 parameters
    //        //  p1: threads(output)
    //        //  p2: thread attribute
    //        //  p3: start routine, where new thread begins
    //        //  p4: arguments to the thread
    //        // ************************************************************
    //        if (atoi(argv[1]) % 2 == 0) {
    //            if (pthread_create(&threads[threadIndex], 
    //                NULL, 
    //                &classifier_layer_pthread, 
    //                new int(threadIndex))) {
    //                printf("Could not create thread %d\n", threadIndex);
    //                return EXIT_FAILURE;
    //            }
    //        } else {
    //            if (pthread_create(&threads[threadIndex], 
    //                NULL, 
    //                &classifier_layer_blocked_pthread, 
    //                new int(threadIndex))) {
    //                printf("Could not create thread %d\n", threadIndex);
    //                return EXIT_FAILURE;
    //            }
    //        }
    //    }
    //    
    //    // Wait for each of the threads to terminate with join
    //    for (threadIndex = 0; threadIndex < NumProcs; ++threadIndex) {
    //        if (pthread_join(threads[threadIndex], NULL)) {
    //            printf("Could not join thread\n");
    //            return -1;
    //        }
    //    }
    //    
    //    if (atoi(argv[1]) >= 2) {
    //        classifier_layer(synapse,neuron_i,neuron_n2);
    //        compare(neuron_n,neuron_n2,Nn);
    //        cout << "mults: " << Nn*Ni <<  " sigmoids: " << Nn << "\n";
    //    }
    //    //int calc = classifier_layer_blocked();
    //    //if(calc > 0) {
    //    //    cout << "calc: " << calc << "\n";
    //    //}
    //    //cout << "Perf Run Complete\n";
    //} else {
    //    cout << "incorrect usage\n";
    //}
    //stopTimer(&timer); // End the time measuremnt here since the algorithm ended
    //end_roi();
    //printf("Total Execution time: %lld ns\n", getTimerNs(&timer));


    return 0;
}

