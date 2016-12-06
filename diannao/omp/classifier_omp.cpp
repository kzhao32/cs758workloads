#include <iostream>
#include "dnn.hpp"
#include <omp.h>
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

int classifier_layer_blocked_omp(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], 
                                 VTYPE (&neuron_n)[Nn]) {
    int total_calc=0;
    VTYPE sum[Nn]={0};
    int nnn;
    int iii;
    int nn;
    int ii;
    int n;
    int i;
    VTYPE sum_sc;
    //cout << "Ni = " << Ni << "; Tii = " << Tii << "; Tn = " << Tn << "; nn = " << nn << endl;
    #pragma omp parallel for \
        shared(sum,synapse,neuron_i) \
        private(nnn,iii,nn,ii,n,sum_sc,i)
    for (nnn = 0; nnn < Nn; nnn += Tnn) { // tiling for output neurons;
        for (iii = 0; iii < Ni; iii += Tii) { // tiling for input neurons;
            for (nn = nnn; nn < nnn + Tnn; nn += Tn) {
                /*        for (int n = nn; n < nn + Tn; n++) {
                cout << "i-n" << n << " " << nn+Tn << "\n";
                sum[n] = 0;
                }*/
                for (ii = iii; ii < iii + Tii; ii += Ti) {
                    //total_calc++;

                    // — Original code —
                    for (n = nn; n < nn + Tn; n++) {
                        sum_sc=0;
                        for (i = ii; i < ii + Ti; i++) {
                            sum_sc += (synapse[n][i] * neuron_i[i])>>1;
                            //sum_sc += synapse[n][i] * i;
                        }
                        sum[n]+=sum_sc>>1;
                    }
                }
            }
        }
        for (nn = nnn; nn < nnn + Tnn; nn++) {
            neuron_n[nn] = sigmoid(sum[nn]);
        }
    }
    return total_calc;
}

int classifier_layer_omp(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], VTYPE (&neuron_n)[Nn]) {
    
    int total_calc=0;
    
    int n; 
    VTYPE temp;
    int i; 
    // — Original code —
    #pragma omp parallel for \
        shared(synapse,neuron_i) \
        private(n,temp,i)
    for (n = 0; n < Nn; n++) {
        VTYPE temp=0;
        for (i = 0; i < Ni; i++) {
            temp += (synapse[n][i] * neuron_i[i])>>1;
        }
        neuron_n[n] = sigmoid(temp);
        //    total_calc++;
    }
    return total_calc;
}

int classifier_layer_blocked(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], 
                             VTYPE (&neuron_n)[Nn]) {
    int total_calc=0;
    VTYPE sum[Nn]={0};
    for (int nnn = 0; nnn < Nn; nnn += Tnn) { // tiling for output neurons;
        for (int iii = 0; iii < Ni; iii += Tii) { // tiling for input neurons;
            for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
                /*        for (int n = nn; n < nn + Tn; n++) {
                cout << "i-n" << n << " " << nn+Tn << "\n";
                sum[n] = 0;
                }*/
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
        omp_set_num_threads(atoi(argv[2]));
        //NumProcs = atoi(argv[2]);
    }
    
    hwtimer_t timer;
    initTimer(&timer);

    begin_roi();
    startTimer(&timer); // Start the time measurment here before the algorithm starts

    if(argc==4) {
        // } else if(argc==2 && string(argv[1])=="perf") {
    } else if(argc==3) {
        //int calc;
        if (atoi(argv[1]) % 2 == 0) {
            classifier_layer_omp(synapse,neuron_i,neuron_n);
        } else {
            classifier_layer_blocked_omp(synapse,neuron_i,neuron_n);
        }
        if (atoi(argv[1]) >= 2) {
            classifier_layer(synapse,neuron_i,neuron_n2);
            // doesnt seem to work for classifier since all 0
            compare(neuron_n,neuron_n2,Nn);
            cout << "mults: " << Nn*Ni <<  " sigmoids: " << Nn << "\n";
        }
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

