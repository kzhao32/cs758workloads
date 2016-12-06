#include <iostream>
#include <string>
#include "dnn.hpp"
#include <omp.h>
#include "hwtimer.h"

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

int pooling_layer_blocked_omp(VTYPE (&neuron_i)[NYPAD][NXPAD][Ni],
    VTYPE (&neuron_n)[NYSCL][NXSCL][Ni]) {
    int c=0;

    int ii;
    int i;
    int ky;
    int kx;
    VTYPE value[Ni]={0};
    for (int yy = 0; yy < Ny; yy += Ty) {
        for (int xx = 0; xx < Nx; xx += Tx) {
            for (int iii = 0; iii < Ni; iii += Tii) {
                // — Original code — (excluding ii loop)
                int yout = yy/Sy;
                for (int y = yy; y < yy + Ty; y += Sy) {
                    int xout = xx/Sx;
                    // if moving parallel for here, then use barrier and single before incrementing xout
                    // cant move parallel for loop here because xout should be set before future iterations start
                    for (int x = xx; x < xx + Tx; x += Sx) {
                        #pragma omp parallel for \
                            shared(neuron_i,neuron_n,yout,y,xout,x) \
                            private(ii,i,ky,kx,value)
                        for (ii = iii; ii < iii + Tii; ii += Ti) {
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
                        xout++;
                    }
                    yout++;
                }
            }
        }
    }
    return c;
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

void pooling_layer_omp(VTYPE (&neuron_i)[NYPAD][NXPAD][Ni],
                       VTYPE (&neuron_n)[NYSCL][NXSCL][Ni]) {
                           
    int x;
    int i;
    int ky;
    int kx;
    VTYPE value[Ni]={0};
    // — Original code —
    int yout = 0;
    int xout = 0;
    for (int y = 0; y < Ny; y += Sy) {
        
        #pragma omp parallel for \
            shared(neuron_i,neuron_n,yout,y,xout) \
            private(x,i,ky,kx,value)
        for (x = 0; x < Nx; x += Sx) {
            for (i = 0; i < Ni; i++) {
                value[i]=0;
            }

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

            for (int i = 0; i < Ni; i++) {
                #ifdef AVG
                    neuron_n[yout][xout][i] = value[i] / (Kx * Ky);
                #else
                    neuron_n[yout][xout][i] = value[i];
                #endif
            }
            //xout++;
        }
        //yout++;
    }
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
        omp_set_num_threads(atoi(argv[2]));
        //NumProcs = atoi(argv[2]);
    }

    hwtimer_t timer;
    initTimer(&timer);

    begin_roi();
    startTimer(&timer); // Start the time measurment here before the algorithm starts

    if(argc==4) {

        //cout << "Did nothing\n";

        //  } else if(argc==2 && string(argv[1])=="perf") {
    } else if(argc==3) {
        if (atoi(argv[1]) % 2 == 0) {
            pooling_layer_omp(*neuron_i,*neuron_n);
        } else {
            pooling_layer_blocked_omp(*neuron_i,*neuron_n);
        }
        
        cout << "Perf Run Complete\n";
        if (atoi(argv[1]) >= 2) {
            pooling_layer(*neuron_i,*neuron_n2);
            compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n2,NYSCL*NXSCL*Ni);
            cout << "adds: " << NYSCL*NXSCL*Ni*Ky*Kx <<  "\n";
            cout << "argc:" << argc << "\n";
        }
    } else {
        cout << "incorrect usage\n";
        /*
        int calc = 0;
        pooling_layer_omp(*neuron_i,*neuron_n);
        pooling_layer(*neuron_i,*neuron_n2);

        if(calc > 0) {
            cout << "calc: " << calc << "\n";
        }

        compare((VTYPE*)*neuron_n,(VTYPE*)*neuron_n2,NYSCL*NXSCL*Ni);
        cout << "adds: " << NYSCL*NXSCL*Ni*Ky*Kx <<  "\n";
        cout << "argc:" << argc << "\n";
        */
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

