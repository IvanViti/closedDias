#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef DOUBLE
 #define REAL double
 #define MAXT 256
#else
 #define REAL double
 #define MAXT 256
// #define REAL float
// #define MAXT 512
#endif



using namespace std;

class parameters {
public:
        double xi,muVar,xVar,yVar,eV,Ec,L,T,alphaOne,alphaTwo,rejection,boltzmann,changeToV,xMoment,highKd,KdFrequency;
        int N,tSteps,nParticles,relax,grabJ,recordLength,whichBox;


        char boxName[256];
        char lineName[256];
        char timeName[256];

};


class vectors {
public:
        REAL *reducedProb,*particles,*probabilities,*potentials,*substrate,*hereP,*hereProb,*herePot,*hereS,*boxR,*hereBoxR,*hereXDiff,*hereYDiff,*Ematrix,*jumpRecord,*tempDos,*tempPar,*tempPot,*invertedDos,*watcher,*aMatrix,*timeRun,*sumRun,*TField,*KdArray;
        REAL *rangeMatrix,*extraArray,*sumArray,*hereSum,*KdArray;
        REAL results[5];
        int *picked;
        int *herePicked;//wtf cuda
        int tStep;
        REAL min1,min2,min3,min4,max1,max2,max3,max4;
};

__global__ void potSwap(parameters p,int i1, int j1, int i2, int j2,int intN,REAL *particles,REAL *boxR,REAL *potentials);
__global__ void particleSwap(int i,int j,int k,int l,int intN,REAL *particles);
__global__ void findE(int intN, REAL *Ematrix, REAL *particles, REAL *potentials, REAL *substrate);
__global__ void particleMove(int i,int j,int k,int l,int intN,REAL *particles);

int c_checkIn(int i, int j, int N);
__device__ int g_checkIn(int i, int j, int N); 
