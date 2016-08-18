//advanced cuda system
//sped up algorithms


/*
	Code guide: first matrices are initialized. they are used to keep track of the particles, the probabilities to jump, the substrate, and the general electric potential.
	Input parameters are also taken in. paramLoad tells you which parameters can be changed in the input file. The general electric potential is calculated in cuda. This reduces a n^4 problem to a n^2 one. A site is picked at random at the CPU (part of the monte-carlo process) and the probabilities with the particles around it are calculated at the gpu. The probabilities are then returned to the CPU where the second part of the Monte-Carlo algorithm occurs. Here, the site which the subject particle will interact with is chosen randomly but with weights according to the probabilities. The jump is made, and the system starts over.  

	The relaxation is based on a previous code. Basically, it consists of 2 steps. First, every site is tested against its 4 neighbors to see if any quick optimization can be done. Then the density of states is found. This gives a quick global view of which sites can be swapped to minimize system energy. 

*/

#include <stdio.h>
#include <stdlib.h> /* for rand() */
#include <unistd.h> /* for getpid() */
#include <time.h> /* for time() */
#include <math.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <cuda.h>
#include <sstream>
#include <string>
#include "closedMott.h"


#define PI	3.1415926535897932384626433832795
#define TWOPI 	6.28318530717958647692528676655901

// construct REAL "type," depending on desired precision
// set the maximum number of threads


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

int currentCount = 0;
int countThese = 1;
int tIndex = 0;

typedef struct {
	REAL re;
	REAL im;
} COMPLEX;

//absolute value on the gpu
__device__ int G_abs(int a) {
	
	if (a < 0) {
		a = -1*a;
	}


return a;
}

//wrote own modulo algorithms since computer modulo (%) does negative modulo's incorrectly (-3%10 = -3 instead of 7)
 __device__ int G_mod(int a,int b)
{
        while (a < 0) {
                a = a + b;
        }
        while (a >= b) {
                a = a - b;
        }
return a;
}
// I need a version of my modulo for the gpu and for the CPU 
int C_mod(int a, int b) {
        while (a < 0) {
                a = a + b;
        }
        while (a >= b) {
                a = a - b;
        }
return a;
}


//gpu matrix copying
__global__ void matrixCopy(int intN, REAL * matrixIn,REAL *matrixOut){
         int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
        if (idx < intN*intN) {
                matrixOut[idx] = matrixIn[idx];

        }

}




//Here, the gpu's find the general electric potential at each lattice site. 
__global__ void findPotential(REAL *particles,REAL *potentials,  REAL *boxR,parameters p) { 
        int i,j,checkx,checky;
	int intN = (int) p.N;
	int halfRange = p.N/2;//gets forced to (N-1)/2 since odd
	 

	int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
        double sum,distanceTerm;
	int k,l;
	 if(idx<intN*intN) {
        	i = idx/intN;
                j = idx%intN;
		sum = 0;
                       for(l = 0 ; l < p.N; l++) {
                                for(k = 0; k < p.N; k++) {

                                        checkx = i - halfRange + k;
                                        checky = j - halfRange + l;
					if(g_checkIn(checkx,checky,p.N)) {
	                                        if ((k != halfRange) || (l != halfRange)) { //dont do self-potential
  							distanceTerm = boxR[i + intN*j + intN*intN*k + intN*intN*intN*l];				
                	                               	sum = sum +  particles[(checkx) + intN*(checky)]/distanceTerm;
                        	                }
                                	}
				}
                        }
                potentials[i + intN*j] = sum*p.changeToV;

	}
}


//check for a CUDA error, use argument for identification
bool errorAsk(const char *s="n/a")
{
    cudaError_t err=cudaGetLastError();
    if(err==cudaSuccess)
        return false;
    printf("CUDA error [%s]: %s\n",s,cudaGetErrorString(err));
    return true;
};


 __device__ double findBlockade(int p,int thisp,double Ec)
{
	int deltaP = G_abs(thisp - p);
	int rho; //for rho*Ec constant multiplier regarding how much stacking is happening	

	if (deltaP==0) { //if they are the same, then something is trying to stack 
		return Ec;
	}

	if (deltaP == 2) { //if they are off by "1" then one guy is moving freely
		return 0;
	}

	rho = deltaP/2; // if it's not the first two, then the system is relaxing
	return -rho*Ec;
	
}

//The first half of the heart of this program. Here the probabilities are calculated based on the energy change of the system and on the localization of the electron.
__global__ void findProbabilities(REAL *TField,REAL *probabilities,REAL *particles,REAL *potentials,REAL *substrate,REAL *boxR,REAL *watcher,int tStep,int x, int y, parameters p)
{
//	REAL number = 11;
    int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    int i,j,thisi,thisj,thatp,thisp,hyperIndex,N;
	double potConstant,currentPart,distancePart,blockadePart,potentialPart,substratePart,energyPart,electronT;
//	double doublej, doublei,r;
//	potConstant = 1.17e-13;
//	potConstant = Ec;
	potConstant = -1;
	N = p.N;

	if(idx<N*N) {        
		i = idx/N;
                j = idx%N;
		i = i-N/2;
		j = j-N/2;

		thisi = i + x;
		thisj = j + y;
		if (g_checkIn(thisi,thisj,N)) {			

			hyperIndex = x + N*y + N*N*(idx/N) + N*N*N*(idx%N);
	
//		doublei = i;
//		doublej = j;
//		r = sqrt(doublei*doublei + doublej*doublej);	
//		distancePart = -2.000*boxR[idx];
			distancePart = -2*boxR[hyperIndex]/(p.xi);		
//		distancePart = 0;
			thatp = particles[x + N*y];
			thisp = particles[thisi + N*thisj];
	
			if(particles[x + N*y] > particles[thisi + N*thisj]) { //situation 1

				blockadePart = -1*findBlockade(thatp,thisp,p.Ec)/boxR[hyperIndex];
				potentialPart = -potConstant*(potentials[thisi + N*thisj] - potentials[x + N*y] - p.changeToV/boxR[hyperIndex]);
				substratePart = substrate[thisi+ N*thisj];
				currentPart = p.eV*i;
				electronT = TField[x + N*y];
//			currentPart = 0;			
//			blockadePart = 0;
//			potentialPart= 0;
//			substratePart= 0;

			
			}

			if (particles[x + N*y] < particles[thisi + N*thisj]) { //situation 2

				blockadePart = -1*findBlockade(thatp,thisp,p.Ec)/boxR[hyperIndex]; 
				potentialPart = potConstant*(potentials[thisi + N*thisj] - potentials[x + N*y] + p.changeToV/boxR[hyperIndex]);
				substratePart = -substrate[thisi + N*thisj];
				currentPart = -p.eV*i;
				electronT = TField[thisi + N*thisj];
	//			currentPart = 0;
	//			substratePart = 0;
	//			potentialPart = 0;
	//			blockadePart = 0;
	
			}
	
			if ( particles[x + N*y] == particles[thisi + N*thisj] ){// stacking
				electronT = (TField[x + N*y] + TField[thisi + N*thisj])/2;
	
				if (particles[x + N*y] < 0) { //then p1 is getting more negative and p2 is getting more positive (like in situation 1) 
			                blockadePart = -1*findBlockade(thatp,thisp,p.Ec)/boxR[hyperIndex];
		                        potentialPart = potConstant*(potentials[thisi + N*thisj] - potentials[x + N*y] + p.changeToV/boxR[hyperIndex]);
	        	                substratePart = -substrate[thisi + N*thisj];
	                	        currentPart = -p.eV*i;	
	
				}
	
				else  {  //then p1 is getting more positive and p2 is getting more negative (situation2-like transfer is happening)
					blockadePart = -1*findBlockade(thatp,thisp,p.Ec)/boxR[hyperIndex];
	                	        potentialPart = -potConstant*(potentials[thisi + N*thisj] - potentials[x + N*y] - p.changeToV/boxR[hyperIndex]);
	        	                substratePart = substrate[thisi+ N*thisj];
		                        currentPart = p.eV*i;
				}
	
	//			currentPart = 0;
	//			substratePart = 0;
	//			potentialPart = 0;
	//			blockadePart = 0;
	
			}
	
	
			energyPart = p.alphaTwo*(blockadePart+potentialPart+substratePart+currentPart)/electronT;
	
	                if (energyPart > 0) {
			              energyPart = 0;  
			}
	
			probabilities[idx] = exp(distancePart+energyPart);
			watcher[idx] = distancePart+p.alphaTwo*(blockadePart+potentialPart+substratePart+currentPart)/electronT;
	
	
	
			if ((thisi==x && thisj==y )  ){
	//		probabilities[idx] = 1; //force probability of jumping to self to 1 (avoids 0/0 problems)
	//		probabilities[idx] = 0; //rejection free monte carlo algorithm 
				probabilities[idx] = p.rejection;
			}
		}
		else {
			probabilities[idx] = 0;
		}
		
	}
};

__device__ void simpleFill(REAL *jumpRecord, REAL fillVal, int t) {
        jumpRecord[t] = fillVal;
}


__device__ void fillRecord(REAL *jumpRecord,REAL fillVal,int N) {
	int found = 0;
	int n = 0;
	while ((found == 0) && (n < N)) {
		if(jumpRecord[n] == 999) {
			found = 1;
			jumpRecord[n] = fillVal; 
			
		}
		n++;
	}
}



//calculates which direction the electron went and how far (not necessary if you are not measuring anything)
 __global__ void interaction(parameters p,int x,int y,int newx,int newy,REAL *particles,REAL *jumpRecord,REAL *boxR,int tStep) {
        int N = p.N,obsx,obsy;
        int whichWay = 0;
        REAL fillVal;

        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx < 1) {//only needs to be done once
		if (particles[x + y*N] == particles[newx + newy*N]) {
			if (particles[x + y*N] > 0) {
				whichWay = 1;
			}
			else {
				whichWay = -1;
			}
		}

                else if (particles[x + y*N] > particles[newx + newy*N] ) {
                        whichWay = 1;
                }

                else if (particles[x + y*N] < particles[newx + newy*N]) {
                        whichWay = -1;
                }

                obsx = newx + ( p.N/2 - x);
                obsy = newy + ( p.N/2 - y);
	
		if (g_checkIn(obsx,obsy,p.N)) {
	                if(p.grabJ == 1) {
        	                      fillVal = -whichWay*(obsx-p.N/2);
                	}

	                if(p.grabJ == 0) {
        	                fillVal = boxR[x + N*y + N*N*obsx + N*N*N*obsy]/p.L;
                	}
		}
		else {
			fillVal = 1e9; //ridiculous value will highlight errors
		}


//	        fillRecord(jumpRecord,fillVal,p.recordLength);
		simpleFill(jumpRecord,fillVal,tStep);

	}
}

//this section does the various outputs such as particle positions or general electric potential
//this one outputs how far electrons jumped
void showJump(int N,int x,int y,int newx,int newy,REAL* hereP) {
	double r,deltax,deltay;
	deltax = (x-newx);
	deltay = (y-newy);

	r = sqrt(deltax*deltax + deltay*deltay);
	
	
	cout<<r<<endl;


}

//this is for showing the electron positions
void showMove(REAL* hereP,int N) {
        int i,j;
        for ( j = 0; j < N;j++) {
                for( i = 0; i < N; i++) {
                        cout<<hereP[i + N*j]<<" ";
                }
        cout<<endl;
        }


}

//sums the potentials (during relaxation this should generally decrease)
double sumEnergy(REAL* hereField,int N) {
	int i,j;
	double sum;
	sum = 0;
	for ( j = 0; j < N;j++) {
                for( i = 0; i < N; i++) {
                        sum = sum +  hereField[i + N*j];
                }
        }
	
	return sum;
}

//to double check i had no particles leaking
void countParticles(REAL* hereP, int N) {
        int i,j;
        double sum;
        sum = 0;
        for ( j = 0; j < N;j++) {
                for( i = 0; i < N; i++) {
                        sum = sum +  hereP[i + N*j];
                }
        }

        cout<<sum<<endl;



}

//finalizes the monte carlo algorithm by picking a site at random (weighted)
__global__ void weightedWheel(parameters p, double randomNum,REAL *reducedProb, int *picked) {
	int N = p.N;	
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
        double pickedValue = randomNum*reducedProb[N*N -1];

	if ((idx > 0) && (idx < N*N)) {
		if ((reducedProb[idx - 1] < pickedValue) && (reducedProb[idx] > pickedValue)) {
			picked[0] = idx;
		}
	}
	if (idx == 0) {
		if (pickedValue < reducedProb[0]) {
			picked[0] =idx;
		}	
	}

}

void printLineCPU(REAL * c_line, char *fileName) {
	int k;
	FILE *fp1;
	fp1 = fopen(fileName, "w");
	for (k = 0; k < tIndex; k++) {
		
		fprintf(fp1, "%lf ", c_line[k]);
	}
	fclose(fp1);

}

//print the CPU matrix to a file
void printBoxCPU(REAL *c_array,int size, char * fileName) {
        int k,l;
        FILE    *fp1;
//        char    str1[256];
//        sprintf(str1, "box.txt");
        fp1 = fopen(fileName, "w");
        for (k = 0; k < size ; k++){
                for(l = 0; l < size; l++) {

                        fprintf(fp1, "%lf ",1e9*c_array[k + l*size]);
                }
        fprintf(fp1,"\n");
        }
//cleanup
        fclose(fp1);
}

void printMagnifyGPU(REAL *g_array,int size,char * fileName) {
        REAL *c_array;
        c_array =  new REAL[size*size];
        int k,l;
        cudaMemcpy(c_array,g_array,size*size*sizeof(REAL),cudaMemcpyDeviceToHost);
        FILE    *fp1;
        fp1 = fopen(fileName, "w");

        for (k = 0; k < size ; k++){
                for(l = 0; l < size; l++) {
                        fprintf(fp1, "%lf ",c_array[k + l*size]*1e100);
                }
        fprintf(fp1,"\n");
        }
//cleanup
        fclose(fp1);
        delete[] c_array;



}

/*
void printBoxGPU(REAL *g_array,int size,char * fileName) {
        REAL *c_array;
        c_array =  new REAL[size*size];
        int k,l;
        cudaMemcpy(c_array,g_array,size*size*sizeof(REAL),cudaMemcpyDeviceToHost);
	ofstream myfile;
	myfile.open (fileName);
        for (k = 0; k < size ; k++){
                for(l = 0; l < size; l++) {
			myfile<<c_array[k + l*size]<<" ";
		}
	myfile<<endl;
	}

	myfile.close();
	delete[] c_array;
}
*/


//print the gpu matrix to a file
void printBoxGPU(REAL *g_array,int size,char * fileName) {
        REAL *c_array;
        c_array =  new REAL[size*size];
        int k,l;
        cudaMemcpy(c_array,g_array,size*size*sizeof(REAL),cudaMemcpyDeviceToHost);
        FILE    *fp1;
        fp1 = fopen(fileName, "w");

        for (k = 0; k < size ; k++){
                for(l = 0; l < size; l++) {
/*
			if(c_array[k + l*size] == 0) {
				c_array[k + l*size] = 999;
			}
*/
                        fprintf(fp1, "%lf ",c_array[l + k*size]); //transposed l & k since thats how octave reads it
                }
        fprintf(fp1,"\n");
        }
//cleanup
        fclose(fp1);
	delete[] c_array;
}


void printIntGPU(int *g_array,int size,char * name) {//can probably overload using C++11
        int *c_array;
        c_array =  new int[size];
        int k;
        cudaMemcpy(c_array,g_array,size*sizeof(int),cudaMemcpyDeviceToHost);
        FILE    *fp1;
        fp1 = fopen(name, "w");
        for (k = 0; k < size ; k++){

                        fprintf(fp1, "%i ",c_array[k]);
                        fprintf(fp1,"\n");
        }
//cleanup
        fclose(fp1);
        delete[] c_array;
}


//print gpu array to a file
void printLineGPU(REAL *g_array,int size,char * name) {
        REAL *c_array;
        c_array =  new REAL[size];
        int k;
        cudaMemcpy(c_array,g_array,size*sizeof(REAL),cudaMemcpyDeviceToHost);
        FILE    *fp1;
        fp1 = fopen(name, "w");
        for (k = 0; k < size ; k++){

                        fprintf(fp1, "%lf ",c_array[k]);
			fprintf(fp1,"\n");
        }
//cleanup
        fclose(fp1);
	delete[] c_array;
}

/*
//print a single number to a file
void printSingle(double nimeRun,char *fileName){
	FILE    *fp1;
	fp1 = fopen(fileName, "w");
	fprintf(fp1, "%lf ",timeRun);
	fclose(fp1);
}
*/

//loading previous results
REAL *loadMatrix(REAL *hereMatrix,char* fileName) {
//      infile.open (fileName, ifstream::in);
//      REAL * buffer;
//        ifstream read(fileName);
        ifstream infile(fileName);
        string line;
        int counter = 0;
        double d;
        while (getline(infile, line)) {
                istringstream iss(line);
                if (iss >> d) {
                        hereMatrix[counter] = d;

                        counter++;
                }


        }


        return hereMatrix;
}

//tracking really the sum of the probabilities
void trackTime(REAL *timeRun, REAL sum,int recordLength) {
	if (tIndex < recordLength) { //prevent bad bad memory writing
		timeRun[tIndex] = sum;
		tIndex++;
	}
}



//second part of the heart of this code. Here the probabilities are summed and a number is picked from 0 to that number. The code then sums through the probabilities untill it reaches that number. In this way, probabilities which are higher will have a larger chance of getting picked. 
void particleScout(vectors &v,int x,int y, double randomNum,int blocks, int threads,parameters p) {
	int lastx,lasty,newx,newy;
	thrust::device_ptr<REAL> g_go = thrust::device_pointer_cast(v.probabilities);
        thrust::device_ptr<REAL> g_return = thrust::device_pointer_cast(v.reducedProb);

//	double sum;
	thrust::inclusive_scan(g_go, g_go + p.N*p.N, g_return); // in-place scan 
//	sum = thrust::reduce(g_go, g_go + p.N*p.N);	

//	trackTime(v.timeRun, sum,p.recordLength); 
	
	weightedWheel<<<blocks,threads>>>(p, randomNum,v.reducedProb, v.picked);
	cudaMemcpy(v.herePicked,v.picked,sizeof(int),cudaMemcpyDeviceToHost);	
/*
cudaMemcpy(v.hereProb,v.probabilities,p.N*p.N*sizeof(REAL),cudaMemcpyDeviceToHost);
cout<<"cell "<<v.herePicked[0]<<" was picked with a weight "<<v.hereProb[v.herePicked[0]]<<" out of a total "<<sum<<endl;
//	printMagnifyGPU(v.reducedProb,p.N,"reduced.dat");
*/
//	printBoxGPU(v.reducedProb,p.N,"reduced.dat");

//printMagnifyGPU(v.probabilities,p.N,"magnified.dat");

        lastx = v.herePicked[0]/p.N;
        lasty = v.herePicked[0]%p.N;
        newx = x - p.N/2 +  lastx;
        newy = y - p.N/2 +  lasty;
	if (c_checkIn(newx,newy,p.N)) {
//cout<<x<<" "<<y<<" "<<newx<<" "<<newy<<endl;
	
		interaction<<<blocks,threads>>>(p,x,y,newx,newy,v.particles,v.jumpRecord,v.boxR,v.tStep);

        	potSwap<<<blocks,threads>>>(p, x, y,newx,newy,p.N,v.particles,v.boxR,v.potentials);
	        particleMove<<<blocks,threads>>>(x, y, newx,newy,p.N,v.particles);
	        findE<<<blocks,threads>>>(p.N, v.Ematrix,v.particles,v.potentials,v.substrate);
	}
		errorAsk("particleJump");
}

__global__ void Eflip(int intN, double boltzmann, REAL *tempDos,REAL *Ematrix,REAL *TField) {

         int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
         if(idx<intN*intN) {
                tempDos[idx] = exp(-Ematrix[idx]/(boltzmann*TField[idx]));
        }

}



void findFirst(parameters p,int blocks,int threads,vectors &v) {
	double randomNum;
	Eflip<<<blocks,threads>>>(p.N, p.boltzmann, v.tempDos,v.Ematrix,v.TField);
	errorAsk("Eflip"); //check for error
	

	thrust::device_ptr<REAL> g_go = thrust::device_pointer_cast(v.tempDos);//tempDos memory being recycled
        thrust::device_ptr<REAL> g_return = thrust::device_pointer_cast(v.reducedProb);// also reduced prob memory
        thrust::inclusive_scan(g_go, g_go + p.N*p.N, g_return); // in-place scan 

        randomNum = drand48();//place where the wheel lands
        weightedWheel<<<blocks,threads>>>(p, randomNum,v.reducedProb, v.picked);
        cudaMemcpy(v.herePicked,v.picked,sizeof(int),cudaMemcpyDeviceToHost);
}

void findTime(parameters p,int blocks,int threads,vectors &v) {
	int x,y;
	double totalSum,result;
	totalSum = 0;
	thrust::device_ptr<REAL> g_go = thrust::device_pointer_cast(v.probabilities);

	for (y = 0; y < p.N; y++) {
		for(x = 0;x < p.N; x++) {
			findProbabilities<<<blocks,threads>>>(v.TField,v.probabilities,v.particles,v.potentials,v.substrate,v.boxR,v.watcher,v.tStep,x,y,p);
			result = thrust::reduce(g_go, g_go + p.N*p.N);		
			totalSum += result;
/*
cout<<result<<endl;
if ((result < -0) && (result > -10000000)) {
//if ((result ==0)) {
printBoxGPU(v.probabilities,p.N,p.boxName);
cout<<"gets to here"<<endl;
}
*/

		}
	}
//	if (result == 0) {
//		printBoxGPU(v.probabilities,p.N,p.boxName);
//		cout<<"gets to here"<<endl;
//	}

//	cout<<totalSum<<endl;	
	trackTime(v.sumRun, totalSum,p.recordLength);

}

//the particles are picked here. This is also where the system is run from. (find potential, find probabilities, and move particle are done here)
void findJump(vectors &v,int threads,int blocks,parameters p) {
	int x,y;	
	double randomNum;

//	printBoxGPU(v.potentials,p.N,"pot0.dat");

	findTime(p,blocks,threads,v);	

//	printBoxGPU(v.potentials,p.N,"pot1.dat");
		
	findFirst( p, blocks,threads,v);//find the first particle according to exp(-beta)        

//	printBoxGPU(v.potentials,p.N,"pot2.dat");


	x = v.herePicked[0]%p.N;
        y = v.herePicked[0]/p.N;
	findProbabilities<<<blocks,threads>>>(v.TField,v.probabilities,v.particles,v.potentials,v.substrate,v.boxR,v.watcher,v.tStep,x,y,p);
	errorAsk("find probabilities"); //check for error
//	printBoxGPU(v.potentials,p.N,"pot3.dat");
	
	randomNum = drand48();
	particleScout(v, x, y, randomNum, blocks, threads,p);
//	printBoxGPU(v.potentials,p.N,"pot4.dat");

}

//calculate energy contribution from stacked particles
__global__ void G_stackE(REAL *particles,REAL *stacked,int intN) {
        int i,j;
        double blockade = 1.97e-5;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
        i = idx/intN;
        j = idx%intN;

	if(idx < intN*intN) {
		if (particles[i + j*intN] > 1) {
			stacked[idx] = blockade;

		}
	}
}


//calculate energy contribution from the substrate
__global__ void G_subE(REAL *substrate,REAL *particles,REAL *combined,int intN) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if(idx < intN*intN) {
	        combined[idx] = substrate[idx]*particles[idx];
	}


}

//filling a gpu array using CPU numbers
__global__ void fillSum(int index,int intN,int addSub,REAL *sumArray,REAL numToInsert) {
//	  int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
//	if(idx < 1) {
	REAL	dSign = (REAL) addSub;
	sumArray[index] = dSign*numToInsert;
//	}
}

//change particle to hole (or back)
__global__ void particleSwitch(int i,int j,int intN,REAL *particles) {

	if (particles[i + j*intN] == -1) {
		particles[i + j*intN]= 1;
	}
	else {
		particles[i + j*intN]= -1;
	}
}

//fill dos (gpu) matrix with sums (CPU)
__global__ void dosPut(int i,int j,int intN,REAL *Ematrix,REAL sum) {
	Ematrix[i + j*intN] =	sum;
}


//find the density of states at each site
 void G_dos(REAL * sumArray,REAL *extraArray,REAL *boxR,REAL *particles,REAL *substrate,REAL *Ematrix,REAL *potentials,int slices,int threads,int blocks,parameters ,parameters p) {
	int i,j,intN;//not sure about Sums
	intN = p.N;
        thrust::device_ptr<REAL> g_go = thrust::device_pointer_cast(potentials);
	thrust::device_ptr<REAL> sumArrayPtr = thrust::device_pointer_cast(sumArray);
	thrust::device_ptr<REAL> extraArrayPtr = thrust::device_pointer_cast(extraArray);
	REAL result;

        for (j = 0; j < intN; j++) {
                for (i = 0; i < intN; i++) {
			findPotential<<<blocks,threads>>>(particles,potentials, boxR,p);

			result = thrust::reduce(g_go, g_go + intN*intN);        
			fillSum<<<blocks,threads>>>(0,intN,-1,sumArray,result);
//                        fillSum<<<blocks,threads>>>(0,intN,1,sumArray,result);

			G_subE<<<blocks,threads>>>(substrate,particles,extraArray,intN);
			result = thrust::reduce(extraArrayPtr,  extraArrayPtr + intN*intN);
			fillSum<<<blocks,threads>>>(1,intN,-1,sumArray,result);
//                        fillSum<<<blocks,threads>>>(1,intN,1,sumArray,result);

			G_stackE<<<blocks,threads>>>(particles,extraArray,intN);
			result = thrust::reduce(extraArrayPtr,  extraArrayPtr + intN*intN);
                        fillSum<<<blocks,threads>>>(2,intN,-1,sumArray,result);
//			fillSum<<<blocks,threads>>>(2,intN,1,sumArray,result);
			
			particleSwitch<<<blocks,threads>>>(i,j,intN,particles);

			findPotential<<<blocks,threads>>>(particles,potentials, boxR,p);
			result = thrust::reduce(g_go, g_go + intN*intN);
                        fillSum<<<blocks,threads>>>(3,intN,1,sumArray,result);
//                        fillSum<<<blocks,threads>>>(3,intN,-1,sumArray,result);

                        G_subE<<<blocks,threads>>>(substrate,particles,extraArray,intN);
                        result = thrust::reduce(extraArrayPtr,  extraArrayPtr + intN*intN);
			fillSum<<<blocks,threads>>>(4,intN,1,sumArray,result);
//                        fillSum<<<blocks,threads>>>(4,intN,-1,sumArray,result);

                        G_stackE<<<blocks,threads>>>(particles,extraArray,intN);
                	result = thrust::reduce(extraArrayPtr,  extraArrayPtr + intN*intN);
			fillSum<<<blocks,threads>>>(5,intN,1,sumArray,result);
//                        fillSum<<<blocks,threads>>>(5,intN,-1,sumArray,result);

                        particleSwitch<<<blocks,threads>>>(i,j,intN,particles);
			
			result = thrust::reduce(sumArrayPtr, sumArrayPtr + 6); 
//			result = 0;	
			dosPut<<<blocks,threads>>>(i,j,intN,Ematrix,result);
		}
	}


}


//random substrate is created here
REAL *createSub(REAL *hereS,double muVar,int N) {
        int i,j;

        for(j = 0; j < N; j++ ) {
                for(i = 0; i < N; i++) {

                        hereS[i + N*j] = drand48()*muVar*2 - muVar;
        //              if(i > nx/2) hereS[i + ny*j] = 50000000;        
                }

        }





        return hereS;
}



// creates the variation in x & y matrices
REAL * createDiff(REAL * hereDiff, double var, int N) {
	int i,j;

	for(j = 0; j < N; j++) {
		for(i = 0; i < N; i++) {
			hereDiff[i + N*j] = drand48()* var*2 - var;
		}
	}

	return hereDiff;
}

//fill a matrix with 0s
REAL *C_zeros(double N, REAL *A) {
	int idx;
	for (idx = 0; idx < N; idx++) {
		A[idx] = 0;
	}
return A;


}
//creates and fills matrices
REAL *C_random(double N,double nparticles,REAL *A) {
        int idx,idy,count,index;
        int randx,randy;
        count = 0;


        for (idx = 0; idx < N; idx++) {
                for( idy = 0; idy < N; idy++) {
                        index = int(idy + idx*N);
                        A[index] = -1;


                }
        }

	        while(count < nparticles) {
                randx = drand48()*N;
                randy = drand48()*N;
                randx = floor(randx);
                randy = floor(randy);

                index = int(randx*N + randy);
                 if (A[index] < 2) {
                        A[index] = A[index] + 1;
                        count++;
                }
        }



return A;

}



//creates and fills matrices when filled percent > 100%
REAL *C_more(double N,double nparticles,REAL *A) {
        int idx,idy,count,index;
        int randx,randy;
        count = 0;


        for (idx = 0; idx < N; idx++) {
                for( idy = 0; idy < N; idy++) {
                        index = int(idy + idx*N);
                        A[index] = 1;


                }
        }

	


                while(count < (nparticles-N*N)) {
                randx = drand48()*N;
                randy = drand48()*N;
                randx = floor(randx);
                randy = floor(randy);

                index = int(randx*N + randy);
                 if (A[index] < 2) {
                        A[index] = A[index] + 1;
                        count++;
                }
        }



return A;

}


//creates the "distance hyper-matrix" 1/r
REAL *createR(REAL *A,REAL *diffX, REAL *diffY,double N,double L,double xi) {
	double r,doublel,doublek,deltaX,deltaY;
	double diffXThere,diffYThere,diffXHere,diffYHere;
	int i,j,k,l,intN,idx,kObs,lObs,kNew,lNew;
	
	intN = N;
	for (idx = 0; idx < N*N*N*N; idx++) {

                i = idx%(intN);
                j = (idx%(intN*intN) - idx%(intN))/intN; 
                k = (idx%(intN*intN*intN) - idx%(intN*intN))/(intN*intN) ; 
                l = (idx%(intN*intN*intN*intN) - idx%(intN*intN*intN))/(intN*intN*intN) ;

/*
                k = idx%(intN);
                l = (idx%(intN*intN) - idx%(intN))/intN;
                i = (idx%(intN*intN*intN) - idx%(intN*intN))/(intN*intN) ;
                j = (idx%(intN*intN*intN*intN) - idx%(intN*intN*intN))/(intN*intN*intN) ;
*/

                doublek = (double) k;
                doublel = (double) l;

		kNew = i + k - N/2;
		lNew = j + l - N/2;

                kObs = C_mod(kNew,N);
                lObs = C_mod(lNew,N);
		
                diffXHere = diffX[i + intN*j];
                diffXThere = diffX[kObs + intN*lObs];

                if((kNew < 0) || (kNew > N)) {
      //                  diffXHere = -diffX[i + intN*j];
                        diffXThere = -diffX[kObs + intN*lObs];
                }
                diffYHere = diffY[i + intN*j];
                diffYThere = diffY[kObs + intN*lObs];

                if((lNew < 0) || (lNew > N)) {
        //                diffYHere = -diffY[i + intN*j];
                        diffYThere = -diffY[kObs + intN*lObs];
                }

		deltaX = diffXHere - (diffXThere + L*(doublek - N/2));
		deltaY = diffYHere - (diffYThere + L*(doublel - N/2));
		r = sqrt(deltaX*deltaX + deltaY*deltaY);

		A[idx] = r;


	}
/*
	for (i = 0; i < N; i++) {
		for(j = 0; j < N ; j++) {
			 cout<<A[1 + intN*1 + intN*intN*i + intN*intN*intN*j]/L<<" ";
		}
		cout<<endl;
	}
*/

//cout<<A[1 + intN*2 + intN*intN*1 + intN*intN*intN*1]/L<<endl;

return A;
	
}

//create hexagonal lattice position tensor
REAL *createHex(REAL *A,REAL *diffX, REAL *diffY,double N,double L,double xi) {
        double r,doublel,doublek,deltaX,deltaY;
        double diffXThere,diffYThere,diffXHere,diffYHere;
        int i,j,k,l,intN,idx,kObs,lObs,kNew,lNew;

        intN = N;
        for (idx = 0; idx < N*N*N*N; idx++) {

                i = idx%(intN);
                j = (idx%(intN*intN) - idx%(intN))/intN;
                k = (idx%(intN*intN*intN) - idx%(intN*intN))/(intN*intN) ;
                l = (idx%(intN*intN*intN*intN) - idx%(intN*intN*intN))/(intN*intN*intN) ;

                doublek = (double) k;
                doublel = (double) l;

                kNew = i + k - N/2;
                lNew = j + l - N/2;

                kObs = C_mod(kNew,N);
                lObs = C_mod(lNew,N);

                diffXHere = diffX[i + intN*j];
                diffXThere = diffX[kObs + intN*lObs];

                if((kNew < 0) || (kNew > N)) {
                        diffXThere = -diffX[kObs + intN*lObs];
                }
                diffYHere = diffY[i + intN*j];
                diffYThere = diffY[kObs + intN*lObs];

                if((lNew < 0) || (lNew > N)) {
                        diffYThere = -diffY[kObs + intN*lObs];
                }

                if ( (l%2)==1 ){
                        if (doublek < N/2) {
                                deltaX = diffXHere - (diffXThere + L*(doublek - N/2) - L/2);
                        }
                        else {
                                deltaX = diffXHere - (diffXThere + L*(doublek - N/2) + L/2);
                        }
                }
                else {
                        deltaX = diffXHere - (diffXThere + L*(doublek - N/2));
                }

                deltaY = diffYHere - (diffYThere + .866*L*(doublel - N/2));

                r = sqrt(deltaX*deltaX + deltaY*deltaY);

                A[idx] = r;
	}
/*
        for (i = 0; i < N; i++) {
                for(j = 0; j < N ; j++) {
                         cout<<A[1 + intN*1 + intN*intN*i + intN*intN*intN*j]<<" ";
                }
                cout<<endl;
        }
*/

return A;

}


//clumps all of the original electrons ( to show relaxation)
REAL *C_clump(double N,double nparticles,REAL *A) {
        int idx;

	for (idx = 0;idx < N*N; idx++) {
		A[idx] = -1;
	}

	for (idx = 0; idx < nparticles; idx++) {
		A[idx] = 1;
	}
	
	
return A;
}

//electrons evenly spaced out (to try to calculate average jump distances with a general electric potential)
REAL *C_spread(double N,double nparticles,REAL *A) {
        int idx,i,j,intN;
	intN = (int) N;
	
	
        for (idx = 0;idx < N*N; idx++) {
                A[idx] = -1;
        }

        for (idx = 0; idx < N*N; idx++) {
        	i = idx/intN;
        	j = idx%intN;
		
		if((i + j)%2) {        
			A[idx] = 1;
        	}
	}


return A;
}

//take whatever is at a and swap it with whatever is at b
__global__ void particleSwap(int i,int j,int k,int l,int intN,REAL *particles) {
	int temp;
//	  int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;

//	if (idx < 1) {
	temp = particles[i + j*intN];
        particles[i + j*intN]= particles[k + l*intN];
	particles[k + l*intN] = temp;
//	}
}
//take whatever is at a and swap it with whatever is at b
__device__ void g_particleSwap(int i,int j,int k,int l,int intN,REAL *particles){
	
        int temp;
        int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (idx < 1) {
	temp = particles[i + j*intN];
        particles[i + j*intN]= particles[k + l*intN];
        particles[k + l*intN] = temp;	
	}
}

__global__ void particleMove(int i,int j,int k,int l,int N,REAL *particles) {
        int deltaP = G_abs(particles[k + N*l] - particles[i + N*j]);
        int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;

        if (idx < 1) {
	        if (deltaP == 0) {
 		        if (particles[i + j*N] > 0) { //particle is moving from p1 to p2
        			particles[i + j*N] = particles[i + j*N] - 2;
				particles[k + l*N] = particles[k + l*N] + 2;	
			}
			else { // hole is moving from p1 to p2 (particle is moving from p2 to p1)
				particles[i + j*N] = particles[i + j*N] + 2;
                        	particles[k + l*N] = particles[k + l*N] - 2;
	                }
		}

		if (deltaP == 2) {
			if (particles[i + j*N] > particles[k + l*N]) {
				particles[i + j*N] = particles[i + j*N] - 2;
	                        particles[k + l*N] = particles[k + l*N] + 2;
			}
			else {
				particles[i + j*N] = particles[i + j*N] + 2;
	                        particles[k + l*N] = particles[k + l*N] - 2;
			}	
		}
	
		if (deltaP > 2) { 
	                if (particles[i + j*N] > particles[k + l*N]) {
	                        particles[i + j*N] = particles[i + j*N] - 2;
	                        particles[k + l*N] = particles[k + l*N] + 2;
	                }
	                else {
	                        particles[i + j*N] = particles[i + j*N] + 2;
	                        particles[k + l*N] = particles[k + l*N] - 2;
	                }
	
		}
	
	}	
	
}

//change coordinates from observer to particle
__device__ int changeCoordinates(int intN, int x1, int x2) {

	int modulox,newCoord;
        if (x2 < intN/2) {
		modulox = x2;
	}
	else {
		modulox = x2 - intN;
	}

	newCoord = intN/2 + modulox ;

return newCoord;

}



//perform a swap of two particles and recalculate all of the values
__global__ void slowSwap(parameters p,int i1,int j1,int i2, int j2,int intN, REAL* tempPar,REAL *tempPot, REAL* tempDos, REAL* particles,REAL *boxR,REAL* substrate, REAL *Ematrix, REAL *watcher,REAL *potentials) {
	double distance1, distance2;
	int xPre,yPre,x,y;

         int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
         if(idx<intN*intN) {
		tempPar[idx] = particles[idx];
                tempPot[idx] = potentials[idx];
                tempDos[idx] = Ematrix[idx];
		if (particles[i1 + intN*j1] != particles[i2 + intN*j2]) {
			if(particles[i1 + intN*j1] == 1) {
				tempPar[i1 + intN*j1] = -1;
        	                
				yPre = idx/intN;
	                        xPre = idx%intN;				

				x =  xPre + (intN/2 - i1);//for closed boundary dont do G_mod
                	        y =  yPre + (intN/2 - j1); //instead only do if  yPre + (intN/2 - j1) > 0 or < intN
                		if(g_checkIn(x,y,intN)) { //only apply potential changes inside the system
			                distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
	
		                        if (distance1 > 0) {
		                                tempPot[idx] = tempPot[idx] + p.changeToV/distance1;
						tempDos[idx] = tempPot[idx]*tempPar[idx] + substrate[idx]*tempPar[idx];
	        	                }
					else {
	//					tempPot[idx] = tempPot[idx] - substrate[idx]*particles[idx];	
						tempDos[idx] = tempPot[idx]*tempPar[idx] + substrate[idx]*tempPar[idx];

					}
//	probe = distance1;
					tempPar[i2 + intN*j2] = 1;
					
					distance2 = boxR[i2 + intN*j2 + intN*intN*x + intN*intN*intN*y];
					if (distance2 > 0) {
                        	                tempPot[idx] = tempPot[idx] - p.changeToV/distance2;
						tempDos[idx] = tempPot[idx]*tempPar[idx]+ substrate[idx]*tempPar[idx];
        	                        }
					else {
//              	                          tempPot[idx] = tempPot[idx] + substrate[idx]*particles[idx];
					
						tempDos[idx] = tempPot[idx]*tempPar[idx] - substrate[idx]*tempPar[idx];
	
	                                }		

				}		
			}
			else {
                                tempPar[i1 + intN*j1] = 1;

//                                xPre = idx/intN;
//                                yPre = idx%intN;

				yPre = idx/intN;
                                xPre = idx%intN;

                                x =  xPre + (intN/2 - i1);//for closed boundary dont do G_mod
                                y =  yPre + (intN/2 - j1); //instead only do if  yPre + (intN/2 - j1) > 0 or < intN
                                if(g_checkIn(x,y,intN)) { //only apply potential changes inside the system
	                                distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
//				distance1 = boxR[x + intN*y + intN*intN*i1 + intN*intN*intN*j1];
//				watcher[idx] = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
	                                if (distance1 > 0) {
        	                                tempPot[idx] = tempPot[idx] - p.changeToV/distance1;
						tempDos[idx] = tempPot[idx]*tempPar[idx] + substrate[idx]*tempPar[idx];
                        	        }
					else {
//                                        tempPot[idx] = tempPot[idx] + substrate[idx]*particles[idx];
						tempDos[idx] = tempPot[idx]*tempPar[idx] + substrate[idx]*tempPar[idx];
//					watcher[idx] = tempPot[idx]*tempPar[idx] ;
        	                        }

                	                tempPar[i2 + intN*j2] = -1;
					x = (int) G_mod(xPre + ( intN/2 - i2),intN);
	                                y = (int) G_mod(yPre + (intN/2 - j2),intN);
	
        	                        distance2 = boxR[i2 + intN*j2 + intN*intN*x + intN*intN*intN*y];
                	                if (distance2 > 0) {
                        	                tempPot[idx] = tempPot[idx] + p.changeToV/distance2;
						tempDos[idx] = tempPot[idx]*tempPar[idx] + substrate[idx]*tempPar[idx];
        	                        }
	                                else {
//                                        tempPot[idx] = tempPot[idx] - substrate[idx]*particles[idx];
						tempDos[idx] = tempPot[idx]*tempPar[idx] + substrate[idx]*tempPar[idx];
//					watcher[idx] = tempPot[idx]*tempPar[idx] ;
					}
				}
			watcher[idx] = substrate[idx]*tempPar[idx];
			}
//		tempDos[idx] = probe;
		}

	        else {
                        tempDos[idx] = Ematrix[idx];
                }
	}
}	

//calculate substrate contribution to syatem energy
__global__ void subAdd(int intN, REAL *particles,REAL *potentials,REAL *substrate){

         int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
         if(idx<intN*intN) {
		potentials[idx] = potentials[idx] + substrate[idx]*particles[idx];
	}

}

//calculate electrical potential contribution to system energy
__global__ void potAdd(parameters p,int i1, int j1, int intN,REAL *particles, REAL *potentials, REAL *boxR){
        int x,y;
        int xPre,yPre;
        REAL distance1;

         int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
         if(idx<intN*intN) {

//for the -0.5,0.5 system
                if (particles[i1 + intN*j1] == 1) {

		        yPre = idx/intN;//if it works
                        xPre = idx%intN;//it works
			x = xPre + (intN/2 - i1);
                        y = yPre + (intN/2 - j1);
                        if(g_checkIn(x,y,intN) ) {
	                        distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
        	                if (distance1 > 0) {
                	       		potentials[idx] = potentials[idx] -.5*p.changeToV/distance1; // .5 since Im coming from neutral
                        	}
			}
			
		}
		
		if (particles[i1 + intN*j1] == -1) {

                        yPre = idx/intN;//if it works
                        xPre = idx%intN;//it works
			x = xPre + (intN/2 - i1);
                        y = yPre + (intN/2 - j1);
                        if(g_checkIn(x,y,intN) ) {
                        	distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
	                        if (distance1 > 0) {
					potentials[idx] = potentials[idx] + .5*p.changeToV/distance1;
	                        }
			}

                }

	
/*
//for the -1,0,1 system , maybe it doesnt matter
                        yPre = idx/intN;//if it works
                        xPre = idx%intN;//it works
                        x = xPre + (intN/2 - i1);
                        y = yPre + (intN/2 - j1);
			if(g_checkIn(x,y,intN) ) {
	                        distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
        	                if (distance1 > 0) {
                	                potentials[idx] = potentials[idx] - particles[i1 + intN*j1]*.5*p.changeToV/distance1;
                        	}
			}
*/	

	}
}

//calculate change in electric potentials when a particle is removed
__global__ void potSub(parameters p,int i1, int j1, int intN,REAL *particles,REAL *boxR,REAL *potentials){
        int x,y;
        int xPre,yPre;
        REAL distance1;

         int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
         if(idx<intN*intN) {
                if (particles[i1 + intN*j1] == -1) {

                        yPre = idx/intN;//if it works
                        xPre = idx%intN;//it works
                        x = xPre + (intN/2 - i1);
                        y = yPre + (intN/2 - j1);
                        if(g_checkIn(x,y,intN) ) {
	                        distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
        	                if (distance1 > 0) {
                	                potentials[idx] = potentials[idx] + p.changeToV/distance1;
                        	}
			}

                }
                if (particles[i1 + intN*j1] == 1) {


 	                yPre = idx/intN;//if it works
                        xPre = idx%intN;//it works
                        x = xPre + (intN/2 - i1);
                        y = yPre + (intN/2 - j1);
                        if(g_checkIn(x,y,intN) ) {
                        	distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
                	        if (distance1 > 0) {
        	                        potentials[idx] = potentials[idx] - p.changeToV/distance1;
	                        }
			}

                }
	



        }
}

//calculate change in potential energy
__global__ void potChange(parameters p,int i1, int j1, int intN,REAL *particles,REAL *boxR,REAL *potentials,REAL* Ematrix){
        int x,y;
        int xPre,yPre;
        REAL distance1;

         int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
         if(idx<intN*intN) {


                if (particles[i1 + intN*j1] == 1) {
			particles[i1 + intN*j1] = 0;
                        yPre = idx/intN;//if it works
                        xPre = idx%intN;//it works
                        x =  xPre + (intN/2 - i1);//for closed boundary dont do G_mod
                        y =  yPre + (intN/2 - j1); //instead only do if  yPre + (intN/2 - j1) > 0 or < intN
                        if(g_checkIn(x,y,intN)) { //only apply potential changes inside the system

	                        distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
        	                if (distance1 > 0) {
                	                potentials[idx] = potentials[idx] + p.changeToV/distance1;
//					potentials[idx] = 999;
	                        }
			}

	                if(particles[idx] == 0) {//empty cells count as positive potential
        	                Ematrix[idx] = potentials[idx];
//                	if (distance1 == 0) {
//				Ematrix[idx] = -potentials[idx];
//			}
			}
        	        else {
                	        Ematrix[idx] = -potentials[idx];
//			if (distance1 == 0) {
//				Ematrix[idx] = potentials[idx];
//                        }
	                }


                }
				
		else {
			particles[i1 + intN*j1] = 1;
                        yPre = idx/intN;//if it works
                        xPre = idx%intN;//it works
                        x =  xPre + (intN/2 - i1);//for closed boundary dont do G_mod
                        y =  yPre + (intN/2 - j1); //instead only do if  yPre + (intN/2 - j1) > 0 or < intN
                        if(g_checkIn(x,y,intN)) { //only apply potential changes inside the system

	                        distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
        	                if (distance1 > 0) {
                	                potentials[idx] = potentials[idx] - p.changeToV/distance1;
	                      }
			}

                if(particles[idx] == 0) {
                        Ematrix[idx] = potentials[idx];
//			if (distance1 == 0) {
//                                Ematrix[idx] = -potentials[idx];
//                        }
                }
                else {
                        Ematrix[idx] = -potentials[idx];
//			if (distance1 == 0) {
//                                Ematrix[idx] = potentials[idx];
//                        }
                }



                }



	}

}

//combine potential energies and substrate energies to find total energies
__global__ void findE(int intN, REAL *Ematrix, REAL *particles, REAL *potentials, REAL *substrate) {
         int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
         if(idx<intN*intN) {
			Ematrix[idx] = particles[idx]*potentials[idx] + particles[idx]*substrate[idx];
//			Ematrix[idx] = particles[idx]*potentials[idx];
//			Ematrix[idx] =particles[idx]*substrate[idx];			
	}


}

//change the density of states from absolute value contribution to reflect removing and adding of a particle
__global__ void dosChange(int intN, REAL *particles,REAL *Ematrix,REAL *potentials) {
         int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
         if(idx<intN*intN) {
		if(particles[idx] == -1) {                        
			Ematrix[idx] = potentials[idx];
		}
		else {
			Ematrix[idx] = -potentials[idx];
		}
        }


}

//place particles
__global__ void particleDrop(int intN,int i ,int j,int newParticle,REAL *particles){
	particles[i + intN*j] = newParticle;
}

//find the potentials after a swap of positions
//generalized for deltaP>0
__global__ void potSwap(parameters p,int i1, int j1, int i2, int j2,int intN,REAL *particles,REAL *boxR,REAL *potentials){ 
        int x,y;
	int xPre,yPre;
	REAL distance1,distance2;
//	REAL before,after;

        int deltaP = G_abs(particles[i2 + intN*j2] - particles[i1 + intN*j1]);

	 int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	 if(idx<intN*intN) {
//		before = Ematrix[idx];


		if (deltaP == 0) { //if they are the same, then something has stacked p1 either started positive and became more positive or it started negative and became more negative
                        yPre = idx/intN;//if it works
                        xPre = idx%intN;//it works

                        x =  xPre + (intN/2 - i1);//for closed boundary dont do G_mod
                        y =  yPre + (intN/2 - j1); //instead only do if  yPre + (intN/2 - j1) > 0 or < intN
                        if(g_checkIn(x,y,intN)) { //only apply potential changes inside the system

				distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y]; // first I do the change at the first position
				if (distance1 > 0) {
//                                if (particles[i1 + intN*j1] < 0) {
        	                        if (particles[i1 + intN*j1] > 0) {
                	                        potentials[idx] = potentials[idx] + p.changeToV/distance1;
                        	        }
                                	else {
                                        	potentials[idx] = potentials[idx] - p.changeToV/distance1;
	                                }
        	               }
			}
			
                        x =  xPre + (intN/2 - i2);//for closed boundary dont do G_mod
                        y =  yPre + (intN/2 - j2); //instead only do if  yPre + (intN/2 - j1) > 0 or < intN
                        if(g_checkIn(x,y,intN)) { //only apply potential changes inside the system

	                        distance2 = boxR[i2 + intN*j2 + intN*intN*x + intN*intN*intN*y];
        	                if (distance2 > 0) {
//                                if (particles[i1 + intN*j1] > 0) { //might be the other way
					if (particles[i1 + intN*j1] <= 0) { //might be the other way
                        	                potentials[idx] = potentials[idx] + p.changeToV/distance2;
					}
	                                else {
        	                                potentials[idx] = potentials[idx] - p.changeToV/distance2;
                	                }
                        	}
			}
		}


	        if ((deltaP == 2) || (deltaP > 2)) { //if they are 1 change away from each other (a particle is traveling without blockade)
		// the last line could be written more simply, but this way shows that both scenarios act the same way
		//system is relaxing, kind of like the opposite of stacking, except this time it can go either way (but I only have to calculate the one way that it is going)

			yPre = idx/intN;//if it works
                        xPre = idx%intN;//it works
			

                        x =  xPre + (intN/2 - i1);//for closed boundary dont do G_mod
                        y =  yPre + (intN/2 - j1); //instead only do if  yPre + (intN/2 - j1) > 0 or < intN
                        if(g_checkIn(x,y,intN)) { //only apply potential changes inside the system

				distance1 = boxR[i1 + intN*j1 + intN*intN*x + intN*intN*intN*y];
				if (distance1 > 0) {
                	                if (particles[i1 + intN*j1] > particles[i2 + intN*j2]) {
						potentials[idx] = potentials[idx] + p.changeToV/distance1;
					}
	                                else {
						potentials[idx] = potentials[idx] - p.changeToV/distance1;
					}
	                       }
			}

                        x =  xPre + (intN/2 - i2);//for closed boundary dont do G_mod
                        y =  yPre + (intN/2 - j2); //instead only do if  yPre + (intN/2 - j1) > 0 or < intN
                        if(g_checkIn(x,y,intN)) { //only apply potential changes inside the system

	                        distance2 = boxR[i2 + intN*j2 + intN*intN*x + intN*intN*intN*y];//might be the other way
				if (distance2 > 0) {
        	                        if (particles[i2 + intN*j2] > particles[i1 + intN*j1]) {
						potentials[idx] = potentials[idx] + p.changeToV/distance2;
					 }
                                	else {
						potentials[idx] = potentials[idx] - p.changeToV/distance2;
			                }
				}
			}		
		}

        }
}
//force a particle to a certain place
void C_particleForce(vectors &v,int intN, int i1, int j1,int i2,int j2,int threads, int blocks,parameters p) {
			
			potSwap<<<blocks,threads>>>(p,i1, j1, i2, j2,intN,v.particles,v.boxR,v.potentials);
			particleSwap<<<blocks,threads>>>(i1, j1, i2,j2,intN,v.particles);
			findE<<<blocks,threads>>>(intN, v.Ematrix,v.particles,v.potentials,v.substrate);


}
int c_checkIn(int i, int j, int N) {
	int in = 0;
	if (i > 0 && i < N && j > 0 && j < N) {
		in = 1;	
	}
		
return in;	
}

__device__ int g_checkIn(int i, int j, int N) {
        int in = 0;
        if (i > 0 && i < N && j > 0 && j < N) {
                in = 1;
        }

return in;
}


//pick which site would result in a decrease of system energy (energies are negative, so this looks for the highest (closest to zero) value)
void C_particlePick(vectors &v,int intN, int i, int j,int threads, int blocks,parameters p) {
        if ((v.results[0] < v.results[1] ) ||(v.results[0] < v.results[2] ) ||(v.results[0] < v.results[3] ) ||(v.results[0] < v.results[4] ) ) {
        
	int iPrev,jPrev,iPost,jPost;
        iPrev = C_mod(i - 1,intN);
        jPrev = C_mod(j - 1,intN);
        iPost = C_mod(i + 1,intN);
        jPost = C_mod(j + 1,intN);

                if ((v.results[1] > v.results[2] ) &&(v.results[1] > v.results[3] ) &&(v.results[1] > v.results[4] )  ) {
			
			potSwap<<<blocks,threads>>>(p, i, j, iPrev,j,intN,v.particles,v.boxR,v.potentials);	
			particleSwap<<<blocks,threads>>>(i, j, iPrev,j,intN,v.particles);
			findE<<<blocks,threads>>>(intN, v.Ematrix,v.particles,v.potentials,v.substrate);

		}

                else if ((v.results[2] > v.results[3] ) &&(v.results[2] > v.results[4] )) {
                	potSwap<<<blocks,threads>>>(p, i, j, i,jPrev,intN,v.particles,v.boxR,v.potentials);
			particleSwap<<<blocks,threads>>>(i, j, i,jPrev,intN,v.particles);
			findE<<<blocks,threads>>>(intN, v.Ematrix,v.particles,v.potentials,v.substrate);

		}

                else if (v.results[3] > v.results[4]) {
   	            	potSwap<<<blocks,threads>>>(p, i, j, iPost,j,intN,v.particles,v.boxR,v.potentials);
			particleSwap<<<blocks,threads>>>(i, j, iPost,j,intN,v.particles);
			findE<<<blocks,threads>>>(intN, v.Ematrix,v.particles,v.potentials,v.substrate);

		}

                else {
			potSwap<<<blocks,threads>>>(p, i, j, i,jPost,intN,v.particles,v.boxR,v.potentials);
               		particleSwap<<<blocks,threads>>>(i, j, i,jPost,intN,v.particles);
			findE<<<blocks,threads>>>(intN, v.Ematrix,v.particles,v.potentials,v.substrate);

		 }
        }
}

//compare a site with its neighbors to find minimal system energy
void fastTest(vectors &v,int i, int j, int intN,int threads, int blocks,parameters p) {
        
	int iPrev,jPrev,iPost,jPost;
        iPrev = C_mod(i - 1,intN);
        jPrev = C_mod(j - 1,intN);
        iPost = C_mod(i + 1,intN);
        jPost = C_mod(j + 1,intN);
	REAL result;
	thrust::device_ptr<REAL> p_tempDos = thrust::device_pointer_cast(v.tempDos);
	thrust::device_ptr<REAL> p_Ematrix = thrust::device_pointer_cast(v.Ematrix);	

	result = thrust::reduce(p_Ematrix, p_Ematrix + intN*intN);
	v.results[0] = result;	
	if (c_checkIn(iPrev,j,intN)) {
		slowSwap<<<blocks,threads>>>(p, i,j, iPrev, j, intN, v.tempPar,v.tempPot,  v.tempDos, v.particles,v.boxR, v.substrate,v.Ematrix, v.watcher,v.potentials);
		result = thrust::reduce(p_tempDos, p_tempDos + intN*intN);
		v.results[1] = result;	
	}
	else {
		v.results[1] = -1e50; //large negative number so that it's not picked
	}

	if (c_checkIn(i,jPrev,intN)) {
	        slowSwap<<<blocks,threads>>>(p, i,j, i, jPrev, intN, v.tempPar,v.tempPot,  v.tempDos, v.particles,v.boxR, v.substrate,v.Ematrix, v.watcher,v.potentials);
        	result = thrust::reduce(p_tempDos, p_tempDos + intN*intN);
		v.results[2] = result;
	}
        else {
                v.results[2] = -1e50; //large negative number so that it's not picked
        }

        if (c_checkIn(iPost,j,intN)) {
	        slowSwap<<<blocks,threads>>>(p, i,j, iPost, j, intN, v.tempPar,v.tempPot,  v.tempDos, v.particles,v.boxR, v.substrate,v.Ematrix, v.watcher,v.potentials);
        	result = thrust::reduce(p_tempDos, p_tempDos + intN*intN);
		v.results[3] = result;	
	}
	else {
                v.results[3] = -1e50; //large negative number so that it's not picked
        }

        if (c_checkIn(i,jPost,intN)) {
	        slowSwap<<<blocks,threads>>>(p, i,j, i, jPost, intN, v.tempPar,v.tempPot,  v.tempDos, v.particles,v.boxR, v.substrate,v.Ematrix, v.watcher,v.potentials);
	        result = thrust::reduce(p_tempDos, p_tempDos + intN*intN);
		v.results[4] = result;	
	}
	else {
                v.results[4] = -1e50; //large negative number so that it's not picked
        }
	

	C_particlePick(v, intN,  i,  j, threads,  blocks,p);	
}


//when a particle is moved far away, there may be a cascade of changes as the particles around it accomodate. This checks for that in a spiral manner
void spiral(parameters p,int index,int blocks, int threads,vectors &v) { 
	int nLevels,xStart,yStart,ringLevel,ringLength,xNow,yNow,xCount,yCount;
	nLevels = 5;
	int intN = p.N;
	xStart = index%intN;
        yStart = index/intN;
	

	for (ringLevel = 1; ringLevel < nLevels; ringLevel++) {
		ringLength = ringLevel * 2 +1; 
		xNow = xStart + ringLevel;
	        yNow = yStart + ringLevel;

		for (xCount = 1; xCount < ringLength; xCount++) {
			xNow = xNow - 1;
			yNow = yNow;
			if(c_checkIn(xNow,yNow,intN)) {
				fastTest(v, xNow, yNow,  intN, threads, blocks,p);
			}
		}
		for (yCount = 1; yCount < ringLength; yCount++) {
			xNow = xNow;
			yNow = yNow - 1;
		        if(c_checkIn(xNow,yNow,intN)) {
                                fastTest(v, xNow, yNow,  intN, threads, blocks,p);
                        }
		}
                for (xCount = 1; xCount < ringLength; xCount++) {
                        xNow = xNow + 1;
                        yNow = yNow;
		        if(c_checkIn(xNow,yNow,intN)) {
                                fastTest(v, xNow, yNow,  intN, threads, blocks,p);
                        }
		}
		for (yCount = 1; yCount < ringLength; yCount++) {
                        xNow = xNow;
                        yNow = yNow + 1;
		        if(c_checkIn(xNow,yNow,intN)) {
                                fastTest(v, xNow, yNow,  intN, threads, blocks,p);
                        }
                }



	}

}

//see if youre close enough to a change in particles to be updated
__global__ void checkRange(int index,REAL *rangeMatrix,int intN) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx < intN*intN) {

	        int i,j,k,l;
	        double di,dj,dk,dl,r,dx,dy;
		i = index/intN;
		j = index%intN;
	        k = idx/intN;
	        l = idx%intN;

	
	        dk = (double) k;
	        dl = (double) l;
	    	di = (double) i;
       		dj = (double) j;

	        dx = dk - di;
        	dy = dl - dj;

	        r = sqrt(dx*dx + dy*dy);
        	rangeMatrix[idx] = 0;
		if (r < 10) {
			rangeMatrix[idx] = 1;
        	}
	}

}

int updateMinMax(vectors &v, int c_stable, REAL min_value, REAL max_value) {//apparently the loop can be 4 switches long (hopefully it doesnt go further)

	if(v.min1 == min_value && v.max1 == max_value) {
		c_stable = 1;	
	}

	if(v.min2 == min_value && v.max2 == max_value) {
		c_stable = 1;
	}
	
        if(v.min3 == min_value && v.max3 == max_value) {
                c_stable = 1;
        }

        if(v.min4 == min_value && v.max4 == max_value) {
                c_stable = 1;
        }


	v.min4 = v.min3;
	v.min3 = v.min2;
	v.min2 = v.min1;
	v.min1 = min_value;
	v.max4 = v.max3;
	v.max3 = v.max2;
	v.max2 = v.max1;
	v.max1 = max_value;	

return c_stable;
}


//see if the system has reached a local minimum
int checkStable(vectors &v,int c_stable,REAL min_value,REAL max_value,int min_offset,int max_offset,int intN,int blocks,int threads,parameters p){
	int i1,i2,j1,j2;
	
	c_stable = updateMinMax(v, c_stable, min_value,max_value); 

        if (c_stable == 0) {
	
                i1 = min_offset%intN;
                j1 = min_offset/intN;
                i2 = max_offset%intN;
                j2 = max_offset/intN;		

                potSwap<<<blocks,threads>>>(p,i1, j1, i2, j2,intN,v.particles,v.boxR,v.potentials);
		particleSwap<<<blocks,threads>>>(i1, j1, i2,j2,intN,v.particles);
		findE<<<blocks,threads>>>(intN, v.Ematrix,v.particles,v.potentials,v.substrate);
		
	}
	
	return c_stable;
}

//move a particle from high system energy to low system energy
int highsToLows(vectors &v,int max_offset,int min_offset,REAL max_value,REAL min_value,int c_stable, int blocks,int threads,parameters p) {
	c_stable = checkStable(v, c_stable, min_value, max_value, min_offset,max_offset,p.N, blocks,threads,p);

	if (c_stable == 0) {
		
		spiral(p,max_offset, blocks, threads,v);
		spiral(p,min_offset, blocks, threads,v);
	}
		
	return c_stable;
}
//grab only the positive values of a matrix ( for using the thrust function)
__global__ void grabPositives(REAL* particles,REAL *extraArray,REAL* Ematrix,int N,int posNeg) {
 int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if (idx < N*N) {
		extraArray[idx] = 0;
		if(particles[idx] == posNeg) {
			extraArray[idx] = Ematrix[idx];
		}
	}


}

//reflect the fact that holes are filled and full sites are emptied 
void __global__ lastFlip(int intN,REAL *invertedDos,REAL *particles) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx < intN*intN) {
		if (particles[idx] == -1) {
			invertedDos[idx] = -invertedDos[idx];
		}
	}
}
//calculate dos by removing the particle and seeing what effect this has on the system
void dosInvert (parameters p,int intN,int threads,int blocks,vectors &v) {//should work for nParticles > 1
	int i,j;
	double  result1,result2;
	thrust::device_ptr<REAL> g_go =  thrust::device_pointer_cast(v.tempDos);

	

	for(j = 0; j < intN; j++) {
		for (i = 0; i < intN; i++) {

//i = 20;
//j = 20;

			matrixCopy<<<blocks,threads>>>(intN, v.potentials ,v.tempPot);
			matrixCopy<<<blocks,threads>>>(intN, v.particles , v.tempPar);
			matrixCopy<<<blocks,threads>>>(intN, v.Ematrix ,v.tempDos);

			potChange<<<blocks,threads>>>(p,i, j,  intN,v.tempPar,v.boxR,v.tempPot,v.tempDos);
//			dosChange<<<blocks,threads>>>(intN, tempPar,tempDos,tempPot);
			result1 = thrust::reduce(g_go, g_go + intN*intN);

			potChange<<<blocks,threads>>>(p,i, j,  intN,v.tempPar,v.boxR,v.tempPot,v.tempDos);
//			dosChange<<<blocks,threads>>>(intN, tempPar,tempDos,tempPot);
			result2 = thrust::reduce(g_go, g_go + intN*intN);

			dosPut<<<blocks,threads>>>( i, j,intN,v.invertedDos, result2 - result1);
			
				
		}
	}
//lastFlip<<<blocks,threads>>>(intN,invertedDos,tempPar);


}

//do the half of the Glatz algorithm which uses a density of states map to find which particle switching is optimal
void switcharoo(vectors &v,int c_stable,int threads, int blocks,parameters p) {
	int intN = p.N;
	int min_offset,max_offset;
	REAL min_value,max_value;
	  thrust::device_ptr<REAL> g_ptr =  thrust::device_pointer_cast(v.Ematrix);
		thrust::device_ptr<REAL> inverted_ptr =  thrust::device_pointer_cast(v.extraArray);
        while (c_stable == 0) {
	

		grabPositives<<<blocks,threads>>>(v.particles,v.extraArray,v.Ematrix,intN,1);

		min_offset = thrust::min_element(inverted_ptr, inverted_ptr + intN*intN) - inverted_ptr;
		min_value = *(inverted_ptr + min_offset);
	
		grabPositives<<<blocks,threads>>>(v.particles,v.extraArray,v.Ematrix,intN,-1);		

              max_offset = thrust::min_element(inverted_ptr, inverted_ptr + intN*intN) - inverted_ptr; //grabbing the smallest positive number
              max_value = *(inverted_ptr + max_offset);
//		max_offset = thrust::max_element(inverted_ptr, inverted_ptr + intN*intN) - inverted_ptr;
//		max_value = *(inverted_ptr + max_offset);

//	potentialse = *(g_ptr + max_offset);
//cout<<min_value<<" "<<max_value<<endl;

	
		c_stable = highsToLows(v, max_offset,min_offset, max_value,min_value,c_stable, blocks,threads,p);
	}
	
}

//2 step relaxation algorithm developed by Glatz
void glatzRelax(int threads,int blocks,parameters p, vectors v) {

	int i,j,intN;
	intN = p.N;
	int N = p.N;
	int c_stable = 0;
	
/*
	for(j = 0; j < intN; j++) {
		for(i = 0; i < intN; i++) {
			potAdd<<<blocks,threads>>>( i, j,  intN, v.particles, v.potentials, v.boxR);

		}
	}
	findE<<<blocks,threads>>>(intN, v.Ematrix,v.particles,v.potentials,v.substrate); 
*/


for (int t = 0; t < 1; t++) {		
//original pair exchange
	for(j = 0; j < N; j++) {
	       	for(i = 0; i < N; i++) {
			fastTest(v, i, j,  intN, threads, blocks,p);
		}
	}
}

//highs to lows
	switcharoo(v,c_stable,threads, blocks,p);
	errorAsk("switching highs to lows");

	dosInvert (p, intN,threads,blocks,v);



	cudaFree(v.extraArray);
	cudaFree(v.rangeMatrix);


}

//initialize jump matrix
__global__ void	jumpFill(REAL* jumpRecord,int N) {
          int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	  if (idx < N) {
		jumpRecord[idx] = 999;
		
		
	}

}


//find minimum of 4 values
__device__ REAL findMin(REAL d1, REAL d2, REAL d3, REAL d4) {
	
	if (d1 < d2 && d1 < d3 && d1 < d4) {
		return d1;
		
	}
        if (d2 < d1 && d2 < d3 && d2 < d4) {
		return d2;
		
        }
        if (d3 < d1 && d3 < d2 && d3 < d4) {
		return d3;
		
        }
        if (d4 < d1 && d4 < d2 && d4 < d3) {
		return d4;
		
        }

	return d1; //default
}

//create matrix which finds individual granule radii(a) by linking it to distance between granules
__global__ void aMaker(REAL *aMatrix,REAL *boxR,int N) {
        int i,j,iPrev,jPrev,iPost,jPost;
	REAL distanceUp,distanceDown,distanceLeft,distanceRight, minDistance;
        int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
        if (idx < N*N) {
	
                i = idx%(N);
                j = (idx%(N*N) - idx%(N))/N;
		

	        iPrev = G_mod(i - 1,N);
	        jPrev = G_mod(j - 1,N);
        	iPost = G_mod(i + 1,N);
	        jPost = G_mod(j + 1,N);
	
		distanceUp = boxR[i + N*j + N*N*i + N*N*N*jPost];
		distanceDown = boxR[i + N*j + N*N*i + N*N*N*jPrev];
		distanceLeft = boxR[i + N*j + N*N*iPrev + N*N*N*j];
		distanceRight = boxR[i + N*j + N*N*iPost + N*N*N*j];

		minDistance = findMin(distanceUp,distanceDown,distanceLeft,distanceRight);
		aMatrix[idx] = minDistance/2;

        }

}
//substrate contribution from the granule size
__global__ void subCombine(REAL *aMatrix,REAL *substrate,REAL L, int N) {

        int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
        if (idx < N*N) {
	substrate[idx] = substrate[idx]*aMatrix[idx]/L;

	}

}

__global__ void tempGradient(REAL *TField,double lowTemp,double highTemp,int N) {

	int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	int i,j;
	double doubleI;
	if (idx < N*N) {
		i = idx/N;
		j = idx%N;
                doubleI = (double) i;
		TField[i + N*j] = lowTemp + doubleI*(highTemp-lowTemp)/N; 
			
	}
}

__global__ void noGradient(REAL *TField,double temperature, int N) {
        int idx=(blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
        if (idx < N*N) {
		TField[idx] = temperature;
        }


}


//load parameters
void paramLoad(parameters &p, char *argv[]){ 

        sprintf(p.lineName, "line.txt");
        sprintf(p.boxName, "box.txt");
        sprintf(p.timeName,"time.txt");
//      N = 32;
        p.N = 30;  //size of system (N x N)
//      N = 256;
        p.muVar = 0; // randomness of substrate (site energy?) -muvar to muvar
//      muVar = 1e-5;
//	p.boltzmann = 1.38e-23;
	p.boltzmann = .01;//test
//	p.changeToV = 1;//test
	p.changeToV = 3.6e-10; // Ke*Q/Kd 
//      eV = .05;
        p.eV = 0; //voltage (arbitrary units for now)
        p.Ec = 1600; //penalty for double-stacking
//      Ec = 1.6e-5;
//      Ec = 1;
//      T = 1;
        p.alphaOne = 1; // technically combined with density of states (not used at the moment)
//	p.alphaTwo = 1; // test
        p.alphaTwo = 1.16e4; //C/Kb (for converting to unitless)
        p.T = 25; //temperature
//      nParticles = input;
        p.nParticles = .5*p.N*p.N; //number of particles
//      nParticles = 1;
//      L = 7e-6;       
        p.L = 1e-8; //10 nm average inter-granule spacing
//      tSteps = 1000000; //for statistically accurate runs (timesteps)
//      tSteps = 10000; //for potential runs
        p.tSteps = 1; // for seeing the fields
//      tSteps = 0;
//      relax = 1;
        p.relax = 0; //wether or not to relax the system before running (should be 0 iff muVar & xyVar = 0)
        p.grabJ=0; //0 grabs average jumping distance , 1 grabs current
	p.whichBox=1;
        p.xi = p.L; //tunneling factor
        p.xVar = 0; //variance of lattice site in x direction
        p.yVar = 0; // typically = xVar
	p.rejection = 0; //default no rejection
	p.xMoment = 0;

	int intVal;
	REAL realVal;
	ifstream is_file(argv[1]);
	string line;
//this part loads  some of the variables
	while( getline(is_file, line) )
	{
		 istringstream is_line(line);
		 string key;
		  if( getline(is_line, key, '=') )
		  {
			string value;
			if( getline(is_line, value) )
//      store_line(key, value);
				if(key == "Temp") {
			       	        realVal = atof(value.c_str());
			                p.T = realVal;
			        }
			        if(key == "muVar") {
			                realVal = atof(value.c_str());
			                p.muVar = realVal;
			        }

			        if(key == "XYvar") {
			                realVal = atof(value.c_str());
			                p.xVar = realVal*p.L;
			                p.yVar = realVal*p.L;
			        }
			        if(key == "tSteps") {
			                intVal = atoi(value.c_str());
			                p.tSteps = intVal;
			        }
			
			        if(key == "L") {
			                realVal = atof(value.c_str());
			                p.L = realVal;
			        }
			
			        if(key == "eV") {
			                realVal = atof(value.c_str());
			                p.eV = realVal;
			        }
			
			        if(key == "relax") {
			                intVal = atoi(value.c_str());
			                p.relax = intVal;
			        }
			
			        if(key == "grabJ") {
			                intVal = atoi(value.c_str());
			                p.grabJ = intVal;
			        }
				if(key == "whichBox") {
					intVal = atoi(value.c_str());
                                        p.whichBox = intVal;
				}
			
			        if(key == "lineName") {
			                 sprintf(p.lineName,  value.c_str());
			
			//                lineName = value.c_str();
			        }
			
			        if(key == "boxName") {
			                 sprintf(p.boxName,  value.c_str());
			//                boxName = value.c_str();
		        	}
			        if(key == "timeName") {
			                 sprintf(p.timeName,  value.c_str());
			        }
				
                                if(key == "rejection") {
                                        realVal = atof(value.c_str());
                                        p.rejection = realVal;
                                }
 
                                if(key == "Ec") {
                                        realVal = atof(value.c_str());
                                        p.Ec = realVal;
                                }

	
			}
		}

	p.recordLength = p.tSteps;
}

//load arrays
void vectorLoad(vectors &v,parameters p,int blocks, int threads){
	int N = p.N;
        cudaMalloc(&v.watcher,N*N*sizeof(REAL));
        cudaMalloc(&v.reducedProb,N*N*sizeof(REAL));
        cudaMalloc(&v.particles,N*N*sizeof(REAL));
        cudaMalloc(&v.probabilities,N*N*sizeof(REAL));
        cudaMalloc(&v.potentials,N*N*sizeof(REAL));
        cudaMalloc(&v.substrate,N*N*sizeof(REAL));
        cudaMalloc(&v.Ematrix,N*N*sizeof(REAL));
        cudaMalloc(&v.tempDos,N*N*sizeof(REAL));
        cudaMalloc(&v.tempPar,N*N*sizeof(REAL));
        cudaMalloc(&v.tempPot,N*N*sizeof(REAL));
        cudaMalloc(&v.invertedDos,N*N*sizeof(REAL));
        cudaMalloc(&v.jumpRecord,p.recordLength*sizeof(REAL));
        cudaMalloc(&v.aMatrix,N*N*sizeof(REAL));
        cudaMalloc(&v.boxR,N*N*N*N*sizeof(REAL));
	cudaMalloc(&v.picked,sizeof(int));	

	v.herePicked = new int[1];
	v.herePicked[0] = 0;
	v.timeRun = new REAL[p.recordLength];
        v.sumRun = new REAL[p.recordLength];
	v.herePot =  new REAL[N*N];
        v.herePot = C_zeros(N, v.herePot);
        v.hereProb = new REAL[N*N];
        v.hereProb = C_random(N,0,v.hereProb);
        v.hereP = new REAL[N*N];
//	v.hereP = C_clump(p.N,p.nParticles,v.hereP);//test relaxation
        v.hereP = C_spread(N,p.nParticles,v.hereP); //test general potential
//	v.hereP = C_zeros(p.N,v.hereP); //zeros for the true neutral map (-2,0,2 ...etc)
//      hereP = C_random(N,nParticles,hereP);   
//      hereP = C_random(N,0,hereP); //empty system
//      hereP = C_more(N,nParticles,hereP);
        v.hereXDiff = new REAL[N*N];
        v.hereYDiff = new REAL[N*N];
        v.hereXDiff = createDiff(v.hereXDiff, p.xVar, N);
        v.hereYDiff = createDiff(v.hereYDiff, p.yVar, N);

        v.hereS = new REAL[N*N];
        v.hereS = createSub(v.hereS,p.muVar,N);
        v.hereBoxR = new REAL[N*N*N*N];
        v.hereBoxR = createR(v.hereBoxR,v.hereXDiff,v.hereYDiff,N,p.L,p.xi);
//      hereBoxR = createHex(hereBoxR,hereXDiff,hereYDiff,N,L,xi);      

        cudaMemcpy(v.watcher,v.herePot,N*N*sizeof(REAL),cudaMemcpyHostToDevice);
        cudaMemcpy(v.potentials,v.herePot,N*N*sizeof(REAL),cudaMemcpyHostToDevice);
        cudaMemcpy(v.Ematrix,v.herePot,N*N*sizeof(REAL),cudaMemcpyHostToDevice);//just filling it with 0s
        cudaMemcpy(v.substrate,v.hereS,N*N*sizeof(REAL),cudaMemcpyHostToDevice);
        cudaMemcpy(v.boxR,v.hereBoxR,N*N*N*N*sizeof(REAL),cudaMemcpyHostToDevice);
        cudaMemcpy(v.particles,v.hereP,N*N*sizeof(REAL),cudaMemcpyHostToDevice);
//        aMaker<<<blocks,threads>>>(v.aMatrix,v.boxR,N);
//        subCombine<<<blocks,threads>>>(v.aMatrix,v.substrate, p.L, N);
        jumpFill<<<blocks,threads>>>(v.jumpRecord,p.recordLength);

	v.min1 = 999;
	v.min2 = 999;
	v.min3 = 999;
	v.min4 = 999;
	v.max1 = 999;
	v.max2 = 999;
	v.max3 = 999;
	v.max4 = 999;


        int sizeSum = 6;

        v.hereSum = new REAL[sizeSum];
        v.hereSum = C_zeros(sizeSum,v.hereSum);

	cudaMalloc(&v.TField,N*N*sizeof(REAL));
        cudaMalloc(&v.extraArray,N*N*sizeof(REAL));
        cudaMalloc(&v.rangeMatrix,N*N*sizeof(REAL));
        cudaMalloc(&v.sumArray,sizeSum*sizeof(REAL));
        cudaMemcpy(v.sumArray,v.hereSum,sizeSum*sizeof(REAL),cudaMemcpyHostToDevice);
	
	int i,j;
        for(j = 0; j < p.N; j++) {
                for(i = 0; i < p.N; i++) {
                        potAdd<<<blocks,threads>>>(p, i, j,  p.N, v.particles, v.potentials, v.boxR);

                }
        }
        findE<<<blocks,threads>>>(p.N, v.Ematrix,v.particles,v.potentials,v.substrate); 
	noGradient<<<blocks,threads>>>(v.TField,15,p.N);
//	tempGradient<<<blocks,threads>>>(v.TField,15,500,p.N); //maximum gradient
//        tempGradient<<<blocks,threads>>>(v.TField,30,31,p.N); //almost no gradient

}

double findDipole(REAL *g_array,int size) {
        double moment;
	REAL *c_array;
        c_array =  new REAL[size*size];
        int k,l;
        cudaMemcpy(c_array,g_array,size*size*sizeof(REAL),cudaMemcpyDeviceToHost);
	for (k = 0; k < size; k++) {
		for (l = 0; l < size; l++) {
			if (c_array[k + size*l] == 1) {
				moment += k*c_array[k + size*l];
			}
		}

	}
	moment = 2*moment/(size*size); //half filled cells

return moment;

}

int main(int argc,char *argv[])
{
	
	cudaDeviceReset();
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaThreadSynchronize();

	parameters p;
        vectors v;

//	srand48(time(0));

	clock_t begin = clock();

	paramLoad(p,argv);
	int threads,blocks;
        int N = p.N;
	N = p.N;
	threads=MAXT;
	blocks=N*N/threads+(N*N%threads==0?0:1);

	vectorLoad(v,p,blocks, threads);

/*	
	char    nameP[256];
        sprintf(nameP, "line.txt");
	hereP = loadMatrix(hereP,nameP);
 */       
//system relax
        if (p.relax == 1) {
		glatzRelax(threads, blocks,p,v );
	}
	

//run simulation
	for(int t = 0; t < p.tSteps ; t++) {
		countThese = 1;
		v.tStep = t;
		findJump(v, threads, blocks,p);
	}


//save data
//	sprintf(str1, "line.txt");
//	printBoxCPU(hereXDiff,N,boxName);
	lastFlip<<<blocks,threads>>>(p.N,v.Ematrix,v.particles);

	switch(p.whichBox) {
	case 1:
		printBoxGPU(v.particles,p.N,p.boxName);
	break;
	case 2:
		printBoxGPU(v.probabilities,p.N,p.boxName);
	break;
	case 3:
		printBoxGPU(v.potentials,p.N,p.boxName);
	break;
	case 4:
		printBoxGPU(v.Ematrix,p.N,p.boxName);
	break;
	case 5:
		//no output box
	break;
	}
	printLineCPU(v.sumRun, p.timeName);
	printLineGPU(v.jumpRecord,p.recordLength,p.lineName);
//	printBoxGPU(v.watcher,p.N,"watcher.dat");	
	p.xMoment = findDipole(v.particles,p.N);	
	cout<<p.xMoment<<endl;
/*
        cudaMemcpy(hereP,jumpRecord,N*N*sizeof(REAL),cudaMemcpyDeviceToHost);
        FILE    *fp1;
//        char    str1[256];
  //      sprintf(str1, "particles.txt");
        fp1 = fopen(fileName, "w");
	for (int k = 0; k < N*N ; k++){
                fprintf(fp1, "%lf ",hereP[k]);
        }
//cleanup
	fclose(fp1);


*/
	delete[] v.herePicked;	
        delete[] v.herePot;
        delete[] v.hereProb;
        delete[] v.hereP;
        delete[] v.hereS;
	delete[] v.hereBoxR;
	cudaFree(v.particles);
	cudaFree(v.probabilities);
        cudaFree(v.potentials);
        cudaFree(v.substrate);
	cudaFree(v.boxR);
	cudaFree(v.Ematrix);
	cudaFree(v.jumpRecord);
	cudaFree(v.picked);
	clock_t end = clock();
  //	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

//cout<<"this took "<<elapsed_secs<<" seconds"<<endl;
}
