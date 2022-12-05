/* General tester for AIDA method */

#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <sys/time.h>
#include <string>
#include <math.h>
#include <vector>
#include <numeric>
#include <bits/stdc++.h>
#include <omp.h>

#include "isolation_formulas.h"
#include "aida_class.h"
#include "distance_metrics.h"

using namespace std;

int main(int argc, char** argv){
	struct timeval start, end;   // Time profiling 
	string tname = "example_cross_1000_50";
	string fname = "../synthetic_data/"+tname;
	ifstream fdata_num(fname+"_num.dat");
	
	double Xnum_temp;
	int    Xnom_temp;
	int n, N = 100, niter = 1, nFnum, nFnom = 1;
	
	/* Load data */
	fdata_num>>n;
	fdata_num.ignore();
	fdata_num>>nFnum;
	
	double *Xnum = new double[n*nFnum], *scoresAIDA = new double[n];
	
	for(int i=0;i<n;i++){
		for(int j=0;j<nFnum-1;j++){
			fdata_num>>Xnum_temp;
			Xnum[j+i*nFnum] = Xnum_temp;
			fdata_num.ignore();
		}
		fdata_num>>Xnum_temp;
		Xnum[nFnum-1+i*nFnum] = Xnum_temp;
	}


/* Uncomment these lines to use nominal features. Otherwise just create an array of zeros. */
//	ifstream fdata_nom(fname+"_nom.dat");
//	fdata_nom>>n;
//	fdata_nom.ignore();
//	fdata_nom>>nFnom;
	
	int *Xnom = new int[n*nFnom];
	
	for(int i=0;i<n*nFnom;i++){
		Xnom[i] = 0;
	}
	
//	for(int i=0;i<n;i++){
//		for(int j=0;j<nFnom-1;j++){
//			fdata_nom>>Xnom_temp;
//			Xnom[j+i*nFnom] = Xnom_temp;
//			fdata_nom.ignore();
//		}
//		fdata_nom>>Xnom_temp;
//		Xnom[nFnom-1+i*nFnom] = Xnom_temp;
//	}
	
	omp_set_num_threads(4);
	string lnorm         = "1";                            // distance norm
	string version[2]    = {"alpha1_","alpharandom_"};
	string score_type[2] = {"expectation","variance"};
	string dist_type     = "manhattan";
	int dmin = (nFnum>5)?nFnum/2:nFnum;  // Dimensions to use in feature bagging.
	int dmax = (nFnum>5)?nFnum-1:nFnum;
	
	string file_number;
	if(argc>1){
		file_number = argv[1];
	}
	else{
	  file_number = "1";
	}
	
  // Train AIDA
  for(string score_t: score_type){
  	for(string alpha_v: version){  
			gettimeofday(&start,NULL);
			double alpha_min, alpha_max;
			if(alpha_v=="alpha1_"){
				alpha_min = 1.;
				alpha_max = 1.;
			}
			else{
				alpha_min = 0.5;
				alpha_max = 1.5;
			}
			AIDA aida(N,"aom",score_t,alpha_min,alpha_max,dist_type);
			aida.fit(n,nFnum,Xnum,nFnom,Xnom,50,512,dmin,dmax,nFnom,nFnom);						
			aida.score_samples(n,scoresAIDA,Xnum,Xnom);
			gettimeofday(&end,NULL);
			
			float delta = ((end.tv_sec-start.tv_sec)*1e6+end.tv_usec-start.tv_usec)/1.e6;
			
			cout<<"AIDA training time: "<<delta/niter<<endl;
			
			ofstream fres("../results/"+tname+"_AIDA_dist"+lnorm+"_"+score_t+"_"+alpha_v+file_number+".dat");			
			fres<<n<<" "<<endl;
			for(int i=0;i<n;i++){
				fres<<scoresAIDA[i]<<endl;
			}
			fres.close();			
		}
	}
	
	
	
	fdata_num.close();
//	fdata_nom.close();
}
