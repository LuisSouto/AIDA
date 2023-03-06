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
#include "io_module.h"

using namespace std;

int main(int argc, char** argv){
	struct timeval start, end;   // Time profiling 
	string tname = "example_cross_1000_50";
	string fname = "../synthetic_data/"+tname;
	string fnum  = fname+"_num.dat";           // filename with numerical features
	
	int n, N = 100, nFnum, nFnom = 1;
	double *Xnum = NULL; // Undefined behaviour if not NULL!
	
	read_data(Xnum,n,nFnum,fnum);
	double *scoresAIDA = new double[n];
	
	int *Xnom = new int[n*nFnom];
	for(int i=0;i<n*nFnom;i++){
		Xnom[i] = 0;
	}
	
	// Uncomment to use provided file with nominal features. In that case, also comment previous
	// declaration of Xnom.
//  int *Xnom = NULL;
//  read_data(Xnom,n,nFnom,fname+"_nom.dat");

	
	omp_set_num_threads(6);
	string lnorm         = "1";                            // distance norm
	string version[2]    = {"alpha1","alpharandom"};
	string score_type[2] = {"expectation","variance"};
	string dist_type     = "manhattan";
  string file_num      = (argc>1)?string("_")+argv[1]:"";
	int dmin = (nFnum>5)?nFnum/2:nFnum;  // Dimensions to use in feature bagging.
	int dmax = (nFnum>5)?nFnum-1:nFnum;
	
  // Train AIDA
  for(string score_t: score_type){
  	for(string alpha_v: version){  
			gettimeofday(&start,NULL);
			double alpha_min, alpha_max;
			if(alpha_v=="alpha1"){
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
			
			cout<<"AIDA training time: "<<delta<<endl;
			
			ofstream fres("../results/"+tname+"_AIDA_dist"+lnorm+"_"+score_t+"_"+alpha_v+file_num+".dat");			
			fres<<n<<" "<<endl;
			for(int i=0;i<n;i++){
				fres<<scoresAIDA[i]<<endl;
			}
			fres.close();			
		}
	}
}
