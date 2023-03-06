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

#include "aida_class.h"
#include "io_module.h"

using namespace std;

int main(int argc, char** argv){
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

  // AIDA setting	
	omp_set_num_threads(1);
	string score_type = "variance";              // Score function
	string dist_type  = "manhattan";             // Distance metric
	int dmin = (nFnum>5)?nFnum/2:nFnum;          // Dimensions to use in feature bagging.
	int dmax = (nFnum>5)?nFnum-1:nFnum;
	int subsample_min = 50, subsample_max = 512; // Min and max sizes of random subsamples
	double alpha_min = 1., alpha_max = 1.;       // Score function parameter
	
  // Train AIDA
	AIDA aida(N,"aom",score_type,alpha_min,alpha_max,dist_type);
	aida.fit(n,nFnum,Xnum,nFnom,Xnom,subsample_min,subsample_max,dmin,dmax,nFnom,nFnom);						
	aida.score_samples(n,scoresAIDA,Xnum,Xnom);
	
	ofstream fres("../results/"+tname+"_AIDA.dat");			
	fres<<n<<" "<<endl;
	for(int i=0;i<n;i++){
		fres<<scoresAIDA[i]<<endl;
	}
	fres.close();			
}
