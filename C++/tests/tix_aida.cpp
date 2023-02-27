/* Explanation method TIX for HiCs datasets */

#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
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
	ifstream fdata(fname+"_num.dat");
	ifstream fdata_cat(fname+"_nom.dat");	
	ifstream foutliers(fname+"_outliers.dat");	
	
	double num;
	int n, N = 100, niter = 1, nFnum, nFnom = 1, n_outliers;
	string pattern;
	
	/* Load outliers chosen by HiCs */
	foutliers>>n_outliers;
	
	/* Load data */
	fdata>>n;
	fdata.ignore();
	fdata>>nFnum;
	
	double *Xnum     = new double[n*nFnum];
	int    *Xnom     = new int[n*nFnom];
	int    *outliers = new int[n_outliers];
	
	for(int i=0;i<n_outliers;i++){
		foutliers>>outliers[i];
	}
	
	for(int i=0;i<n;i++){
		for(int j=0;j<nFnum-1;j++){
			fdata>>num;
			Xnum[j+i*nFnum] = num;
			fdata.ignore();
		}
		fdata>>num;
		Xnum[nFnum-1+i*nFnum] = num;
	}

	for(int i=0;i<n;i++){
		Xnom[i] = 0;
	}	
	
	omp_set_num_threads(1);
	string lnorm      = "1";           // distance norm
	string score_type = "variance";
	string dist_type  = "manhattan";	
  string file_num   = (argc>1)?string("_")+argv[1]:"";	
	
	
  // Train AIDA
	gettimeofday(&start, NULL);
	AIDA aida(N,"aom",score_type,1.,1.,dist_type);
	aida.fit(n,nFnum,Xnum,nFnom,Xnom,50,512,nFnum,nFnum,nFnom,nFnom);
	gettimeofday(&end, NULL);
	
	float delta = ((end.tv_sec-start.tv_sec)*1e6+end.tv_usec-start.tv_usec)/1.e6;
	
	cout<<"AIDA training time: "<<delta/niter<<endl;  
	
  
	delta = 0.;
	ostringstream ref_factor_str;
	double  ref_factor = 1.5;
	ref_factor_str<<setprecision(3)<<ref_factor;
  double *score_dim  = new double[nFnum];
  
	ofstream fex("../results/"+tname+"_TIX"+ref_factor_str.str()+file_num+".dat");
  fex<<n_outliers<<" "<<endl;
  for(int i=0;i<n_outliers;i++){
  	gettimeofday(&start, NULL);
    aida.tix(score_dim,&Xnum[outliers[i]*nFnum],&Xnom[outliers[i]*nFnom],ref_factor,10,1,10,50*nFnum,0.01,0.015);
  	gettimeofday(&end, NULL);
	  delta += ((end.tv_sec-start.tv_sec)*1e6+end.tv_usec-start.tv_usec)/1.e6;
		fex<<outliers[i]<<" ";
		for(int j=0;j<nFnum-1;j++){
			fex<<score_dim[j]<<" ";
		}
		fex<<score_dim[nFnum-1]<<endl;
  }
	fex.close();  
	
  cout<<"AIDA explanation time: "<<delta/niter<<endl;
}
