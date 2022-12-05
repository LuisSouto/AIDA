/* Shuffling functions */

#ifndef SHUFFLING_FUNCTIONS_H__
#define SHUFFLING_FUNCTIONS_H__

#include <iostream>
#include <random>
#include "types_def.h"

using namespace std;

typedef mt19937 RNG;

class rngClass{
	public:
  	RNG *gen_;
	  uniform_real_distribution<double> *distr_;	

		rngClass();
		
		~rngClass();

		double sample(const double &a=0., const double &b=1.);

		void partial_shuffle(const int &nshuf, const int &n, int *indices, int *idx);
		
		void undo_shuffle(const int &nshuf, int *indices, const int *idx);
		
		void partial_shuffle_back(const int &nshuf, const int &n, int *indices, int *idx);

		void undo_shuffle_back(const int &nshuf, const int &n, int *indices, const int *idx);		
		
		void variable_subsampling(const int &n, const int &nsubs, int* subsample_sizes, const int &nFeatures_c, const double* Xc,
		                          const int &nFeatures_d, const int* Xd, double* subsamples_c, int* subsamples_d, int size_min=50,
		                          int size_max=1000);
		                          
		void feature_bagging(const int &n, const int &nsubs, int* ndim_sub, int* id_subdim,
		                     int dmin, int dmax);
		                      
		void destroy_RNG();
	
};

#endif // SHUFFLING_FUNCTIONS_H__
