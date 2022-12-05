/* Aggregation functions */

#include "aggregation_functions.h"

void normalize_scores(const int &nr, const int &nc, double *scores){
	double mean_score = 0.;
	double std_score  = 0.;
	
	for(int i=0;i<nc;i++){
		mean_score = 0.;
		std_score  = 0.;
		for(int j=0;j<nr;j++){
			mean_score += scores[i+j*nc];
			std_score  += scores[i+j*nc]*scores[i+j*nc];
		}
		mean_score /= nr;
		std_score   = sqrt(std_score/nr-mean_score*mean_score);
		for(int j=0;j<nr;j++){
			scores[i+j*nc] = (scores[i+j*nc]-mean_score)/std_score;
		}
	}	
}

void aggregate_average(const int &nr, const int &nc, double* scores_agg, const double *scores){
	for(int i=0;i<nc;i++){
		for(int j=0;j<nr;j++){
			scores_agg[j] += scores[i+j*nc];
		}
	}
	for(int i=0;i<nr;i++){
		scores_agg[i] /= nc;
	}	
}

void aggregate_maximum(const int &nr, const int &nc, double* scores_agg, const double *scores){
	for(int j=0;j<nr;j++){
		scores_agg[j] = scores[j*nc];
	}
	for(int i=1;i<nc;i++){
		for(int j=0;j<nr;j++){
			double new_score = scores[i+j*nc];
			if(new_score>scores_agg[j]){
				scores_agg[j] = new_score;
			}
		}
	}
}

void aggregate_aom(const int &nr, const int &nc, double* scores_agg, const double *scores){
	int bucket_size = 5, nbuckets = nc/bucket_size;
	for(int k=0;k<nr;k++){
		scores_agg[k] = 0.;
		for(int i=0;i<nbuckets;i++){
			double score = scores[i*bucket_size+k*nc];
			for(int j=1;j<bucket_size;j++){
				double new_score = scores[j+i*bucket_size+k*nc];
				if(new_score>score){
					score = new_score;
				}
			}
			scores_agg[k] += score;
		}
		scores_agg[k] /= nbuckets;
	}
}

void compute_counts(const int &n, const int* Xd, std::map<int,int> &count_map, const int &stride){
  for(int i=0;i<n;i++){
  	count_map[Xd[i*stride]]++;
  }
}
