/* Shuffling functions */

#include "rng_class.h"


/* Random class initializer */
rngClass::rngClass(){
	random_device dev;
	gen_   = new RNG(dev());
	distr_ = new uniform_real_distribution<double>(0.,1.);			
}

rngClass::~rngClass(){
	destroy_RNG();
}


/* Uniform sample between [a,b). */
double rngClass::sample(const double &a, const double &b){
	return a+(b-a)*(*distr_)(*gen_);
}


/* Shuffle nshuf elements from indices. */
void rngClass::partial_shuffle(const int &nshuf, const int &n, int *indices, int *idx){
	for(int i=0;i<nshuf;i++){
		idx[i]          = i+(*distr_)(*gen_)*(n-i);
		int ind         = indices[i];
		indices[i]      = indices[idx[i]];
		indices[idx[i]] = ind;
	}
}

void rngClass::undo_shuffle(const int &nshuf, int *indices, const int *idx){
	for(int i=nshuf-1;i>=0;i--){
		int ind         = indices[idx[i]];
		indices[idx[i]] = indices[i];
		indices[i]      = ind;
	}
}

/* Shuffle nshuf elements from indices starting from the end. */
void rngClass::partial_shuffle_back(const int &nshuf, const int &n, int *indices, int *idx){
	for(int i=0;i<nshuf;i++){
		idx[i]          = (*distr_)(*gen_)*(n-i);
		int ind         = indices[n-1-i];
		indices[n-1-i]  = indices[idx[i]];
		indices[idx[i]] = ind;
	}
}

void rngClass::undo_shuffle_back(const int &nshuf, const int &n, int *indices, const int *idx){
	for(int i=nshuf-1;i>=0;i--){
		int ind         = indices[idx[i]];
		indices[idx[i]] = indices[n-1-i];
		indices[n-1-i]  = ind;
	}
}


void rngClass::variable_subsampling(const int &n, const int &nsubs, int* subsample_sizes, const int &nFeatures_c, const double* Xc,
                                    const int &nFeatures_d, const int* Xd, double* subsamples_c, int* subsamples_d, int size_min,
                                    int size_max){
                                     
  if(size_min>n){
  	size_min = n;
  }
  if(size_max>n){
  	size_max = n;
  }
	int *indices = new int[n];
	int *idx     = new int[size_max];

	for(int i=0;i<n;i++){
		indices[i] = i;
	}
	for(int i=0;i<nsubs;i++){
		subsample_sizes[i] = sample(size_min,size_max+1);
	}	

	for(int i=0;i<nsubs;i++){
		/* Indexes of the new subsample */
		partial_shuffle(subsample_sizes[i],n,indices,idx);
		for(int j=0;j<subsample_sizes[i];j++){
			for(int k=0;k<nFeatures_c;k++){
				subsamples_c[k+nFeatures_c*(j+i*size_max)] = Xc[k+indices[j]*nFeatures_c];
			}
			for(int k=0;k<nFeatures_d;k++){
				subsamples_d[k+nFeatures_d*(j+i*size_max)] = Xd[k+indices[j]*nFeatures_d];
			}			
		}		
		undo_shuffle(subsample_sizes[i],indices,idx);
	}
	
	delete[] indices;
	delete[] idx;	
}

void rngClass::feature_bagging(const int &n, const int &nsubs, int* ndim_sub, int* id_subdim,
                               int dmin, int dmax){
                                
  if(dmin>n){
  	dmin = n;
  }
  if(dmax>n){
  	dmax = n;
  }
                      
	int *indices_c = new int[n*nsubs];
	int *idx_c     = new int[n];	

	for(int i=0;i<n;i++){
		indices_c[i] = i;
	}

	for(int i=0;i<nsubs;i++){
		ndim_sub[i] = sample(dmin,dmax+1);
		partial_shuffle(ndim_sub[i],n,indices_c,idx_c);
		for(int j=0;j<ndim_sub[i];j++){
			id_subdim[j+i*n] = indices_c[j];
		}
		undo_shuffle(ndim_sub[i],indices_c,idx_c);
	}
	
	delete[] indices_c;
	delete[] idx_c;	
}


void rngClass::destroy_RNG(){
	delete gen_;
	delete distr_;
}
