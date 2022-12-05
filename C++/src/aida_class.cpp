/* AIDA method: class implementation */

/* This file contains the following methods of the AIDA class:          
      - AIDA: Constructor of the class. 
         INPUT:
         + N: Number of subsamples. Deault N=100.
         
         + aggregate_type: Aggregration function for the scores among different subsamples. 
                           Supported values are: "average", "maximum" and "aom" (Average Of
                           Maximum). Default aggregate_type="average".
                           
         + score_function: Outlier score function. Supported values are "expectation" and 
                           "variance". Default score_function="variance".
         
         + alpha_min: Minimum value of alpha. Must be positive. Default alpha_min=1.
                      
         + alpha_max: Maximum value of alpha. Must be positive. Default alpha_max=1.
                      
         + distance_metric: Distance function to be used with numerical features. Supported
                            values are "manhattan", "euclidean", "infinite", "sqrt" and "random".
                            Using "random" implies a random distance_metric per subsample,
                            where the metrics are chosen uniformly from the previous four options.
                            Default distance_metric="manhattan".
                            
         OUTPUT:
         + Instance of AIDA class.
         
      - ~AIDA: Destructor of the class.
         INPUT: None
         OUTPUT: None
         
      - fit: Creates the subsamples from the training data.
         INPUT:
         + n: Size of the data set.
         
         + nFnum: Number of numerical features.
         
         + Xnum: Data set with only the numerical features.
         
         + nFnom: Number of nominal features.
         
         + Xnom: Data set with only nominal features. The different categories
                 in each feature must be converted to integers,but the algorithm
                 is insensitive to a particular choice of the conversion scheme.
               
         + subsize_min: Minimum size of the subsamples. Default is
                        subsize_min=50.
                        
         + subsize_max: Maximum size of the subsamples. Default is
                        subsize_max=512.
                        
         + dmin_num: Minimal number of numerical features in the feature bagging
                     algorithm. Default is dmin_num=nFnum.
                 
         + dmax_num: Maximal number of numerical features in the feature bagging
                     algorithm. Default is dmax_num=nFnum.
                 
         + dmin_nom: Minimal number of nominal features in the feature bagging
                     algorithm. Default is dmin_nom=nFnom.
                 
         + dmax_nom: Maximal number of nominal features in the feature bagging
                     algorithm. Default is dmax_nom=nFnom.
                   
         OUTPUT: None.
      
      - score_samples: Computes the normalized outlier scores of the test data.
         INPUT:
         + n: Size of the test data.
         
         + scores: Vector to copy the final scores. It should already be
                   allocated with a size of n*double.
                   
         + Xnum: Test data with only numerical features. Should be size (n x nFnum).
         
         + Xnom: Test data with only nominal features. The different categories
                 in each feature must be converted to integers,but the algorithm
                 is insensitive to a particular choice of the conversion scheme.
                 Should be size (n x nFnom).
               
         OUTPUT: None.
               
               
      - compute_scores: Computes the unnormalized score of a single observation.
         INPUT:
         + scores: Vector to copy the final scores. It should already be
                   allocated with a size of N*double.
                   
         + Xnum: Observation with only numerical features. Should be size nFnum.
         
         + Xnom: Observation with only nominal features. Should be size nFnom.
         
         OUTPUT: None.
         
      - compute_scores_subsample: Computes the unnormalized score of a single observation
                                  per subsample.
         INPUT:
         + i: Index of the subsample. Valid values are integer between 0 and (N-1).

         + Xnum: Observation with only numerical features. Should be size nFnum.
         
         + Xnom: Observation with only nominal features. Should be size nFnom.
         
         OUTPUT:
         + score: Unnormalized outlier score of the input observation in the i-th subsample.
         
      - tix: Explanation algorithm. For a given observation, it returns a vector of size
             nFnum with the importance score of each numerical feature. The higher
             the importance score, the more relevant the feature is in the explanation
             process. The function "fit" must be called beforehand in order to create the
             subsamples. Otherwise, the behaviour is undefined.
             
         INPUT:
         + score_dim: Vector to copy the importance scores. It should already by allocated
                      with size nFnum_*double.
                      
         + Xnum: Observation with only numerical features. Should be size nFnum_.
         
         + Xnom: Observation with only nominal features. Should be size nFnom.

         + ref_rate: Refinement rate. Indicates the rate at which the feature space is reduced 
                     at each iteration. Default is ref_rate=2.
                     
         + dmin_num: Minimum number of features required for the refinement step. Default is nFnum_.         
         
         + niter_ref: Number of iterations of the refinement step. Must be positive. Default
                      is niter_ref=1.
                      
         + niter_tix: Number of iterations of the tix scoring function. Must be positive. Default
                      is niter_tix=1.
                      
         + max_len: Maximum path length. Default is max_len=50 x nFnum_.
         
         + Tmin: Minimum temperature for the acceptance criterion. Deault is Tmin=0.01.
         
         + Tmax: Maximum temperature for the acceptance criterion. Must be larger or equal than Tmin.
                 Deault is Tmax=0.015.
                      
                      
         OUTPUT: None.

      - explain: Core of the tix algorithm. Computes the importance scores of the input feature
                 subspace.
                 
         INPUT:
         + score_dim: Vector to copy the importance scores. It should already by allocated
                      with size nFnum_*double.
                      
         + ndim: Number of numerical features to be analyzed.
         
         + id_dim: Indexes of the numerical features to be analyzed. Each index must be an integer
                   between 0 and (nFnum_-1), and there cannot be repetitions.
                   
         + Xnum: Observation with only numerical features. Should be size nFnum_.
         
         + Xnom: Observation with only nominal features. Should be size nFnom.
         
         + niter: Number of iterations. Must be positive. Default is niter=1.
         
         + max_len: Maximum path length. Default is max_len=50 x nFnum_.
         
         + Tmin: Minimum temperature for the acceptance criterion. Deault is Tmin=0.01.
         
         + Tmax: Maximum temperature for the acceptance criterion. Must be larger or equal than Tmin.
                 Deault is Tmax=0.015.         
         
         OUTPUT: None.
         
      - destroy_RNG: Frees the memory related to the random number generator.
      
         INPUT: None.
        
         OUTPUT: None.
        
      - clear(): Frees memory from all allocated arrays during the constructor and fit functions.
      
         INPUT: None.
         
         OUTPUT: None 
*/         
                   
#include "aida_class.h"

/* Deault constructor of the AIDA class */
AIDA::AIDA(const int &N, string aggregate_type, string score_function, const double &alpha_min,
           const double &alpha_max, const string &distance_metric){
  N_     = N;
  myrng_ = new rngClass();
  
  /* Aggregation function for the outlier scores */
	if(aggregate_type=="average"){
		aggregate_scores_ = &aggregate_average;
	}
	else if(aggregate_type=="maximum"){
		aggregate_scores_ = &aggregate_maximum;
	}
	else if(aggregate_type=="aom"){
		aggregate_scores_ = &aggregate_aom;
	}
	else{
		aggregate_scores_ = &aggregate_average;
  }
  
  /* Type of score function */
  if(score_function=="expectation"){
  	score_func = &left_fringe_meanval_gen;
  }
  else if(score_function=="variance"){
  	score_func = &left_fringe_var_gen;
  }
  else{
    score_func = &left_fringe_var_gen;
  }
  
  /* Generate random alpha */
	alpha_.reserve(N_);
  for(int i=0;i<N_;i++){
  	alpha_[i] = myrng_->sample(alpha_min,alpha_max);
  }
  
  /* Select the distance metric */
	distance_ = new distance_pointer[N_];
	for(int i=0;i<N_;i++){
		int dist_type;
		if(distance_metric=="random"){
			dist_type = 4*myrng_->sample();
		}
		else if(distance_metric=="manhattan"){
			dist_type = 0;
		}
		else if(distance_metric=="euclidean"){
			dist_type = 1;
		}
		else if(distance_metric=="infinite"){
			dist_type = 2;
		}
		else if(distance_metric=="sqrt"){
			dist_type = 3;
		}		
		else{
			dist_type = 0;
		}
		if(dist_type==0){
			distance_[i] = &l1dist;
		}
		else if(dist_type==1){
			distance_[i] = &l2dist;
		}
		else if(dist_type==2){
			distance_[i] = &lmaxdist;
		}		
		else{
			distance_[i] = &lsqrtdist;
		}
	}  
}

/* Destructor */
AIDA::~AIDA(){
	destroy_RNG();
	clear();
}


/* Creates N subsamples from the training data Xnum and Xnom */
void AIDA::fit(const int &n, const int &nFnum, const double* Xnum, const int &nFnom,
               const int *Xnom, const int &subsize_min, const int &subsize_max, const int &dmin_num,
               const int &dmax_num, const int &dmin_nom, const int &dmax_nom){
	nFnum_ = nFnum;
	nFnom_ = nFnom;
	subsize_min_ = min(n,subsize_min);
	subsize_max_ = min(n,subsize_max);
	ndim_num.reserve(N_);
	id_numdim.reserve(N_*nFnum_);
	ndim_nom.reserve(N_);
	id_nomdim.reserve(N_*nFnom_);
	subsample_sizes.reserve(N_);
	subsamples_nom.reserve(N_*subsize_max_*nFnom_);
	subsamples_num.reserve(N_*subsize_max_*nFnum_);
	count_map = new map<int,int>[nFnom_*N_];
  
	// Generate the subspaces, treating numeric and nominal features separately
	myrng_->feature_bagging(nFnum_,N_,&ndim_num[0],&id_numdim[0],dmin_num,dmax_num);
	myrng_->feature_bagging(nFnom_,N_,&ndim_nom[0],&id_nomdim[0],dmin_nom,dmax_nom);

  // Generate the subsamples                           
  myrng_->variable_subsampling(n,N_,&subsample_sizes[0],nFnum_,Xnum,nFnom,Xnom,&subsamples_num[0],&subsamples_nom[0],subsize_min_,subsize_max_);
  
  /* Compute the counts of each class in the nominal features */
  for(int i=0;i<N_;i++){
  	int inFd = i*nFnom_;
  	for(int j=0;j<ndim_nom[i];j++){
	    compute_counts(subsample_sizes[i],&subsamples_nom[id_nomdim[j+inFd]+inFd*subsize_max_],count_map[j+inFd],nFnom_); 
	  }
  }
}

/* Compute outlier scores of the test data Xnum and Xnom. */
void AIDA::score_samples(const int &n, double *scores, const double* Xnum, const int* Xnom){
	double *scores_subsample = new double[N_*n];
	fill(scores,scores+n,0.);
	
	// Compute the scores
	for(int i=0;i<n;i++){
		compute_scores(&scores_subsample[i*N_],&Xnum[i*nFnum_],&Xnom[i*nFnom_]);
	}
	
	// Aggregate the scores	
	if(n>1){
		normalize_scores(n,N_,scores_subsample);
	}
	aggregate_scores_(n,N_,scores,scores_subsample);
	
	delete[] scores_subsample;
}

void AIDA::compute_scores(double* scores, const double* Xnum, const int* Xnom){	
	#pragma omp parallel for
	for(int i=0;i<N_;i++){
    scores[i] = compute_scores_subsample(i,Xnum,Xnom);
	}
}

double AIDA::compute_scores_subsample(const int &i, const double* Xnum, const int* Xnom){
	double *dX = new double[subsample_sizes[i]+1];
	dX[0]      = 0.;
	
	int inFc = i*nFnum_;
	int inFd = i*nFnom_;
	int isub = i*subsize_max_;
	int sub2 = subsample_sizes[i]*(subsample_sizes[i]+1.);
	
	for(int j=0;j<subsample_sizes[i];j++){
	  /* Contribution of the numerical features */
		dX[j+1] = distance_[i](ndim_num[i],Xnum,&subsamples_num[nFnum_*(j+isub)],&id_numdim[inFc]);
		
		/* Contribution of the nominal features */
		for(int k=0;k<ndim_nom[i];k++){
			if(Xnom[id_nomdim[k+inFd]]!=subsamples_nom[id_nomdim[k+inFd]+nFnom_*(j+isub)]){
				int counts_j = count_map[k+inFd][subsamples_nom[id_nomdim[k+inFd]+nFnom_*(j+isub)]];
				dX[j+1]     -= log(1.-counts_j*(counts_j-1.)/sub2); 
			}
		}
	}

	sort(dX,dX+subsample_sizes[i]+1);
	
	double score = -score_func(subsample_sizes[i]+1,dX,alpha_[i]);
	
	delete[] dX;
	
	return score;
}


/* Explanation method TIX: refinement step. Currently, only working for numerical features. */
void AIDA::tix(double* score_dim, const double* Xnum, const int* Xnom, const double &ref_rate,
               int dmin_num, const int &niter_ref, const int &niter_tix, int maxlen,
               const double &Tmin, const double &Tmax){
               
	int  nFc    = nFnum_;
	int *id_dim = new int[nFc];
	
	// Default
	if(dmin_num==-1){
		dmin_num = nFc;
	}
	
	double *score_tix = new double[nFc];
	fill(score_dim,score_dim+nFc,0.);
	
	for(int k=0;k<niter_ref;k++){
		nFc      = nFnum_;
		int ndim = nFc;
		std::iota(id_dim,id_dim+ndim,0);  // Feature space from 0 to ndim-1.
		while(ndim>=dmin_num){
			explain(score_tix,ndim,id_dim,Xnum,Xnom,niter_tix,maxlen,Tmin,Tmax);
			sort(id_dim,id_dim+ndim,[&](int i,int j){return score_tix[i]>score_tix[j];});
			int ndim_old = ndim;
			nFc          = ndim;
			if(ndim>dmin_num){
				ndim = max((int)(ndim/ref_rate),dmin_num); // Apply refinement
			}
			else{
				ndim = 0;
			}
			nFc -= ndim;  // Number of features discarded at this iteration
			for(int i=0;i<nFc;i++){
				score_dim[id_dim[i+ndim]] += score_tix[id_dim[i+ndim]]+nFnum_-ndim_old;
//				score_dim[id_dim[i+ndim]] += nFnum_-1-i-ndim;  // Uncomment to use ranks instead of path lengths as scores.
			}
		}
	}
	
	delete[] id_dim;
	delete[] score_tix;
}


/* TIX: importance score calculation. Currently, only working for numerical features. */
void AIDA::explain(double* score_dim, const int &ndim, int* id_dim, const double* Xnum,
                   const int* Xnom, const int &niter, int maxlen, const double &Tmin,
                   const double &Tmax){
                   
	int  nFc      = (ndim>nFnum_)?nFnum_:ndim;
	int  nFc_     = nFnum_;
	int  nthreads = omp_get_max_threads();
	int  sub1     = subsize_max_+1;
	int *idc      = new int[nthreads];
	int *id_dim_c = new int[nFc*nthreads];
	for(int i=0;i<nthreads;i++){
		copy(id_dim,id_dim+nFc,&id_dim_c[i*nFc]);
	}
	
	// Default value
  if(maxlen==-1){
  	maxlen = 50*nFc;
  }	
	
	bool *repeated = new bool[nFc_*nthreads];
	fill(repeated,repeated+nFc_*nthreads,false);
	
	double *score_temp = new double[nFc_*nthreads];
	double *path_len   = new double[nFc_*N_*niter];
	double *dX_dim     = new double[N_*subsize_max_*nFc_];
	double *dX         = new double[sub1*nthreads];
	double *dX_sorted  = new double[sub1*nthreads];
  fill(score_dim,score_dim+nFc_,0.);  	
	fill(path_len,path_len+nFc_*N_*niter,0.);	
  fill(score_temp,score_temp+nFc_*nthreads,0.);  	
  fill(dX_dim,dX_dim+N_*subsize_max_*nFc_,0.);

	rngClass *rnd_gens = new rngClass[nthreads];

	// Compute and store distances	
	for(int i=0;i<nthreads;i++){
		dX[i*sub1] = 0.;
		dX_sorted[i*sub1] = 0.;
	}
		
	for(int i=0;i<N_;i++){
		int isub = i*subsize_max_;
	
		for(int j=0;j<subsample_sizes[i];j++){
			for(int k=0;k<nFc;k++){
				dX_dim[id_dim[k]+(j+isub)*nFc_] = abs(Xnum[id_dim[k]]-subsamples_num[id_dim[k]+nFc_*(j+isub)]);
			}
		}
	}
  
  for(int k=0;k<niter;k++){
  	# pragma omp parallel for
  	for(int i=0;i<N_;i++){
			int nFeat = nFc;
			int tid  = omp_get_thread_num();
		  int inFc = i*nFc_;
			int isub = i*subsize_max_;
			double T = (Tmin+(Tmax-Tmin)*rnd_gens[tid].sample())/log(10./9.);
			
			for(int j=0;j<subsample_sizes[i];j++){
				dX[j+1+tid*sub1] = 0.;
				for(int l=0;l<nFc;l++){
					dX[j+1+tid*sub1] += dX_dim[id_dim_c[l+tid*nFc]+(j+isub)*nFc_];
				}
				dX_sorted[j+1+tid*sub1] = dX[j+1+tid*sub1];
			}
			sort(&dX_sorted[1+tid*sub1],&dX_sorted[1+tid*sub1]+subsample_sizes[i]);
			fill(&repeated[tid*nFc_],&repeated[tid*nFc_]+nFc_,false);
		
			double max_new = -left_fringe_var(subsample_sizes[i]+1,&dX_sorted[tid*sub1]);
			double max_old = max_new;
			int len_count  = 0;
			for(int l=0;l<maxlen;l++){
				if(nFeat==1){
					break;
				}
				
				// If the same dimension has been chosen twice before an acceptance, there is no need to recompute anything.
				rnd_gens[tid].partial_shuffle_back(1,nFeat,&id_dim_c[tid*nFc],&idc[tid]);
				int dim_rem = id_dim_c[nFeat-1+tid*nFc];

				if(not repeated[dim_rem+tid*nFc_]){			
		      for(int j=0;j<subsample_sizes[i];j++){     
		      	dX[j+1+tid*sub1]       -= dX_dim[dim_rem+(j+isub)*nFc_];
		      	dX_sorted[j+1+tid*sub1] = dX[j+1+tid*sub1];
		      }
		      
		      sort(&dX_sorted[1+tid*sub1],&dX_sorted[1+tid*sub1]+subsample_sizes[i]);
					score_temp[dim_rem+tid*nFc_] = -left_fringe_var(subsample_sizes[i]+1,&dX_sorted[tid*sub1]);
				}
				max_new = score_temp[dim_rem+tid*nFc_];
				
				// Apply acceptance criterium (SA)
				if(max_new>=max_old){
					path_len[dim_rem+inFc+k*nFc_*N_] = l;
					max_old = max_new;					
					nFeat--;
					for(int j=0;j<nFeat;j++){
						repeated[id_dim_c[j+tid*nFc]+tid*nFc_] = false;
					}
				}
				else{
					if(exp(-(max_new-max_old)/(T*max_old))>rnd_gens[tid].sample()){
						path_len[dim_rem+inFc+k*nFc_*N_] = l;
						max_old = max_new;						
						nFeat--;
						for(int j=0;j<nFeat;j++){
							repeated[id_dim_c[j+tid*nFc]+tid*nFc_] = false;
						}
					}
					else{
						if(not repeated[dim_rem+tid*nFc_]){
			      	for(int j=0;j<subsample_sizes[i];j++){     
			     	  	dX[j+1+tid*sub1] += dX_dim[dim_rem+(j+isub)*nFc_];
			     		}
				     	repeated[dim_rem+tid*nFc_] = true;			  
				    }
					}
				}
				len_count = l+1;
			}

			if(nFeat>1){
				for(int j=0;j<nFeat;j++){
					path_len[id_dim_c[j+tid*nFc]+inFc+k*nFc_*N_]  = len_count;
				}
			}
			else{
				path_len[id_dim_c[tid*nFc]+inFc+k*nFc_*N_]  = len_count;
			}
		}
  }

  double *path_mean  = new double[nFc];
  for(int i=0;i<nFc;i++){
  	path_mean[i] = 0.;
  	for(int j=0;j<N_*niter;j++){
  		double path_j = path_len[id_dim[i]+j*nFc_];
  		path_mean[i] += path_j;
  	}
  	path_mean[i]        /= N_*niter;
  	score_dim[id_dim[i]] = path_mean[i];
  }

  delete[] idc;
  delete[] id_dim_c;
  delete[] repeated;
  delete[] score_temp;
  delete[] dX_dim;
  delete[] dX;
  delete[] dX_sorted;
  delete[] rnd_gens;
  delete[] path_len;
  delete[] path_mean;
}

void AIDA::destroy_RNG(){
	delete myrng_;
}

void AIDA::clear(){
	for(int i=0;i<N_;i++){
		for(int j=0;j<nFnom_;j++){
		  count_map[j+i*nFnom_].clear();
		}
	}
	
	ndim_num.clear();
	id_numdim.clear();
	ndim_nom.clear();
	id_nomdim.clear();
	subsample_sizes.clear();
	subsamples_num.clear();
	subsamples_nom.clear();
	alpha_.clear();
	delete[] count_map;
}
