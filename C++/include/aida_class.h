/* AIDA method (header) */

#ifndef AIDA_CLASS_H__
#define AIDA_CLASS_H__

#include <algorithm>
#include <iostream>
#include <map>
#include <math.h>
#include <numeric>
#include <omp.h>
#include <random>
#include <vector>
#include "types_def.h"
#include "aggregation_functions.h"
#include "distance_metrics.h"
#include "isolation_formulas.h"
#include "rng_class.h"

using namespace std;

class AIDA{
  public:
		int nFnum_, nFnom_, N_;
		int subsize_min_, subsize_max_;
		double alpha_min, alpha_max;
		vector<int> ndim_num, id_numdim;
		vector<int> ndim_nom, id_nomdim;		
		vector<int> subsample_sizes, subsamples_nom;
		vector<double> subsamples_num, alpha_;
		double (*score_func)(const int &, const double*, const double &);
		void   (*aggregate_scores_)(const int &, const int &, double*, const double*);
		rngClass *myrng_;
		map<int,int> *count_map;
		distance_pointer *distance_;
		
		// Constructor
    AIDA(const int &N=100, string aggregate_type="aom", string score_function="variance",
              const double &alpha_min=1., const double &alpha_max=1., const string &distance_metric="manhattan");
		
		~AIDA();
		
    void fit(const int &n, const int &nFnum, const double* Xnum, const int &nFnom,
             const int *Xnom, const int &subsize_min=50, const int &subsize_max=1000, const int &dmin_num=1,
             const int &dmax_num=10, const int &dmin_nom=1, const int &dmax_nom=10);
		
    void score_samples(const int &n, double *scores, const double* Xnum, const int* Xnom);
                     
    void compute_scores(double* scores, const double* Xnum, const int* Xnom);
    
    double compute_scores_subsample(const int &i, const double* Xnum, const int* Xnom);
    
		void tix(double* score_dim, const double* Xnum, const int* Xnom, const double &ref_rate=2.,
             int dmin_num=-1, const int &niter_ref=1, const int &niter_tix=1, int maxlen=-1,
             const double &Tmin=0.01, const double &Tmax=0.015);
		
		void explain(double* score_dim, const int &ndim, int* id_dim, const double* Xnum,
                 const int* Xnom, const int &niter=1, int maxlen=-1, const double &Tmin=0.01,
                 const double &Tmax=0.015);
		
		void destroy_RNG();
		
		void clear();
};

#endif // AIDA_CLASS_H__
