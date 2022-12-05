/* Distance metrics */

#include "distance_metrics.h"

double l2dist(const int &n, const double* x, const double* y, const int* v){
	double l = 0.;
	for(int i=0;i<n;i++){
		double dist = x[v[i]]-y[v[i]];
		l += dist*dist;
	}
	
	return sqrt(l);
}

double l1dist(const int &n, const double* x, const double* y, const int* v){
	double l = 0.;
	for(int i=0;i<n;i++){
		l += abs(x[v[i]]-y[v[i]]);
	}
	
	return l;
}

double lmaxdist(const int &n, const double* x, const double* y, const int* v){
	double l = abs(x[v[0]]-y[v[0]]);
	for(int i=1;i<n;i++){
		double d = abs(x[v[i]]-y[v[i]]);
		if(d>l){
			l = d;
		}
	}
	
	return l;
}

double lsqrtdist(const int &n, const double* x, const double *y, const int* v){
	double l = 0.;
	for(int i=0;i<n;i++){
		l += sqrt(abs(x[v[i]]-y[v[i]]));
	}
	
	return l*l;
}
