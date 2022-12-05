/* Theoretical formulas for isolating points in 1D */

#include "isolation_formulas.h"

/* Expectation function with alpha=1 */
double left_fringe_meanval(const int &n, const double *Z){
	if(n==1){
		return 0.;
	}

	double meanval = 1., cumprob;
	for(int i=1;i<n-1;i++){
	 double d = Z[i+1]-Z[i];
	 cumprob  = Z[i+1]-Z[0];
	 if(cumprob>0.){
		 meanval += d/cumprob;
	 }
	 else{
	   meanval += 1.;
	 }
	}
	
	return meanval;
}


/* Expectation function with general alpha */
double left_fringe_meanval_gen(const int &n, const double *Z, const double &alpha){
	if(n==1){
		return 0.;
	}

	double meanval = 1., cumprob = pow(Z[1]-Z[0],alpha);
	for(int i=1;i<n-1;i++){
	 double d = pow(Z[i+1]-Z[i],alpha);
	 cumprob += d;
	 if(cumprob>0.){
		 meanval += d/cumprob;
	 }
	 else{
	   meanval += 1.;
	 }
	}
		
	return meanval;
}

/* Variance function with alpha=1 */
double left_fringe_var(const int &n, const double *Z){
	if(n==1){
		return 0.;
	}

	double varval = 0.;
	for(int i=1;i<n-1;i++){
	 if((Z[i+1]-Z[0])>0.){
		 double d = (Z[i+1]-Z[i])/(Z[i+1]-Z[0]);
		 varval  += d*(1.-d);
	 }
	 else{
	   varval += 0.25; 
	 }
	}
	
	return varval;
}

/* Variance function with general alpha */
double left_fringe_var_gen(const int &n, const double *Z, const double &alpha){
	if(n==1){
		return 0.;
	}

	double varval = 0., cumprob = pow(Z[1]-Z[0],alpha);
	for(int i=1;i<n-1;i++){
	 double d = pow(Z[i+1]-Z[i],alpha);
	 cumprob += d;
	 if(cumprob>0.){
		 varval += d/cumprob*(1.-d/cumprob);	 
	 }
	 else{
	   varval += pow(0.25,alpha); 
	 }
	}
	
	return varval;
}
