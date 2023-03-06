/* Functions for reading and writing files.

   This file contains the following functions:
      - read_data: reads info about the input data set from given file.
         INPUT:
         + X: Data set. Memory will be allocated INSIDE the function, so X should be given as an
                        EMPTY pointer.
         + n: Number of observations. The value of n will be inferred from the input file, so
                                      previous values of n will be overriden.
         + nF: Number of features. The value of nF will be inferred from the input file, so
                                   previous values of nF will be overriden.
         + filename: Name of the input file, with extension (.dat,.csv) included.
                                   


*/

#include "io_module.h"

template <typename T>
void read_data(T* &X, int &n, int &nF, std::string filename){
	T X_temp;
	
	/* Load data */
	std::ifstream fdata(filename);
	fdata>>n;
	fdata.ignore();
	fdata>>nF;
	
	// Make sure pointer is free before allocating memory
	if(X!=NULL)
    delete[] X; 
    
	X = new T[n*nF];
	
	for(int i=0;i<n;i++){
		for(int j=0;j<nF-1;j++){
			fdata>>X_temp;
			X[j+i*nF] = X_temp;
			fdata.ignore();
		}
		fdata>>X_temp;
		X[nF-1+i*nF] = X_temp;
	}
	
	fdata.close();
}

// double and int instantiation
template void read_data(double* &X, int &n, int &nF, std::string filename);
template void read_data(int* &X, int &n, int &nF, std::string filename);
