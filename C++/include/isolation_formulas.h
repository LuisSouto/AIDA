/* Theoretical formulas for isolating points in 1D */

#ifndef isolation_formulas_h__
#define isolation_formulas_h__

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace std;

double left_fringe_meanval(const int &n, const double *Z);

double left_fringe_meanval_gen(const int &n, const double *Z, const double &alpha=1.);

double left_fringe_var(const int &n, const double *Z);

double left_fringe_var_gen(const int &n, const double *Z, const double &alpha=1.);

#endif // isolation_formulas_h__
