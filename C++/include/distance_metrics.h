/* Distance metrics */

#ifndef DISTANCE_METRICS_H__
#define DISTANCE_METRICS_H__

#include <math.h>
#include <algorithm>
#include "types_def.h"

using namespace std;

double l2dist(const int &n, const double* x, const double* y, const int* v);

double l1dist(const int &n, const double* x, const double* y, const int* v);

double lmaxdist(const int &n, const double* x, const double* y, const int* v);

double lsqrtdist(const int &n, const double* x, const double *y,  const int* v);

#endif // DISTANCE_METRICS_H__
