/* Aggregation functions */

#ifndef AGGREGATION_FUNCTIONS_H__
#define AGGREGATION_FUNCTIONS_H__

#include <iostream>
#include <map>
#include <math.h>

using namespace std;

void normalize_scores(const int &nr, const int &nc, double *scores);

void aggregate_average(const int &nr, const int &nc, double* scores_agg, const double *scores);

void aggregate_maximum(const int &nr, const int &nc, double* scores_agg, const double *scores);

void aggregate_aom(const int &nr, const int &nc, double* scores_agg, const double *scores);

void compute_counts(const int &n, const int* Xd, std::map<int,int> &count_map, const int &stride=1);

#endif // AGGREGATION_FUNCTIONS_H__
