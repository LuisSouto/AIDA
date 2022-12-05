
FILE=$PWD
IND=$FILE/include
OBJ=$FILE/obj

g++ -Wall -O3 -fopenmp -I$IND $OBJ/aggregation_functions.o $OBJ/isolation_formulas.o $OBJ/distance_metrics.o $OBJ/rng_class.o $OBJ/aida_class.o $1 -o $2
