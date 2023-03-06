
FILE=$PWD
IND=$FILE/include
OBJ=$(find $FILE/obj/ -name '*.o')


g++ -Wall -O3 -fopenmp -I$IND $OBJ $1 -o $2
