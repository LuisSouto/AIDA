/* Type definitions */

#ifndef TYPES_DEF_H__
#define TYPES_DEF_H__

#include <vector>

using namespace std;

typedef double (*distance_pointer)(const int &, const double*, const double*, const int*);
typedef std::vector<double>::iterator double_iter;
typedef std::vector<int>::iterator int_iter;

#endif // TYPES_DEF_H__
