/* Header file for io_module.cpp */

#ifndef IO_MODULE_H__
#define IO_MODULE_H__

#include <iostream>
#include <fstream>
#include <string>

template <typename T>
void read_data(T* &X, int &n, int &nF, std::string filename);

#endif // IO_MODULE_H__
