# AIDA
Implementation of the Analytic Isolation Distance-based Anomaly (AIDA) detection algorithm and the Tempered Isolation-based eXplanation (TIX) algorithm. For a detailed description of the algorithms, please see: https://arxiv.org/abs/2212.02645

The algorithms are implemented in the C++/src folder, with the corresponding headers in C++/include. Main files to test the algorithms are given in the C++/tests folder. We recommend to use aida_example.cpp and tix_example.cpp, respectively.

Python code is also provided to analyze the results produced by AIDA and TIX. These are, respectively, Python/analyze_aida.py and Python/analyze_tix.py.

The format of the input data is the same as the examples provided in the synthetic_data folder.

In order to compile and run the code on Ubuntu (syntax may change for other Linux distributions) go to the C++ folder in a terminal and write:
```
  make
  sh build_AIDA.sh tests/example_aida.cpp bin/example_aida.out
  ./bin/example_aida.out
```

The ```make``` step is only required once, unless the contents of the C++/include and C++/src folders are modified.
