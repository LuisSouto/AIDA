#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20/10/22 13:17

@Author: Luis Antonio Souto Arias

@Software: PyCharm
"""

import numpy as np
import matplotlib.pyplot as plt

##

nF      = 101
n       = 1000
n_out   = 1
step    = 5
n2      = int(n/2)
v       = -1.+2.*np.arange(n2)/(n2-1)

for i in range(step,nF,step):
    Z = np.zeros((n,i))
    idn = np.arange(n)
    id_outn = np.random.choice(idn,size=n_out,replace=False)
    Z[:,:-2]  = np.random.random((n,i-2))
    Z[:n2,-2] = v
    Z[:n2,-1] = 0.02*(-1+2.*np.random.random((n2,)))
    Z[n2:,-2] = 0.02*(-1+2.*np.random.random((n2,)))
    Z[n2:,-1] = v
    Z[id_outn,-2] = 0.5
    Z[id_outn,-1] = 0.5

    fdata = open("../synthetic_data/example_cross_"+str(n)+"_"+str(i)+"_num.dat","w")
    fout  = open("../synthetic_data/example_cross_"+str(n)+"_"+str(i)+"_outliers.dat","w")
    print(n,i,file=fdata,sep=', ')
    for j in range(n):
        print(*Z[j],file=fdata,sep=', ')
    print(n_out,file=fout)
    print(*id_outn,file=fout)
    fdata.close()
    fout.close()
##

plt.figure()
plt.plot(Z[:,0],Z[:,3],"b*",color='0.7')
plt.plot(Z[id_outn[0],0],Z[id_outn[0],3],"r^")
plt.savefig("../figures/cross_inlier.eps",format="eps")
plt.show()

plt.figure()
plt.plot(Z[:,-2],Z[:,-1],"b*",color='0.7')
plt.plot(Z[id_outn[0],-2],Z[id_outn[0],-1],"r^")
plt.savefig("../figures/cross_outlier.eps",format="eps")
plt.show()


##

