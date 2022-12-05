#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/10/22 13:19

@Author: Luis Antonio Souto Arias

@Software: PyCharm
"""

import numpy as np
import matplotlib.pyplot as plt

## Generate data

n     = 1000
nF    = 2
n_out = 20
n_in  = int((n-n_out)/2)

np.random.seed(5)
X  = np.zeros((n,nF))
r   = 0.15*np.sqrt(np.random.random((n_in,)))
th  = 2*np.pi*np.random.random((n_in,))
R   = 0.6*np.sqrt(np.random.random((n_in,)))
thR = 2*np.pi*np.random.random((n_in,))
X[:n_in,0] = r*np.cos(th)
X[:n_in,1] = r*np.sin(th)
X[n_in:2*n_in,0] = 1.5+R*np.cos(thR)
X[n_in:2*n_in,1] = 1.5+R*np.sin(thR)
X[-n_out,0] = 0.2
X[-n_out,1] = -0.2
X[-n_out+1:-n_out+6,0] = 0.7+0.2*np.random.random((5,))
X[-n_out+1:-n_out+6,1] = 0.7+0.2*np.random.random((5,))
X[-n_out+6:,0] = 0.8+0.2*np.random.random((14,))
X[-n_out+6:,1] = -0.3+0.2*np.random.random((14,))

## Plot the data

plt.figure()
plt.plot(X[:,0],X[:,1],"b*")
plt.show()


## Save the data

fdata = open("../synthetic_data/example_2D_clusters.dat","w")

print(n,nF,file=fdata,sep=", ")
for i in range(n):
    print(*X[i,:],file=fdata,sep=", ")

fdata.close()

fout = open("../synthetic_data/example_2D_clusters_outliers.dat","w")

print(n_out,file=fout)
print(*np.arange(n-n_out,n),file=fout)

fout.close()

##
plt.figure()
plt.plot(X[:n-n_out,0],X[:n-n_out,1],"*",color='0.79')
plt.plot(X[n-n_out:,0],X[n-n_out:,1],"rx")
plt.xlim([-0.5,2.2])
plt.ylim([-0.5,2.2])
plt.savefig("../figures/example_2D_clusters_data.eps",format="eps")
plt.show()

xs = np.random.choice(np.arange(n),50,False)
plt.figure()
plt.plot(X[-1,0],X[-1,1],"r^")
plt.text(1.05*X[-1,0],1.05*X[-1,1],"A",fontsize=18)
plt.plot(X[20,0],X[20,1],"ro")
plt.text(-0.1+X[20,0],0.05+X[20,1],"B",fontsize=18)
plt.plot(X[xs,0],X[xs,1],"*",color='0.79')
plt.xlim([-0.5,2.2])
plt.ylim([-0.5,2.2])
plt.savefig("../figures/example_2D_clusters_subsample.eps",format="eps")
plt.show()

dx = np.abs(X[-1]-X[xs]).sum(-1)
dx = np.concatenate([[0],np.sort(dx)])

plt.figure()
plt.plot(dx[1:]/dx[-1],np.zeros_like(dx[1:]),"ko")
plt.plot(dx[0],0,"rx")
plt.xlabel('Sorted distances',fontsize=18)
plt.savefig("../figures/example_2D_clusters_distance1.eps",format="eps")
plt.show()

dx = np.abs(X[20]-X[xs]).sum(-1)
dx = np.concatenate([[0],np.sort(dx)])

plt.figure()
plt.plot(dx[1:]/dx[-1],np.zeros_like(dx[1:]),"ko")
plt.plot(dx[0],0,"rx")
plt.xlabel('Sorted distances',fontsize=18)
plt.savefig("../figures/example_2D_clusters_distance2.eps",format="eps")
plt.show()

##

