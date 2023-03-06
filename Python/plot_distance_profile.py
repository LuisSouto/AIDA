#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31/10/22 2:41

@Author: Luis Antonio Souto Arias

@Software: PyCharm
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Load data

ffolder   = '../synthetic_data/'
fname     = 'example_cross_1000_50'
foutliers  = ffolder+fname+'_outliers.dat'
fdata      = ffolder+fname+'_num.dat'
ref_factor = '2_'
fexplain   = '../results/'+fname+'_TIX'+ref_factor
finfo      = ffolder+fname+'_outlier_info.dat'

n_out = pd.read_csv(foutliers,nrows=1,header=None).to_numpy().squeeze()
outls = np.atleast_1d(pd.read_csv(foutliers,skiprows=1,header=None,sep=" ",engine="python").to_numpy().squeeze())
n,nF  = pd.read_csv(fdata,nrows=1,header=None,sep=",").to_numpy().squeeze()
X     = pd.read_csv(fdata,skiprows=1,header=None,sep=", ",engine="python").to_numpy().squeeze()

n_exec = 10
explain_score = np.zeros((n_exec,n_out,nF))

for i in range(n_exec):
    explain_score[i] = pd.read_csv(fexplain+str(i+1)+'.dat',skiprows=1,nrows=n_out+0,header=None,sep=" ",engine="python",usecols=np.arange(1,nF+1)).to_numpy(dtype='float32').squeeze()/nF

explain_score_avg = explain_score.mean(0)
## Distance profile plot

id      = 0
ndim    = min(10,nF)
imp_dim = np.argsort(-explain_score_avg[id])

l  = 1.
dx = np.abs(X[outls[id],imp_dim[:ndim]]-X[:,imp_dim[:ndim]])**l

dist_list = []
plt.figure()
for i in range(ndim):
    dist = np.sort(dx[:,0:i+1].sum(1)**(1./l))
    dist_list.append(dist/dist.max())

plt.boxplot(dist_list,vert=False,patch_artist=True,manage_ticks=False)
plt.xlabel('Distance profile boxplot',fontsize=18)
plt.ylabel('Number of features',fontsize=18)
plt.gca().invert_yaxis()
plt.yticks([2,4,6,8,10])
plt.show()

## Bar plot with feature scores

plt.figure()
plt.bar(np.arange(nF),explain_score_avg[id]-explain_score_avg[id].min())
plt.xlabel('Feature id',fontsize=18)
plt.ylabel('Importance score',fontsize=18)
plt.show()

print('Current outlier id: ',outls[id])
print('10 most relevant dimensions: '+str(imp_dim[:ndim]))

## Find size of minimal subspace (only possible if the features defining the outlier are known)

ns = 1

nsubs = 1                  # Number of outlier subspaces
v     = [[nF-2,nF-1]]      # List of outlier subspaces
ids   = np.array([0])      # Indexes of which subspace belongs each outlier

nids = np.zeros((ns,))
ams  = np.zeros((n_exec,ns))    # Average minimal subspace

for i in range(n_out):
    nids[len(v[ids[i]])-2] += 1

for k in range(n_exec):
    for j in range(n_out):
        imp_dim = np.argsort(-explain_score[k][j])
        ams[k,len(v[ids[j]])-2] += np.where(np.isin(imp_dim,v[ids[j]],assume_unique=True))[0][-1]+1

ams /= nids
print('Average minimal subspaces:',*ams.mean(0))
print('Standard deviation:',*ams.std(0))
##

