#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/03/23 2:41

@Author: Luis Antonio Souto Arias

@Software: PyCharm
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Load data

ffolder   = '../synthetic_data/'
fname     = 'example_cross_1000_50'
foutliers = ffolder+fname+'_outliers.dat'
fdata     = ffolder+fname+'_num.dat'
fexplain  = '../results/'+fname+'_TIX.dat'
finfo     = ffolder+fname+'_outlier_info.dat'

n_out = pd.read_csv(foutliers,nrows=1,header=None).to_numpy().squeeze()
outls = np.atleast_1d(pd.read_csv(foutliers,skiprows=1,header=None,sep=" ",engine="python").to_numpy().squeeze())
n,nF  = pd.read_csv(fdata,nrows=1,header=None,sep=",").to_numpy().squeeze()
X     = pd.read_csv(fdata,skiprows=1,header=None,sep=", ",engine="python").to_numpy().squeeze()

explain_score = pd.read_csv(fexplain,skiprows=1,header=None,sep=" ",engine="python",usecols=np.arange(1,nF+1)).to_numpy(dtype='float32').squeeze()/nF
explain_score = np.reshape(explain_score,(n_out,nF))

## Bar plot with feature scores
id      = 0            # Index of the outlier to be analyzed. It should be between 0 and n_out-1.
ndim    = min(10,nF)   # Number of features to use in the DPP
imp_dim = np.argsort(-explain_score[id])

plt.figure()
plt.bar(np.arange(nF),explain_score[id]-explain_score[id].min())
plt.xlabel('Feature id',fontsize=18)
plt.ylabel('Importance score',fontsize=18)
plt.show()

print('Current outlier id: ',outls[id])
print('10 most relevant dimensions: '+str(imp_dim[:ndim]))

## Distance profile plot
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

##

