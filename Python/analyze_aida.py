#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/03/23

@Author: Luis Antonio Souto Arias

@Software: PyCharm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

## Load the data

ffolder    = '../synthetic_data/'
fname      = 'example_cross_1000_50'
fresults   = '../results/'
foutliers  = ffolder+fname+'_outliers.dat'
fdata      = ffolder+fname+'_num.dat'

n_outliers = pd.read_csv(foutliers,nrows=1,header=None).to_numpy().squeeze()
outliers   = pd.read_csv(foutliers,skiprows=1,header=None,sep=" ",engine="python").to_numpy(dtype='int32').squeeze()
n,nF       = pd.read_csv(fdata,nrows=1,header=None,sep=",").to_numpy().squeeze()
X          = pd.read_csv(fdata,skiprows=1,header=None,sep=", ",engine="python").to_numpy().squeeze()

fAIDA       = fresults+fname+'_AIDA.dat'
scores_AIDA = pd.read_csv(fAIDA,skiprows=1,header=None,sep=" ",engine="python").to_numpy().squeeze()

## Plot the ROC curve

# Outlier indicator
y           = np.zeros((n,))
y[outliers] = 1

auc_AIDA = roc_auc_score(y,scores_AIDA)
fpr_AIDA,tpr_AIDA,thresholds_AIDA = roc_curve(y,scores_AIDA)

plt.figure()
plt.plot(fpr_AIDA,tpr_AIDA,'b*-')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(['AIDA'])
plt.show()

print('AUC AIDA: ',format(auc_AIDA,"0.3f"))

##

