#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 4/10/22 12:57

@Author: Luis Antonio Souto Arias

@Software: PyCharm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

## Load the data

ffolder    = '../synthetic_data/'
fname      = 'example_cross_1000_50'
fresults   = '../results/'
foutliers  = ffolder+fname+'_outliers.dat'
fdata      = ffolder+fname+'_num.dat'
fLOF       = fresults+fname+'_LOF.dat'
fAVGKNN    = fresults+fname+'_AVGKNN.dat'
version    = np.array(["alpha1_","alpharandom_"])
score_type = np.array(["expectation_","variance_"])
lnorm      = "1_"
aida_types = version.size*score_type.size

n_outliers = pd.read_csv(foutliers,nrows=1,header=None).to_numpy().squeeze()
outliers   = pd.read_csv(foutliers,skiprows=1,header=None,sep=" ",engine="python").to_numpy(dtype='int32').squeeze()

n,nF       = pd.read_csv(fdata,nrows=1,header=None,sep=",").to_numpy().squeeze()
X          = pd.read_csv(fdata,skiprows=1,header=None,sep=", ",engine="python").to_numpy().squeeze()
scores_LOF = pd.read_csv(fLOF,skiprows=1,header=None,sep=" ",engine="python").to_numpy().squeeze()
scores_AVGKNN = pd.read_csv(fAVGKNN,skiprows=1,header=None,sep=" ",engine="python").to_numpy().squeeze()

y           = np.zeros((n,))
y[outliers] = 1
n_exec      = 10

## Plot the ROC curve

auc_AIDA = np.zeros((aida_types,n_exec))
iter = 0
for score_t in score_type:
    for alpha_v in version:
        for i in range(n_exec):
            fAIDA            = fresults+fname+'_AIDA_dist'+lnorm+score_t+alpha_v+str(i+1)+'.dat'
            scores_AIDA      = pd.read_csv(fAIDA,skiprows=1,header=None,sep=" ",engine="python").to_numpy().squeeze()
            auc_AIDA[iter,i] = roc_auc_score(y,scores_AIDA)
        iter += 1

auc_IF   = np.zeros((n_exec,))
auc_INNE = np.zeros((n_exec,))
for i in range(n_exec):
    fIF         = fresults+fname+'_IF_'+str(i+1)+'.dat'
    fINNE       = fresults+fname+'_INNE_'+str(i+1)+'.dat'
    scores_IF   = pd.read_csv(fIF,skiprows=1,header=None,sep=" ",engine="python").to_numpy().squeeze()
    scores_INNE = pd.read_csv(fINNE,skiprows=1,header=None,sep=" ",engine="python").to_numpy().squeeze()
    auc_IF[i]   = roc_auc_score(y,scores_IF)
    auc_INNE[i] = roc_auc_score(y,scores_INNE)

fpr_AIDA,tpr_AIDA,thresholds_AIDA = roc_curve(y,scores_AIDA)
fpr_IF,tpr_IF,thresholds_IF       = roc_curve(y,scores_IF)
fpr_LOF,tpr_LOF,thresholds_LOF    = roc_curve(y,scores_LOF)
fpr_INNE,tpr_INNE,thresholds_INNE = roc_curve(y,scores_INNE)
fpr_AVGKNN,tpr_AVGKNN,thresholds_AVGKNN = roc_curve(y,scores_AVGKNN)

plt.figure()
plt.plot(fpr_AIDA,tpr_AIDA,'b*-')
plt.plot(fpr_IF,tpr_IF,'kx-')
plt.plot(fpr_LOF,tpr_LOF,'r^-')
plt.plot(fpr_INNE,tpr_INNE,'go-')
plt.plot(fpr_AVGKNN,tpr_AVGKNN,'ms-')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(('AIDA','IF','LOF','INNE','AVGKNN'))
plt.show()

auc_LOF    = roc_auc_score(y,scores_LOF)
auc_AVGKNN = roc_auc_score(y,scores_AVGKNN)

AIDAtypes = ['E1','ER','V1','VR']
for i in range(aida_types):
    print('AUC AIDA '+AIDAtypes[i]+':',format(auc_AIDA[i].mean(),"0.3f"),format(auc_AIDA[i].std(),"0.3f"))
print('AUC IF:     ',format(auc_IF.mean(),"0.3f"),format(auc_IF.std(),"0.3f"))
print('AUC INNE:   ',format(auc_INNE.mean(),"0.3f"),format(auc_INNE.std(),"0.3f"))
print('AUC LOF:    ',format(auc_LOF,"0.3f"))
print('AUC AVGKNN: ',format(auc_AVGKNN,"0.3f"))

##

