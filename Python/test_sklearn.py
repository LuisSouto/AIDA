#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/10/22 14:07

@Author: Luis Antonio Souto Arias

@Software: PyCharm
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.inne import INNE
from pyod.models.knn import KNN
from pyod.models.lunar import LUNAR
from pyod.models.deep_svdd import DeepSVDD

## Compute anomaly scores

tname = 'example_cross_1000_30'
name = "../synthetic_data/"+tname
filename = name+"_num.dat"
file_info = name+"_outliers.dat"
n,nF = pd.read_csv(filename,
                   nrows=1,
                   header=None,
                   sep=","
                   ).to_numpy().squeeze()
X = pd.read_csv(filename,
                skiprows=1,
                header=None,
                sep=",",
                engine='python'
                ).to_numpy()
n_outliers = pd.read_csv(file_info,
                         nrows=1,
                         header=None
                         ).to_numpy().squeeze()
outliers   = pd.read_csv(file_info, skiprows=1,
                         header=None,
                         sep=" ",
                         engine="python"
                         ).to_numpy().squeeze()

for i in range(10):
    iForest = IsolationForest(random_state=i,
                              max_samples=256,
                              n_estimators=100
                              ).fit(X)
    scores = -iForest.score_samples(X)
    scores = (scores - scores.mean()) / scores.std()
    fres_IF = open('../results/' + tname + '_IF_' + str(i+1) + '.dat', 'w')
    print(n, file=fres_IF)
    for j in range(n):
        print(scores[j], file=fres_IF)
    fres_IF.close()

    inne = INNE(n_estimators=100, random_state=i, max_samples=8)
    inne.fit(X)
    scores_inne = inne.decision_function(X)
    scores_inne = (scores_inne - scores_inne.mean()) / scores_inne.std()
    fres_inne = open('../results/' + tname + '_INNE_' + str(i+1) + '.dat', 'w')
    print(n, file=fres_inne)
    for j in range(n):
        print(scores_inne[j], file=fres_inne)
    fres_inne.close()

    deep_svdd = DeepSVDD(batch_size=32, random_state=i, epochs=100)
    deep_svdd.fit(X)
    scores_DeepSVDD = deep_svdd.decision_function(X)

    fres_deep_svdd = open('../results/' + tname + '_DeepSVDD_' + str(i+1)
                          + '.dat', 'w')
    print(n,file=fres_deep_svdd)
    for j in range(n):
        print(scores_DeepSVDD[j],file=fres_deep_svdd)
    fres_deep_svdd.close()

    lunar = LUNAR(n_neighbours=5)
    lunar.fit(X)
    scores_LUNAR = lunar.decision_function(X)

    fres_lunar = open('../results/' + tname + '_LUNAR' + str(i+1)
                      + '.dat', 'w')
    print(n, file=fres_lunar)
    for j in range(n):
        print(scores_LUNAR[j], file=fres_lunar)
    fres_lunar.close()

LOF1 = LocalOutlierFactor(p=1, n_neighbors=min(20, int(0.05 * n)))
LOF1.fit(X)
scores_LOF = -LOF1.negative_outlier_factor_

avgknn = KNN(p=1, n_neighbors=min(20, int(0.05 * n)), method='mean')
avgknn.fit(X)
scores_avgknn = avgknn.decision_function(X)

## Save scores in output file
fres_LOF = open('../results/'+tname+'_LOF.dat','w')

print(n,file=fres_LOF)

for i in range(n):
    print(scores_LOF[i],file=fres_LOF)
fres_LOF.close()

fres_avgknn = open('../results/'+tname+'_AVGKNN.dat','w')
print(n,file=fres_avgknn)
for i in range(n):
    print(scores_avgknn[i],file=fres_avgknn)
fres_avgknn.close()

##
