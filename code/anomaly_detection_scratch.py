# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:44:46 2019

@author: amitabh.gunjan
"""


"""
Code for anomaly detection using PCA.

	3) Perform PCA over the data.  
	4) Use the anomaly detection logic to get the anomalies after doing PCA.
    5) Create a function to get the metrics related to the classification.

"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

with open('D:/other/Programming/python/anomaly-detection-using-pca/kdd-data/colNames.txt', 'r') as f:
    col_names = f.readlines()
col_names = [i.split(":")[0] for i in col_names]
col_names.append("class")
kdd_data = pd.read_csv("D:/other/Programming/python/anomaly-detection-using-pca/kdd-data/kddcup.data_10_percent.csv")

# Filter normal data points
kdd_data.columns = col_names
kdd_data_normal = kdd_data[kdd_data['class']=='normal.']
kdd_data_normal_numeric = kdd_data_normal.select_dtypes(include=['number'])
 
X_train, X_test = train_test_split(kdd_data_normal_numeric, test_size=0.33, random_state=42)
pca = PCA()
fit = pca.fit(X_train)
fit.explained_variance_
fit.components_
def get_empirical_distribution(X_train):
    """
    Get the empirical quantiles of the sum of ratio of principal component scores 
    and the eigenvalues using the observations from the training dataset.
    From the paper - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.66.299&rep=rep1&type=pdf
    """      

    evals = fit.explained_variance_
    ss_standardized_prcomp_scores = []
    fit_transform = fit.transform(X_train)
    for i in fit_transform:
        prcomp_obs_sq = np.square(i)
        ss_prcomp_scores = np.sum((np.divide(prcomp_obs_sq, evals)))
        ss_standardized_prcomp_scores.append(ss_prcomp_scores)  
        
    return ss_standardized_prcomp_scores

ecdf = get_empirical_distribution(X_train)

def get_quantiles_from_ecdf(ss_prcomp_scores, val, num_major_comps):
    """
    Compare an observation's sum of ratio of principal component scores 
    and the eigenvalues using the observations from the training dataset to the empirical quantiles.
    """
    major_comp_scores = ss_prcomp_scores[0:num_major_comps]
    minor_comp_scores = ss_prcomp_scores[num_major_comps:]
    ecdf_major = ECDF(major_comp_scores)
    quantile_major = ecdf_major(val)
    ecdf_minor = ECDF(minor_comp_scores)
    quantile_minor = ecdf_minor(val)
    return quantile_major, quantile_minor

# Get the non-normal instances. Perform classification over these.
# Bucket all different kinds of non-normal observations into one.
kdd_data_non_normal = kdd_data[kdd_data['class'] !='normal.']
kdd_data_non_normal_numeric = kdd_data_non_normal.select_dtypes(include=['number'])
Y = kdd_data_non_normal['class']
X_train_nn, X_test_nn, Y_train_nn, Y_test_nn = train_test_split(kdd_data_non_normal_numeric, Y, test_size=0.33, random_state=42)
prcomp_scores = get_empirical_distribution(X_test_nn)
quantile_major, quantile_minor = get_quantiles_from_ecdf(ss_prcomp_scores=ecdf, val=prcomp_scores, num_major_comps=5)

anomalies_major = [True if i > 0.9899 else False for i in quantile_major]
anomalies_minor = [True if i > 0.9899 else False for i in quantile_minor]

anomalies_ = []
for i in range(len(anomalies_major)):
    if anomalies_major[i] == True:
        anomalies_.append(True)
    elif anomalies_minor[i] == True:
        anomalies_.append(True)
    else:
        anomalies_.append(False)

Y_true = [True for i in Y_test_nn]
confusion_matrix(Y_true, anomalies_)
precision_score(Y_true, anomalies_)
recall_score(Y_true, anomalies_)



