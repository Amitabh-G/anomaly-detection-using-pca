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
fit_ = fit.transform(X_test)
fit_[0]

