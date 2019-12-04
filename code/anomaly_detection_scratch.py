# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:44:46 2019

@author: amitabh.gunjan
"""


"""
Code for anomaly detection using PCA.
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

with open('D:/work/anomaly-detection/kdd-data/colNames.txt', 'r') as f:
    col_names = f.readlines()

col_names = [i.split(":")[0] for i in col_names]
col_names.append("class")
kdd_data = pd.read_csv("D:/work/anomaly-detection/kdd-data/kddcup.data_10_percent.csv")
# Filter normal data points

#kdd_data = kdd_data.iloc[:,:-1]
kdd_data.columns = col_names
kdd_data_normal = kdd_data[kdd_data['class']=='normal.']

kdd_data_normal_numeric = kdd_data_normal.select_dtypes(include=['number'])


pca = PCA()
fit = pca.fit(kdd_data_normal_numeric)
np.shape(fit.components_)
fit.transform()
