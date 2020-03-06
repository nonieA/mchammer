import pandas as pd
import os
import numpy as np
import random
from sklearn.datasets import make_classification
import seaborn as sns
import timeit
import test_data_gen as tdg
from sklearn.cluster import KMeans

test_df = make_classification(n_samples = 100,
                              n_features= 10,
                              n_informative = 8,
                              n_redundant= 2,
                              n_classes = 4,
                              n_clusters_per_class = 1,
                              class_sep= 2)
td2 = test_df[0]
td_rand = random_order(td2,4)
td_minmax = min_max(td2,4)
td_pca = pca_trans(td2,4)
km = KMeans(4)
km = km.fit(td2)

k = [2,4,5]
n = [100]
feat = [10]
noise = [0.05,0.1]

sep = [0.5,1]


test_data_list = tdg.many_data(k = k,n=n,feat = feat,noise = noise,sep = sep,seed =4)
