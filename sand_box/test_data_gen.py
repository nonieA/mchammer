import pandas as pd
import os
import numpy as np
import random
from sklearn.datasets import make_classification
import seaborn as sns
import timeit

# testing quickest method to generate data
k = [2,4,5]
n = [100,500,600,700]
feat = [14,53,10]
noise = [0.3,0.05,0.1]

sep = [0.1,0.5,1]
split =[np.random.uniform(0,1/i,(i-1)) for i in k]

# super nested lists

def many_data(k,n,feat,noise,sep,split,seed):
    np.random.seed(seed)
    data = [make_classification(n_samples=num,
                         n_features=f,
                         n_informative=f - int(round(f * noi, 0)),
                         n_redundant=int(round(f * noi, 0)),
                         n_classes=c,
                         n_clusters_per_class=1,
                         class_sep=s)
            for num in n for f in feat for noi in noise for c in k for s in sep]
    return(data)



