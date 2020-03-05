import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statistics import stdev
import random

def random_order(arr, seed):
    np.random.seed(seed)
    new_arr = arr.copy()
    [np.random.shuffle(i) for i in new_arr.T]
    return(new_arr)

def min_max(arr,seed):
    np.random.seed(seed)
    new_arr = [[random.uniform(min(i),max(i)) for j in i] for i in arr.T]
    return(np.array(new_arr).T)


def pca_trans(arr,seed):
    np.random.seed(seed)
    pca = PCA()
    pca = pca.fit(arr)
    eig = pca.components_
    tm = pca.transform(arr)
    gaus_arr = np.empty([0,tm.shape[0]])
    for i in tm.T:
        sd = stdev(i)
        add = [random.gauss(0,sd) for j in i]
        gaus_arr = np.vstack([gaus_arr,add])
    return(np.dot(gaus_arr.T,eig))


test_df = make_classification(n_samples = 100,
                     n_features= 50,
                     n_informative = 48,
                     n_redundant= 2,
                     n_classes = 4,
                     n_clusters_per_class = 1,
                     class_sep= 2)
td2 = test_df[0]
td_rand = random_order(td2,4)
td_minmax = min_max(td2,4)
td_pca = pca_trans(td2,4)


