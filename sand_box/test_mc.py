import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statistics import stdev

def random_order(arr, seed):
    np.random.seed(seed)
    new_arr = arr.copy()
    for i in new_arr.T:
        np.random.shuffle(i)
    return(new_arr)

def min_max(arr,seed):
    np.random.seed(seed)
    new_arr = np.empty((0,arr.shape[0]))
    for i in arr.T:
        add = [random.uniform(min(i),max(i)) for j in i]
        new_arr = np.vstack([new_arr,add])
    return(new_arr.T)

def uniform(arr,seed):
    np.random.seed(seed)
    new_arr = np.empty((0,arr.shape[0]))
    for i in arr.T:
        add = np.linspace(min(i),max(i),len(i))
        new_arr = np.vstack([new_arr, add])
    return(new_arr.T)

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



