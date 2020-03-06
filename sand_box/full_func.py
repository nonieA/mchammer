import numpy as np
import pandas as pd
import random
import test_data_gen as tdg
import test_mc as tmc
import statistics_clust as sc
import datetime as dt

def getscore(df,mc_method,reps = 1000,):
    gen = getattr(tmc, mc_method )
    data_list = [gen(df,i) for i in range(reps)]
    def get_p(mc_list,og_n):
        mc_add = np.append(mc_list,og_n)
        return((list(np.flip(np.sort(mc_add))).index(og_n) + 1)/(len(mc_list) + 1))
    def one_k(data_list,k):
        results = [sc.km_out(i,k) for i in data_list]
        og = sc.km_out(df,k)
        results = np.array(results).T
        methods = ['hubers','norm','tss']
        out_dict = {methods[i]:[np.mean(results[i]),
                                np.var(results[i]),
                                og[i],
                                get_p(results[i],og[i])] for i in range(len(methods))}
        return(out_dict)
    return([one_k(data_list,i) for i in [2,3]])


def full_func(df_in):
    df_lab = df_in[0]
    df = df_in[1]
    method_list = ['min_max','random_order','pca_trans']
    out_dic = {str(i):getscore(df,i,reps = 10) for i in method_list}
    return([df_lab,out_dic])





