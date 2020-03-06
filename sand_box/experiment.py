import multiprocessing as mp
import full_func as ff
import pandas as pd

# todo change parameters in fullfunc before runnign
data_list = list(map(ff.full_func,test_data_list))

test_df = data_list[0][1]['min_max'][0]
def turn_df(point):
    df = pd.concat([pd.DataFrame(i) for i in data_list[0][1]['min_max']]).reset_index(drop = True).T
    def rename(x):
        name_list = {'0':'mean','1':'var','2':'og_var','3':'p_val'}
        clust = str(int(x/4 + 2))
        name = name_list[str(x % 4)]
        return(name+ '_' + clust)
    df = df.rename(rename,axis = 1)
    return(df)

test2 = data_list[0][1]['min_max']

def widen_df(impt):
    out = {k:turn_df(v) for k,v in impt.items()}
    out = [v.assign(mc_type = k) for k,v in out.items()]
    out = pd.concat(out)
    return(out)

def widen2(impt):
    out = {i[0] : widen_df(i[1]) for i in impt}
    out = [v.assign(df_type=k) for k, v in out.items()]
    out = pd.concat(out)
    return(out)

test = widen2(data_list)


[print(i) for i in impt.items()]


