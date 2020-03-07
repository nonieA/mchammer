import multiprocessing as mp
import full_func as ff
import pandas as pd


# todo define input vars
# todo run

data_list = list(map(ff.full_func,test_data_list))


def turn_df(point):
    df = pd.concat([pd.DataFrame(i) for i in data_list[0][1]['min_max']]).reset_index(drop = True).T
    def rename(x):
        name_list = {'0':'mean','1':'var','2':'og_var','3':'p_val'}
        clust = str(int(x/4 + 2))
        name = name_list[str(x % 4)]
        return(name+ '_' + clust)
    df = df.rename(rename,axis = 1)
    return(df)


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
test = test.reset_index()
test_split = test.df_type.str.split(' ')
col_names = ['drop1','k','drop2','n','drop3','feat','drop4','noise','drop6','sep']
test_split = pd.DataFrame(test_split.tolist(),columns = col_names)
test_final = pd.concat([test,test_split], axis = 1).drop(['df_type','drop1','drop2','drop3','drop4','drop6'], axis = 1)
test_final.to_csv('out_data.csv')

