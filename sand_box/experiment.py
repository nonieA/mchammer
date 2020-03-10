import multiprocessing as mp
import full_func as ff
import pandas as pd
import test_data_gen as tdg
import seaborn as sns
import re
from sklearn.cluster import KMeans
from statistics_clust import acc
import numpy as np
from sklearn.datasets import make_classification
# todo define input vars
# todo run
k = [2,4,5]
n = [100]
feat = [10,20]
noise = [0,0.1,0.5]
sep = [0.5,1,3]

exp_list = tdg.many_data(k = k,n=n,feat = feat,noise = noise,sep = sep,seed =4)



data_list = list(map(ff.full_func, exp_list))


def turn_df(point):
    df = pd.concat([pd.DataFrame(i) for i in point]).reset_index(drop = True).T
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
test_final.to_csv('out_data2.csv')


data = pd.read_csv('out_data2.csv').drop('Unnamed: 0', axis = 1)

p_columns = [i for i in data.columns.tolist() if 'p_val' in i]

tp_data = data[['index','mc_type', 'k', 'feat', 'noise', 'sep'] + p_columns]

tp_data.loc[:,p_columns] = tp_data.loc[:,p_columns].apply(lambda x: [0 if i > 0.05 else 1 for i in x.tolist()])

tp_data['Sums'] = tp_data[p_columns].sum(axis = 1)


tp_out = tp_data.groupby(['mc_type','index','noise','sep'])['Sums'].value_counts().to_frame()
tp_out = pd.DataFrame('tp_out')
tp_out = tp_out.rename(columns={'Sums':'Total'}).reset_index()

tp_acc = tp_data.drop('Sums', axis = 1)
tp_acc = pd.melt(tp_acc,id_vars = [i for i in tp_acc.columns.to_list() if i not in p_columns], value_vars = p_columns)
tp_acc[['variable']] = tp_acc['variable'].apply(lambda x: int(re.sub('p_val_','',x)))

def tp_fp(a,b,p):
    if a == b and p == 1:
        return('tp')
    elif a == b and p == 0:
        return('fn')
    elif a != b and p == 0:
        return('tn')
    else:
        return('fp')

def multiply(a,b):
    return(a * b)
tp_acc['acc'] = [tp_fp(tp_acc.iloc[i,2],tp_acc.iloc[i,6],tp_acc.iloc[i,7]) for i in range(len(tp_acc))]
tp_acc2 = tp_acc.groupby(['mc_type','index','noise','sep'])['acc'].value_counts().to_frame()
tp_acc2 = tp_acc2.rename(columns={'acc':'Total'}).reset_index()
tp_small = tp_acc.groupby(['mc_type','index'])['acc'].value_counts().to_frame()
tp_small = tp_small.rename(columns={'acc':'Total'}).reset_index()
tp_small['both'] = tp_small['mc_type'] + tp_small['index']
tp_small = tp_small.pivot_table( index = 'both', columns=  'acc', values = 'Total',fill_value= 0).reset_index()
tp_small['Sense'] = tp_small['tp']/(tp_small['tp'] + tp_small['fn'])
tp_small['Spec'] = tp_small['tn']/(tp_small['fp'] + tp_small['tn'])
tp_sense = pd.melt(tp_small, id_vars = ['both'], value_vars = ['tp','fn','Sense'])
tp_spec = pd.melt(tp_small, id_vars = ['both'], value_vars = ['fp','tn','Spec'])
tp_acc3 = tp_acc2
tp_acc3['both'] = tp_acc
sns.barplot(x= 'both', y = tp_sense[tp_sense.acc != 'Sense']['value'] , hue = 'acc' , data = tp_sense,dodge=False)

tp_pos = tp_acc.groupby(['index','mc_type','k','noise','sep','feat'])['acc'].value_counts().to_frame()
tp_pos = tp_pos.rename(columns={'acc':'Total'}).reset_index()
tp_pos = tp_pos[tp_pos.acc == 'tp']
tp_pos2 = pd.merge(tp_data, tp_pos, how = 'left', on = ['index','mc_type','k','noise','sep','feat'])
tp_pos2['only'] = tp_pos2.apply(lambda x: 1 if x['acc'] == 'tp' and x['Sums'] == 1 else 0, axis = 1)
tp_pos10 = tp_pos2[tp_pos2.feat == 10]
tp_pos20 = tp_pos2[tp_pos2.feat == 20].reset_index()
tp_pos10['only'] = tp_pos10['only'] + tp_pos20['only']
tp_pos_out = tp_pos10.drop(p_columns + ['acc','Total'], axis =1 )
only = [tp_pos10[tp_pos2.k == i]['only'].tolist() for i in tp_pos10.k.unique()]
tp_fin = tp_pos10[tp_pos10.k == 2]
tp_fin['only'] = fin_on

tp_pos10['sum2'] = tp_pos10['Sums'] + tp_pos20['Sums']
Sums = [tp_pos10[tp_pos2.k == i]['Sums'].tolist() for i in tp_pos10.k.unique()]
Sums = np.array(Sums).sum(axis= 0)

tp_fin['Sums'] = Sums
def many_data(k,n,feat,noise,sep,seed):
    np.random.seed(seed)
    data = [['k ' + str(c) + ' n ' + str(num) + ' feat ' + str(f) + ' noise ' + str(noi) + ' sep ' + str(s),
                make_classification(n_samples=num,
                         n_features=f,
                         n_informative=f - int(round(f * noi, 0)),
                         n_redundant=int(round(f * noi, 0)),
                         n_classes=c,
                         n_clusters_per_class=1,
                         class_sep=s)]
            for num in n for f in feat for noi in noise for c in k for s in sep]
    data = [[i[0], i[1][0], i[1][1]] for i in data]
    return(data)

def acc2(y_true,y_pred):
    conf = confusion_matrix(y_true,y_pred)
    conf_list = conf.flatten()
    indx_list = np.flip(np.argsort(conf_list))
    indx_list = [[int(i/len(conf)),i%len(conf)] for i in indx_list]
    new_nm = []
    og_nm = []
    for i in indx_list:
        if i[0] not in new_nm and i[1] not in og_nm:
            new_nm.append(i[0])
            og_nm.append(i[1])
    y_pred2 = [new_nm[og_nm.index(i)] for i in y_pred]
    correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred2[i])/len(y_true)
    return(correct)

exp_list2 = many_data(k = k,n=n,feat = feat,noise = noise,sep = sep,seed =4)

def km_func(df_list):
    name = df_list[0]
    cluster_df = df_list[1]
    y_true = df_list[2]
    k = y_true.max() + 1
    km = KMeans(n_clusters= k, random_state=4, n_init=40)
    km = km.fit(cluster_df)
    y_pred = km.labels_
    acs = acc2(y_true,y_pred)
    return([name, acs])

accs_list = [km_func(i) for i in exp_list2]
test = pd.DataFrame(accs_list, columns=['df_type', 'acru'])

test_split = test.df_type.str.split(' ')
col_names = ['drop1','k','drop2','n','drop3','feat','drop4','noise','drop6','sep']
test_split = pd.DataFrame(test_split.tolist(),columns = col_names)
test_final = pd.concat([test,test_split], axis = 1).drop(['df_type','drop1','drop2','drop3','drop4','drop6'], axis = 1)
tp_pos3 = tp_pos2
test_final = test_final.select_dtypes(include = 'object').astype('float')
test_final['acru'] = test['acru']
tp_pos3 = pd.merge(tp_pos3, test_final,  how = 'left', on = ['k','noise','sep','feat'],copy = True)
tp_pos3 = tp_pos3.drop(['n_x','n_y'] + p_columns, axis = 1)