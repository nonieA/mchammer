import pandas as pd
import re

# functions
def tss_change(row):
    if row['index'] == 'tss':
        row[p_columns] = row[p_columns].apply(lambda x: 1-x)
    return(row)

def count_up(df, group_bys, sum_col, col_name, sum = False):
    if sum == False:
        df2 = (df.groupby(group_bys)[sum_col]
               .value_counts()
               .to_frame()
               .rename(columns={sum_col:col_name})
               .reset_index())
    else:
        df2 = (df.groupby(group_bys)[sum_col]
               .sum()
               .to_frame()
               .rename(columns={sum_col: col_name})
               .reset_index())
    return(df2)

def tp_fp(a,b,p):
    if a == b and p == 1:
        return('tp')
    elif a == b and p == 0:
        return('fn')
    elif a != b and p == 0:
        return('tn')
    else:
        return('fp')

def df_sort(df, group, keep_columns):
    def df_split(df, split, group,keep_columns):
        df2 = (df[df[group] == split]
               .rename(columns={'acc':split})
               .drop(group, axis = 1)
               .sort_values(by = keep_columns))
        return(df2)
    df_list = [df_split(df, i,group, keep_columns) for i in df[group].unique().tolist()]
    out_df = df_list[0]
    for i in df_list[1:]:
        out_df = pd.merge(out_df, i, how = 'outer', on = keep_columns)
    return(out_df)


# import data
data = pd.read_csv('out_data2.csv').drop('Unnamed: 0', axis = 1)

p_columns = [i for i in data.columns.tolist() if 'p_val' in i]

#select relevent columns
tp_data = data[['index','mc_type', 'k', 'feat', 'noise', 'sep'] + p_columns]

# invert tss
tp_data = tp_data.apply(tss_change, axis = 1)

# change to yes no
tp_data.loc[:,p_columns] = tp_data[p_columns].apply(lambda x: [0 if i > 0.05 else 1 for i in x.tolist()])

# find total number of correct
tp_data['Sums'] = tp_data[p_columns].sum(axis = 1)

# data sum total identified per type,sep noise and method
tp_out = count_up(tp_data,['mc_type','index','noise','sep'],'Sums','Total')
tp_out = tp_out.drop('Sums', axis = 1)

# sensitivity data frame
id_cols = [i for i in tp_data.drop('Sums', axis = 1).columns.to_list() if i not in p_columns]

tp_sense = tp_data.drop('Sums', axis = 1).pipe(pd.melt, id_vars = id_cols, value_vars = p_columns)
tp_sense[['variable']] = tp_sense['variable'].apply(lambda x: int(re.sub('p_val_','',x)))
tp_sense['acc'] = [tp_fp(tp_sense.iloc[i,2],tp_sense.iloc[i,6],tp_sense.iloc[i,7]) for i in range(len(tp_sense))]
tp_sense_final = count_up(tp_sense,['index','mc_type'], 'acc','Acc')
tp_sense_final = pd.pivot_table(tp_sense_final, index = ['index', 'mc_type'], values= 'Acc', columns= ['acc'], fill_value=0).reset_index()
tp_sense_final['sense'] = tp_sense_final.tp /(tp_sense_final.tp + tp_sense_final.fn)
tp_sense_final.to_csv('./sand_box/out_data/sense_final.csv')

# only correct df
tp_only = df_sort(tp_sense.drop('value', axis = 1), 'variable',id_cols)
tp_only['succs'] = tp_only[[2,3,4,5,6]].apply(lambda x: 0 if any('f' in i for i in x.to_list()) else 1, axis = 1)
tp_only = count_up(tp_only,['index','mc_type','sep','noise'], 'succs','count', sum = True)
tp_only.to_csv('./sand_box/out_data/only.csv')

# total count
tp_count = count_up(tp_data,['index','mc_type','sep','noise'],sum_col='Sums',col_name ='sums',sum = True)
tp_count.to_csv('./sand_box/out_data/count.csv')

# pval count
tp_pval = data.copy()
tp_pval['p_min'] = tp_pval[p_columns].apply(lambda x: re.sub('p_val_','',str(x.idxmin())) if any(i <= 0.05 for i in x) else 0,axis = 1)
tp_pval['cor'] = tp_pval.apply(lambda x: 1 if x['p_min'] == str(x['k']) else 0, axis = 1)
tp_pval = count_up(tp_pval, ['index','mc_type','sep','noise'], sum_col = 'cor', col_name= 'Cor',sum = True)
tp_pval.to_csv('./sand_box/out_data/pval.csv')

# by cluster
tp_clust = df_sort(tp_sense.drop('value', axis = 1), 'variable',id_cols)
tp_clust['succs'] = tp_clust[[2,3,4,5,6]].apply(lambda x: 0 if any('f' in i for i in x.to_list()) else 1, axis = 1)
tp_clust = count_up(tp_clust,['index','mc_type','sep','noise','k'], 'succs','count', sum = True)
tp_clust.to_csv('./sand_box/out_data/clust.csv')


#