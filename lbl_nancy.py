# Load in all necessary modules
import numpy as np
import pandas as pd
import os
from funs_support import find_dir_GI, stopifnot
import seaborn as sns
from matplotlib import pyplot as plt

# Assign data directory
dir_base = find_dir_GI()
dir_data = os.path.join(dir_base, 'data')
dir_output = os.path.join(dir_base, 'output')

# Define the file names
path_robarts = os.path.join(dir_data,'df_lbls_robarts.csv')
path_nancy = os.path.join(dir_data,'nancy_score_histo.xlsx')

stopifnot(all([os.path.exists(x) for x in  [dir_data, path_nancy, path_robarts]]),'Error! path not found!')

########################################
# ----- STEP 1: LOAD IN THE DATA ----- #
########################################

# df_size = pd.read_csv(os.path.join(dir_data,'df_size.csv'))
# img = df_size.plot.scatter('width','height')
# img.set_title('Raw images size distribution (in pixels)')
# img.figure.savefig(os.path.join(dir_output,'fig_sizes.png'))

df_robarts = pd.read_csv(path_robarts)
df_nancy = pd.read_excel(path_nancy).drop(columns=['ID code','PATH ID.1'])
df_nancy.rename(columns={'PATH ID':'ID'},inplace=True)
df_long = df_nancy.melt(id_vars='ID',var_name='tissue',value_name='score')
# Add three columns
cn_lbls= ['CII','AIC','ULC']
df_long[cn_lbls] = pd.DataFrame([[np.NaN,np.NaN,np.NaN]],index=df_long.index)
# Clean up some values
df_long.score = np.where(df_long.score == 4.0,4,df_long.score)
df_long.score = np.where(df_long.score == 'N/A (too small)',np.NaN,df_long.score)
df_long.score = np.where(df_long.score == '2 (no chronic inflammation)',5,df_long.score)
# Dictionary mapping:
# Grade 0= no chronic or acute inflammatory infiltrate+ ulcerated
# Grade 1= chronic inflammatory infiltrate, no acute infiltrate or ulceration
# Grade 2 = chronic infiltrate + mild acute inflammatory infiltrate + no erosions
# Grade 3= chronic infiltrate + moderate/severe acute infiltrate + no ulceration
# Grade 4= chronic and acute infiltrate and ulcerations
di_score = {0:[0,0,0], 1:[1,0,0], 2:[1,1,0], 3:[1,2,0], 4:[1,2,1], 5:[0,1,0]}
tmp = []
for ii, xx in enumerate(df_long.score):
    if np.isnan(xx):
        tmp.append([np.NaN] * 3)
    else:
        tmp.append(di_score[xx])
# Add on
df_long[cn_lbls] = np.vstack(tmp)

# Subset to valid images
df_merge = df_long[df_long.score.notnull()].merge(df_robarts.drop(columns=['CII','EOU','LPN','NIE']),on=['ID','tissue'],how='inner')
# Change columns order
df_merge = df_merge[['ID','tissue','file','lab_dt','sex','age_lab','score'] + cn_lbls].sort_values('ID').reset_index(drop=True)
df_merge.to_csv(os.path.join(dir_data,'df_lbls_nancy.csv'),index=False)

##########################################################
# ----- STEP 2: PRINT OFF SOME BASIC SUMMARY STATS ----- #
##########################################################

# Correlation between scores
ax = sns.heatmap(df_merge[cn_lbls].corr())
ax.set_ylim(4, 0)
ax.set_title('Correlation by score',size=18)
ax.figure.savefig(fname=os.path.join(dir_output,'nancy_corr_lbl.png'))
plt.close()

# Correlation between labels
tmp_df = df_merge.melt(id_vars=['ID','tissue'],value_vars=cn_lbls,var_name='scores')
tmp_df = tmp_df.pivot_table(index=['ID','scores'],columns='tissue',values='value').reset_index().iloc[:,2:].corr()
tmp_df.index.name=''
tmp_df.columns.name = ''
plt.figure(figsize=(10,8))
ax = sns.heatmap(tmp_df,square=True); ax.set_ylim(6, 0)
ax.set_title('Correlation by tissue',size=18)
ax.figure.savefig(fname=os.path.join(dir_output,'nancy_corr_tissue.png'))
plt.close()

# Tabular frequency of scores
df_merge.insert(0,'type',np.where(df_merge.ID.str.contains('S18'),'test','train'))
df_tab = df_merge.melt(id_vars=['ID','type'],value_vars=cn_lbls,var_name='score').groupby(['type','score','value']).size().reset_index()
df_tab = df_tab.rename(columns={0:'n'}).pivot_table(index=['score','type'],columns='value',values='n').reset_index()
df_tab = df_tab.melt(id_vars=['score','type'],value_name='n')
df_tab.n = np.where(df_tab.n.isnull(),0,df_tab.n).astype(int)
df_tab.rename(columns={'value':'ordinal'},inplace=True)
df_tab = df_tab.sort_values(by=['score','type']).reset_index(drop=True)
tmp = df_tab.groupby(['type','score'])['n'].sum().reset_index().rename(columns={'n':'tot'})
df_tab = df_tab.merge(tmp,how='right',on=['type','score'])
df_tab['share'] = df_tab.n / df_tab.tot
# Aggregate
df_tab_agg = df_tab.groupby(['score','ordinal'])['n'].sum().reset_index()
df_tab_agg['bool'] = np.where(df_tab_agg.ordinal >= 1, 'yes','no')
df_tab_agg = df_tab_agg.merge(df_tab_agg.groupby('score').n.sum().reset_index().rename(columns={'n':'tot'}),on='score',how='left')
df_tab_agg['share'] = df_tab_agg.n / df_tab_agg.tot * 100
df_tab_agg = df_tab_agg.groupby(['score','bool']).share.sum().reset_index()

# plot it
g = sns.catplot(x='bool',y='share',row='score',kind='bar',data=df_tab_agg)
g.set_xlabels(''); g.set_ylabels('Share (%)')
#g.set_titles('Frequency of Yes/No scores for Nancy',size=14)
g.savefig(os.path.join(dir_output,'dist_nancy_lbls.png'))
plt.close()

