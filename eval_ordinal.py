"""
SCRIPT TO EVALUATE BINARY LABELLING MODEL

i) Loads in most recent binary CNN model
ii) Applies random crop to test set images
"""

# Load necessary modules
import os
import torch
import numpy as np
import pandas as pd
from skimage import io
import seaborn as sns
from matplotlib import pyplot as plt

# Parameters
crop = 501 # Size of crop to perform (same size that network can receive)
nsim = 500 # Number of simulations to perform for inference
cn_lbls = ['robarts_CII', 'robarts_LPN','robarts_NIE', 'nancy_CII', 'nancy_AIC']

dir_base = os.getcwd()
dir_output = os.path.join(dir_base,'..','output')
dir_data = os.path.join(dir_base,'..','data')
dir_cropped = os.path.join(dir_data,'cropped')
dir_cleaned = os.path.join(dir_data, 'cleaned')
dir_networks = os.path.join(dir_base,'saved_networks')

from support.acc_funs import auc, auc_ordinal, pairwise_auc
from models import mdls_torch

##########################################
# ----- STEP 1: LOAD DATA AND PREP ----- #
##########################################

# Load the labels
df_data = pd.read_csv(os.path.join(dir_data,'df_lbls_anon.csv'))
df_data.insert(2,'tissue',[x[-1] for x in df_data.file.str.replace('.png','').str.split('_')])
df_data.drop(columns=['file'],inplace=True)

# Columns for dataframe
cn_robarts = ["robarts_CII", "robarts_LPN", "robarts_NIE"]
cn_nancy = ["nancy_CII", "nancy_AIC"]

# Initialize model
torch.manual_seed(12345)
mdl = mdls_torch.CNN_ordinal()
mdl_inf = mdls_torch.CNN_ordinal()
fn_pre = pd.Series(os.listdir(dir_networks))
fn = fn_pre[fn_pre.str.split('epoch|\\.',expand=True).iloc[:,1].astype(int).idxmax()]
mdl_inf.load_state_dict(torch.load(os.path.join(dir_networks,fn)))

# Load the patient dictionary
di_ID = np.load(os.path.join(dir_data,'di_ID.npy'),allow_pickle='TRUE').item()

##########################################################
# ----- STEP 2: LOOP OVER IMAGES AND STORE RESULTS ----- #
##########################################################

fn_score = 'df_ordinal_score.csv'

if not os.path.exists(os.path.join(dir_data,fn_score)):
    tmp = []
    for ii, rr in df_data.iterrows():
      print('Image %i of %i' % (ii+1,df_data.shape[0]))
      files_ii = di_ID[rr['ID'] + '_' + rr['tissue']]
      files_ii = [os.path.join(dir_cropped, rr['type'],rr['ID'], rr['tissue'], x) for x in files_ii]
      img_ii = io.imread_collection(files_ii).concatenate().transpose([0,3,2,1])
      img_ii = torch.Tensor(img_ii / 255)
      score_ii = mdl_inf(img_ii)
      storage_ii = pd.DataFrame(score_ii.detach().numpy(),columns=cn_robarts + cn_nancy)
      storage_ii['ID'] = rr['ID']
      storage_ii['tissue'] = rr['tissue']
      storage_ii = storage_ii.merge(pd.DataFrame(rr).T,on=['ID','tissue'],how='left',suffixes=['_score','_lbl'])
      tmp.append(storage_ii)

    # Merge and save
    df_score = pd.concat(tmp).melt(id_vars=['ID','tissue','type'],var_name='tmp')
    df_score['histo'] = df_score.tmp.str.split('_',expand=True).iloc[:,0]
    df_score['lbl'] = df_score.tmp.str.split('_',expand=True).iloc[:,1]
    df_score['y'] = df_score.tmp.str.split('_',expand=True).iloc[:,2]
    df_score.drop(columns='tmp',inplace=True)
    df_score.value = df_score.value.astype(float)
    df_score.to_csv(os.path.join(dir_data,fn_score),index=False)
else:
    # Load score data
    df_score = pd.read_csv(os.path.join(dir_data,fn_score)).rename(columns={'lbl':'idx','type':'split'})

# Aggregate performance
df_mu_score = df_score.groupby(['ID','tissue','histo','idx','y'])['value'].mean().reset_index()
df_mu_score = df_mu_score.pivot_table(index=['ID','tissue','histo','idx'],columns='y',values='value',aggfunc=lambda x: x).reset_index()
df_mu_score = df_mu_score.merge(df_data[['type','ID','tissue']],on=['ID','tissue'],how='left')
df_mu_score['histo_idx'] = df_mu_score.histo + '_' + df_mu_score.idx
df_mu_score['lbl012'] = np.where(df_mu_score.lbl==3,2,df_mu_score.lbl) # Remove the 3s
df_mu_score = df_mu_score[df_mu_score.lbl.notnull()].reset_index(drop=True)
df_mu_score['lbl012'] = df_mu_score.lbl012.astype(int)
df_mu_score['lbl'] = df_mu_score.lbl.astype(int)

# Save figure of distribution
g = sns.FacetGrid(data=df_mu_score,row='type',col='histo_idx',hue='lbl012',sharex=False,sharey=False)
g.map(sns.distplot,'score',hist=True,rug=True)
g.add_legend()
g.savefig(os.path.join(dir_output,'ordinal_dist012.png'))

#####################################
# ----- STEP 3: TRAIN STACKER ----- #
#####################################

# # Train a multiclass logistic
# cn_X = ['med','se','l25','u75']
# df_wide = df_inf.pivot_table(values=cn_X,index=['ID','tissue','type'],columns='lbl',aggfunc=lambda x: x).reset_index()
# df_wide.columns = pd.Series([x[0]+'_'+x[1] for x in df_wide.columns]).str.replace('_$','')
# df_wide = df_inf[['ID','tissue','lbl','y3']].merge(df_wide).sort_values(['lbl','ID']).reset_index(drop=True)
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
#
# # Loop through and train
# u_lbls = df_wide.lbl.unique()
# di_mdl = dict(zip(u_lbls,[{} for x in u_lbls]))
# for ll in u_lbls:
#     print('Label: %s' % (ll))
#     tmp_df = df_wide[df_wide.lbl == ll]
#     X = tmp_df.loc[:,tmp_df.dtypes=='float'].values
#     y = tmp_df.y3.values
#     X_train, X_test = X[tmp_df['type']=='train'], X[tmp_df['type']=='test']
#     y_train, y_test = y[tmp_df['type']=='train'], y[tmp_df['type']=='test']
#     mdl_pipe = Pipeline(steps=[('normalize', StandardScaler()),('mcl',LogisticRegression())])
#     mdl_pipe.set_params(normalize__copy=False,mcl__penalty='l1', mcl__C=1,
#                         mcl__intercept_scaling=100, mcl__solver='liblinear',mcl__multi_class='ovr')
#     mdl_pipe.fit(X=X_train,y=y_train)
#     # Compare metrics
#     mdl_pipe.predict_proba(X_train)

######################################
# ----- STEP 4: CALCULATE AUCs ----- #
######################################

# Loop through and calculate AUCs
tmp = []
for tt in ['train','test']:
  for hi in df_mu_score.histo_idx.unique():
    tmp_df = df_mu_score[(df_mu_score.type == tt) & (df_mu_score.histo_idx == hi)].copy()
    # Binary AUC, Pairwise AUC, Ordinal AUC
    tmp_df['y01'] = np.where(tmp_df.lbl == 0, 0, 1)
    tmp_auc_bin, tmp_auc_bin = auc(tmp_df.y01.values, tmp_df.score.values), \
                                auc_ordinal(tmp_df.lbl.values, tmp_df.score.values)
    tmp_slice = pd.DataFrame({'type': tt, 'lbl': hi, 'auc_bin': tmp_auc_bin, 'auc_ord': tmp_auc_bin},index=[0])
    tmp_pairwise = pairwise_auc(tmp_df.lbl012.values, tmp_df.score.values).drop(columns='n')
    tmp_pairwise[['y1', 'y0']] = tmp_pairwise[['y1','y0']].astype(int)
    tmp_pairwise['cc'] = 'auc_' + tmp_pairwise.y1.astype(str) + 'v' + tmp_pairwise.y0.astype(str)
    tmp_pairwise = tmp_pairwise.assign(ii=0).pivot('ii', 'cc', 'auc').reset_index().drop(columns='ii')
    tmp_slice = pd.concat([tmp_slice, tmp_pairwise], axis=1)
    tmp.append(tmp_slice)
# Merge
df_res = pd.concat(tmp).melt(id_vars=['type','lbl'],var_name='msr',value_name='auc')
df_res = df_res[df_res.auc.notnull()].reset_index(drop=True)
df_res.type = pd.Categorical(df_res.type,categories=['train','test'])
df_res.msr = df_res.msr.str.replace('auc_','') #pd.Categorical(,categories=['bin','ord','1v0','2v0','2v1'])

g = sns.FacetGrid(data=df_res,col='lbl',hue='type',sharey=False,sharex=False,col_wrap=3)
g.map(plt.scatter, 'msr','auc')
g.add_legend()
g.set_xlabels('Comparison type')
g.set_ylabels('AUC')
g.savefig(os.path.join(dir_output,'ordinal_auc_metrics.png'))


# di_inf = np.load(os.path.join(dir_data,fn_di),allow_pickle=True).item()
#
# holder = []
# for jj, rr in df_lbls.iterrows():
#     ID_jj = rr['QID']
#     tissue_jj = rr['file'].replace('.png','').split('_')[-1]
#     if tissue_jj == 'Splenic':
#         tissue_jj = 'Flexure'
#     lbl_jj = di_inf[ID_jj][tissue_jj]['lbl'].copy() # Extract label
#     score_jj = di_inf[ID_jj][tissue_jj]['score'].copy()
#     df_jj = pd.DataFrame({'ID':ID_jj,'tissue':tissue_jj,'lbl':cn_lbls,'y':lbl_jj,
#            'med':np.quantile(score_jj,q=0.50,axis=0),'l25':np.quantile(score_jj,q=0.25,axis=0),
#             'u75':np.quantile(score_jj, q=0.75, axis=0),'se':np.std(score_jj,axis=0)})
#     holder.append(df_jj)
#
# # Merge and melt
# df_inf = pd.concat(holder).reset_index(drop=True)
# df_inf['type'] = np.where(df_inf.ID.isin(fn_train),'train','test')
# df_inf['y2'] = np.where(df_inf.y == 0, '0','123')
# df_inf['y3'] = np.where(df_inf.y == 0, '0',np.where(df_inf.y == 1,'1','23'))

#############################################################################################################
#############################################################################################################
#############################################################################################################


#############################################################################################################
#############################################################################################################
#############################################################################################################

for tt in ['train','test']:
    for cc in cn_lbls:
        tmp_cc = df_inf[(df_inf.type == tt) & (df_inf.lbl == cc)].copy()
        tmp_cc = tmp_cc[tmp_cc.y.notnull()].reset_index(drop=True)
        print('Set: %s, label: %s, AUC: %0.3f' % (tt, cc,auc(np.where(tmp_cc.y==0,0,1),tmp_cc.yhat.values)))
        print(pairwise_auc(tmp_cc.y.values,tmp_cc.yhat.values))

# Print results for training/test and
g = sns.FacetGrid(df_inf, col='lbl', row='type', hue="y2", palette="Set1")
g = (g.map(sns.distplot, "yhat", hist=True, rug=True))
g.add_legend()
g.savefig(os.path.join(dir_output,'AUC_plot2.png'))
g = sns.FacetGrid(df_inf, col='lbl', row='type', hue="y3", palette="Set1")
g = (g.map(sns.distplot, "yhat", hist=True, rug=True))
g.add_legend()
g.savefig(os.path.join(dir_output,'AUC_plot3.png'))


########################################
# ----- STEP 4: PLOT THE RESULTS ----- #
########################################

# Make training/test folders in the output if they do not exist
for tt in ['train','test']:
    dir_tt = os.path.join(dir_output,tt)
    if os.path.exists(dir_tt):
        import shutil
        shutil.rmtree(dir_tt)
    print('Making %s folder' % tt); os.mkdir(dir_tt)

tmp = []
for jj, rr in df_lbls[df_lbls.type=='test'].iterrows():
    print(jj+1)
    # --- (i) Load the file --- #
    ID_jj = rr['QID']
    file_jj = rr['file2']
    tissue_jj = file_jj.replace('.png','').split('_')[-1]
    path_jj = os.path.join(dir_cleaned, ID_jj,file_jj)
    score_jj = di_inf[ID_jj][tissue_jj]['score'].copy()
    idx_jj = di_inf[ID_jj][tissue_jj]['idx'].copy()
    lbl_jj = di_inf[ID_jj][tissue_jj]['lbl'].copy()
    img_jj = io.imread(path_jj)
    h, w = img_jj.shape[0:2]
    # --- (ii) Normalize score and plot --- #
    type_jj = rr['type']
    cii_ii = score_jj[:, 0].copy()
    sii = cii_ii + 10
    tmp.append(pd.Series({'lbl':lbl_jj[2],'score':sii.mean()}))

    title_jj = 'ID: ' + rr['ID'] + ' (' + tissue_jj + ') - score: ' + str(np.round(sii.mean(), 1)) + ', label:' + str(lbl_jj[0])
    fig, ax = plt.subplots(figsize=(w / 500, h / 500))
    ax.imshow(img_jj)
    im = ax.scatter(x=idx_jj[:, 0], y=idx_jj[:, 1], s=10, c=sii, vmin=-1, vmax=1)
    fig.colorbar(im,orientation='horizontal')
    ax.set_title(title_jj)
    # fig.set_size_inches((w/100,h/100))
    fig.savefig(fname=os.path.join(dir_output, type_jj, file_jj))


df = pd.concat(tmp,axis=1).T
# g = sns.FacetGrid(data=df,col='lbl')
# g.map(sns.distplot,'score',hist=True,rug=True)

