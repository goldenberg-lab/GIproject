"""
SCRIPT TO EVALUATE BINARY LABELLING MODEL

i) Loads in most recent binary CNN model
ii) Applies random crop to test set images
"""

# Load necessary modules
import gc
import os
import torch
from models import mdls_torch
import numpy as np
import pandas as pd
import seaborn as sns
from skimage import io
from matplotlib import pyplot as plt

# Parameters
crop = 501 # Size of crop to perform (same size that network can receive)
nsim = 500 # Number of simulations to perform for inference
# lbl_drop = ['EOU']
# cn_lbls = ['CII','LPN','NIE']

dir_base = os.getcwd()
dir_output = os.path.join(dir_base,'..','output')
dir_data = os.path.join(dir_base,'..','data')
dir_cropped = os.path.join(dir_data,'cropped')
dir_cleaned = os.path.join(dir_data, 'cleaned')
dir_networks = os.path.join(dir_base,'saved_networks')

# Function to calculate the AUC
def auc_score(y,score):
    s1 = score[np.where(y == 1)[0]]
    s0 = score[np.where(y == 0)[0]]
    den = np.sum(y == 1) * np.sum(y == 0)
    num = 0
    for ss in s1:
        num += np.sum(ss > s0)
    auc = num / den
    return(auc)


# Function to perform a random crop
def random_crop(img,height,width,crop_size,ss):
    np.random.seed(ss)
    yidx = np.random.choice(np.arange(height-crop_size))
    xidx = np.random.choice(np.arange(width-crop_size))
    cropped = img[yidx:(yidx+crop_size+1),xidx:(xidx+crop_size)+1].copy()
    xmu = int(xidx+crop_size / 2 )
    ymu = int(yidx + crop_size / 2)
    return(xmu, ymu, cropped)

##########################################
# ----- STEP 1: LOAD DATA AND PREP ----- #
##########################################

# Get the test set names
fn_test = os.listdir(os.path.join(dir_cropped,'test'))
fn_train = os.listdir(os.path.join(dir_cropped,'train'))

# Load the labels
df_lbls = pd.read_csv(os.path.join(dir_data,'df_lbls.csv')).drop(columns=lbl_drop)
# Label train/test
df_lbls.insert(0,'type',np.where(df_lbls.ID.isin(fn_train),'train',np.where(df_lbls.ID.isin(fn_test),'test','error')))
# Remove missing only rows
df_lbls = df_lbls[~(df_lbls[cn_lbls].isnull().mean(axis=1)==1)].reset_index(drop=True)

# Load the Torch model (model recent one)
lst_networks = pd.Series(os.listdir(dir_networks))
# binary models only (no ordinal)
lst_networks = lst_networks[lst_networks.str.contains('binary')].reset_index(drop=True)
tt_networks = pd.to_datetime(lst_networks.str.replace('cnn_binary_|.pt',''),format='%Y_%m_%d_%M:%H')
fn_network = lst_networks[tt_networks.idxmax()]
print(fn_network)

# Initialize
mdl = mdls_torch.CNN_binary(n_tasks = 3)
mdl.load_state_dict(torch.load(os.path.join(dir_networks,fn_network)))
print(mdl(torch.Tensor(np.random.rand(2,3,crop,crop)))) # Check that network works

##########################################################
# ----- STEP 2: LOOP OVER IMAGES AND STORE RESULTS ----- #
##########################################################

# df_inference = df_lbls[['ID','tissue'] + cn_lbls].copy()
# df_inference = df_inference.merge(df_inference.rename(columns=dict(zip(cn_lbls,[x+'_true' for x in cn_lbls]))),
#                                   on=['ID','tissue'])
# df_inference.rename(columns=dict(zip(cn_lbls,[x+'_pred' for x in cn_lbls])),inplace=True)

# # Dictionary to hold results
# di_inf = dict(zip(fn_train+fn_test, [{} for x in range(len(fn_train)+len(fn_test))]))
#
# for jj, rr in df_lbls.iterrows():
#     # --- (i) Load the file --- #
#     tissue_jj = rr['tissue']
#     ID_jj = rr['ID']
#     fold_jj = os.path.join(dir_cleaned, ID_jj)
#     fn_jj = pd.Series(os.listdir(fold_jj))
#     file_jj = fn_jj[fn_jj.str.contains(tissue_jj)].to_list()[0]
#     print('ID: %s (%i of %i)\nTissue: %s, file: %s' %
#           (ID_jj, jj + 1, df_lbls.shape[0],tissue_jj, file_jj))
#     path_jj = os.path.join(fold_jj, file_jj)
#     # Store the labels
#     di_inf[ID_jj][tissue_jj] = {}
#     di_inf[ID_jj][tissue_jj]['lbl'] = rr[cn_lbls].values
#     # Load fill colour image
#     img_jj = io.imread(path_jj)
#     height, width = img_jj.shape[0:2]
#
#     # --- (ii) Perform the inference --- #
#     np.random.seed(jj)
#     score_jj = np.ones([nsim,len(cn_lbls)]) * np.NaN
#     idx_jj = np.ones([nsim,2],dtype=int)
#     kk = 0
#     ii = 0
#     while ii < nsim:
#         kk += 1
#         ymu, xmu, img_kk = random_crop(img_jj,height,width,crop-1,kk)
#         img_kk = img_kk / 255
#         if img_kk.mean() < 0.95:
#             img_kk = torch.Tensor(img_kk.T.reshape([1,3,crop,crop]))
#             score_jj[ii,:] = mdl(img_kk).detach().numpy().flatten()
#             idx_jj[ii,:] = ymu, xmu
#             ii += 1
#     print('Row %i: %i samples needed for %i hits' % (jj+1,kk,nsim))
#     # Store scores and loc
#     di_inf[ID_jj][tissue_jj]['score'] = score_jj.copy()
#     di_inf[ID_jj][tissue_jj]['idx'] = idx_jj.copy()
#     gc.collect()
#
# # Save dictionary for later
# np.save(os.path.join(dir_data,'di_inf.npy'),di_inf)

############################################
# ----- STEP 3: EVALUATE PERFORMANCE ----- #
############################################

di_inf = np.load(os.path.join(dir_data,'di_inf.npy'),allow_pickle=True).item()

holder = []
for jj, rr in df_lbls.iterrows():
    tissue_jj = rr['tissue']
    ID_jj = rr['ID']
    lbl_jj = di_inf[ID_jj][tissue_jj]['lbl'].copy() # Extract label
    score_jj = di_inf[ID_jj][tissue_jj]['score'].copy()
    # Get max score
    df_jj = pd.DataFrame({'ID':ID_jj,'tissue':tissue_jj,'lbl':cn_lbls,'yhat':np.quantile(score_jj,q=0.65,axis=0),'y':lbl_jj})
    holder.append(df_jj)
# Merge and melt
df_inf = pd.concat(holder).reset_index(drop=True)
df_inf['type'] = np.where(df_inf.ID.isin(fn_train),'train','test')
df_inf['y2'] = np.where(df_inf.y == 0, '0','123')
df_inf['y3'] = np.where(df_inf.y == 0, '0',np.where(df_inf.y == 1,'1','23'))

for tt in ['train','test']:
    for cc in cn_lbls:
        tmp_cc = df_inf[(df_inf.type == tt) & (df_inf.lbl == cc)].copy()
        tmp_cc = tmp_cc[tmp_cc.y.notnull()].reset_index(drop=True)
        print('Set: %s, label: %s, AUC: %0.3f' % (tt, cc,auc_score(np.where(tmp_cc.y==0,0,1),tmp_cc.yhat.values)))


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
    if not os.path.exists(os.path.join(dir_output,tt)):
        print('Making %s folder' % tt)
        os.mkdir(os.path.join(dir_output,tt))


for jj, rr in df_lbls.iterrows():
    print(jj+1)
    # --- (i) Load the file --- #
    tissue_jj = rr['tissue']
    ID_jj = rr['ID']
    fold_jj = os.path.join(dir_cleaned, ID_jj)
    fn_jj = pd.Series(os.listdir(fold_jj))
    file_jj = fn_jj[fn_jj.str.contains(tissue_jj)].to_list()[0]
    path_jj = os.path.join(fold_jj, file_jj)
    score_jj = di_inf[ID_jj][tissue_jj]['score'].copy()
    idx_jj = di_inf[ID_jj][tissue_jj]['idx'].copy()
    lbl_jj = di_inf[ID_jj][tissue_jj]['lbl'].copy()
    img_jj = io.imread(path_jj)
    h, w = img_jj.shape[0:2]
    # --- (ii) Normalize score and plot --- #
    title_jj = 'ID: ' + ID_jj + ' (' + tissue_jj + ') - score: ' + str(lbl_jj[0])
    file_jj = ID_jj+'_'+tissue_jj+'.png'
    type_jj = str(np.where(ID_jj in fn_train,'train','test'))
    cii_ii = score_jj[:, 0].copy()
    cii_ii = (cii_ii - cii_ii.min()) / (cii_ii.max() - cii_ii.min())
    sii = 1/np.exp(-3*cii_ii)
    sii = np.round(sii * (20 / sii.max())).astype(int)
    fig, ax = plt.subplots() #figsize=(w / 10, h / 10)
    ax.imshow(img_jj)
    ax.scatter(x=idx_jj[:, 0], y=idx_jj[:, 1], s=sii, c=sii)
    ax.set_title(title_jj)
    # fig.set_size_inches((w/100,h/100))
    fig.savefig(fname=os.path.join(dir_output, type_jj, file_jj))

