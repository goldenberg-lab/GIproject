"""
SCRIPT FOR MULTITASK OUTCOMES WITH BINARY LABELS (0 vs 1/2/3)

i) currently ignoring erosion or ulceration as labels are too imbalanced
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skimage import io
from datetime import datetime as dt

def stopifnot(arg, msg):
    import sys
    if not arg:
        sys.exit(msg)

########################################
# ----- STEP 1: LOAD IN THE DATA ----- #
########################################

# directories
dir_base = os.getcwd()
dir_pretrained = os.path.join(dir_base,'saved_networks')
dir_data = os.path.join(dir_base,'..','data')
dir_cropped = os.path.join(dir_data,'cropped')
dir_train = os.path.join(dir_cropped,'train')
dir_test = os.path.join(dir_cropped,'test')

# Load data
df_data = pd.read_csv(os.path.join(dir_data,'df_lbls_anon.csv'))
df_data.insert(2,'tissue',[x[-1] for x in df_data.file.str.replace('.png','').str.split('_')])
df_data.drop(columns=['file'],inplace=True)
df_train = df_data[df_data.type == 'train'].reset_index(drop=True)
df_test = df_data[df_data.type == 'test'].reset_index(drop=True)

if not os.path.exists(os.path.join(dir_data,'di_ID.npy')):
    di_ID = dict(zip(df_data.ID + '_' + df_data.tissue,[{}]*df_data.shape[0]))
    for ii, rr in df_data.iterrows():
      ID_ii, tissue_ii, type_ii = rr['ID'], rr['tissue'], rr['type']
      path_ii = os.path.join(dir_cropped, type_ii,ID_ii,tissue_ii)
      di_ID[ID_ii + '_' + tissue_ii] = os.listdir(path_ii)
    np.save(os.path.join(dir_data,'di_ID.npy'),di_ID)
    del di_ID


#########################################
# ----- STEP 2: SET  UP THE MODEL ----- #
#########################################

from models import mdls_torch

pretrained = True # whether to load the highest epoch model from before

# STEP 1: INITIALIE CNN #
torch.manual_seed(12345)
mdl = mdls_torch.CNN_ordinal()

if pretrained:
    fn_pre = pd.Series(os.listdir(dir_pretrained))
    fn = fn_pre[fn_pre.str.split('epoch|\\.',expand=True).iloc[:,1].astype(int).idxmax()]
    pretrained = torch.load(os.path.join(dir_pretrained,fn))
    mdl_dict = mdl.state_dict()
    pretrained_dict = {k: v for k, v in pretrained.items() if k in mdl_dict}
    mdl_dict.update(pretrained_dict)
    mdl.load_state_dict(mdl_dict)

# STEP 2: DEFINE LOSS FUNCTION #
loss_fun = nn.BCEWithLogitsLoss(reduction='mean')

# STEP 3: DEFINE THE OPTIMIZER #
optimizer = torch.optim.Adagrad(params=mdl.parameters(),lr=0.001)

# Load dictionary of file directories for training
di_ID = np.load(os.path.join(dir_data,'di_ID.npy'),allow_pickle='TRUE').item()


#######################################
# ----- STEP 3: TRAIN THE MODEL ----- #
#######################################

cn_robarts = ["robarts_CII", "robarts_LPN", "robarts_NIE"]
cn_nancy = ["nancy_CII", "nancy_AIC"]

n_epochs = 5000
m_batch = 10

lst_loss = []
tt_epoch = dt.now()
for ii in range(5000,5000+n_epochs):
    if (ii+1) % 10 == 0:
      print('---------- Epoch %i of %i -----------' % (ii+1, 5000+n_epochs))
    tt_start = dt.now()
    # Randomly iterate over the training data
    np.random.seed(ii)
    # Sample m_batch rows
    df_ii = df_train.sample(m_batch).reset_index(drop=True)
    file_ii = [np.random.choice(di_ID[x+'_'+y],1)[0] for x,y in zip(df_ii.ID,df_ii.tissue)]
    file_ii = [os.path.join(dir_train,x,y,z) for x,y,z in zip(df_ii.ID,df_ii.tissue,file_ii)]
    # Load in images
    img_ii = io.imread_collection(file_ii).concatenate().transpose([0,3,2,1])
    img_ii = torch.Tensor(img_ii / 255)
    # Clear gradients
    optimizer.zero_grad()
    # --- FORWARD PASS --- #
    s_ii = mdl(img_ii)
    lbls_ii = df_ii[cn_robarts + cn_nancy].values
    tmp = []
    for kk in range(s_ii.shape[1]):
        lbls_ii[:,kk]
        y_kk = lbls_ii[:,kk].repeat(m_batch).reshape([m_batch,m_batch]).T > lbls_ii[:,kk].repeat(m_batch).reshape([m_batch,m_batch])
        s_kk = s_ii[:,kk].repeat(m_batch).reshape([m_batch,m_batch]) - s_ii[:,kk].reshape([m_batch,1])
        s_kk = s_kk[torch.Tensor(y_kk) == 1]
        if len(s_kk) > 0:
            tmp.append(s_kk)
    if not np.any(np.array([len(x) for x in tmp]) > 0):
        continue
    s_ii_auc = torch.cat(tmp)
    y_ii = torch.Tensor(np.repeat(1,len(s_ii_auc)))
    loss_ii = loss_fun(s_ii_auc, y_ii)
    lst_loss.append(loss_ii.detach().numpy()+0)

    # --- BACKWARD PASS --- #
    loss_ii.backward()
    optimizer.step()
    if (ii+1) % 10 == 0:
        # Print the gradient variances
        se_conv1 = mdl.conv1.weight.grad.std()
        se_conv2 = mdl.conv2.weight.grad.std()
        se_fc1 = mdl.fc1.weight.grad.std()
        se_fc2 = mdl.fc2.weight.grad.std()
        se_fc3 = mdl.fc3B.weight.grad.std()
        print('Iteration %i\nLoss: %0.3f\nconv1: %0.4f\nconv2: %0.4f\nfc1: %0.4f\nfc2: %0.4f\nfc3: %0.4f' %
              (ii+1,loss_ii.detach().numpy()+0,se_conv1, se_conv2, se_fc1, se_fc2, se_fc3))
    if (ii+1) % 500 == 0:
      print('SAVING ARCHITECTURE')
      out_ii = os.path.join(dir_base,'saved_networks','cnn_conc_epoch' + str(ii+1) + '.pt')
      torch.save(mdl.state_dict(),out_ii)



