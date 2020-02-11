"""
SCRIPT FOR MULTITASK OUTCOMES WITH BINARY LABELS (0 vs 1/2/3)

i) currently ignoring erosion or ulceration as labels are too imbalanced
"""

import os
from datetime import datetime as dt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skimage import io

def stopifnot(arg, msg):
    import sys
    if not arg:
        sys.exit(msg)

########################################
# ----- STEP 1: LOAD IN THE DATA ----- #
########################################

# Load in the label information
dir_base = os.getcwd()
dir_data = os.path.join(dir_base,'..','data')
df_lbls = pd.read_csv(os.path.join(dir_data,'df_lbls.csv'))
# Extract clinical info
cn_clin = ['lab_dt','sex','age_lab']
df_clin = df_lbls[['ID'] + cn_clin]
df_clin = df_clin[~df_clin.duplicated()].reset_index(drop=True)
u_IDs = df_clin.ID.values
# Remove clinical info from label
df_lbls = df_lbls.drop(columns=cn_clin)

# Folder with the cropped images
dir_cropped = os.path.join(dir_data, 'cropped')
# Get the file size dimensions
dir_images = os.path.join(dir_data, 'cleaned')

#####################################################
# ----- STEP 2: CREATE THE TRAINING/TEST SETS ----- #
#####################################################

# Split train/test set based on year
df_clin.lab_dt = pd.to_datetime(df_clin.lab_dt)
df_clin['year'] = df_clin.lab_dt.dt.strftime('%Y').astype(int)
u_IDs_train = df_clin[df_clin.year <= 2017].ID.values
u_IDs_test = df_clin[df_clin.year > 2017].ID.values
print('Training sample images: %i from %i people\nTesting sample images: %i from %i people' %
      (df_lbls.ID.isin(u_IDs_train).sum(),len(u_IDs_train),
       df_lbls.ID.isin(u_IDs_test).sum(),len(u_IDs_test)))

# Make a subset for iterating over
df_train = df_lbls[df_lbls.ID.isin(u_IDs_train)].reset_index(drop=True).drop(columns={'EOU'})
df_test = df_lbls[df_lbls.ID.isin(u_IDs_test)].reset_index(drop=True).drop(columns={'EOU'})
# Remove rows with only missing data
cn_lbls = ['CII','LPN','NIE']
df_train = df_train[~(df_train[cn_lbls].isnull().mean(axis=1)==1)].reset_index(drop=True)
df_test = df_test[~(df_test[cn_lbls].isnull().mean(axis=1)==1)].reset_index(drop=True)
# # Print the label balance by the training/test
# df_train.NIE.value_counts().reset_index().sort_values('index')
# df_test.NIE.value_counts().reset_index().sort_values('index')

# binaryize labels
df_train[cn_lbls] = np.where(df_train[cn_lbls].isnull(),np.NaN,np.where(df_train[cn_lbls]==0,0,1))
df_test[cn_lbls] = np.where(df_test[cn_lbls].isnull(),np.NaN,np.where(df_test[cn_lbls]==0,0,1))


########################################
# ----- STEP 3: SET UP THE MODEL ----- #
########################################

# 1) Load the binary conv-net model
from models import mdls_torch
torch.manual_seed(12345)
mdl = mdls_torch.CNN_binary(n_tasks = 3)
# np.random.seed(111)
# print(mdl(torch.Tensor(np.random.rand(2,3,501,501))))

# ---- STEP 2: DEFINE LOSS FUNCTION ---- #
loss_fun = nn.BCEWithLogitsLoss(reduction='mean')

# ---- STEP 3: DEFINE THE OPTIMIZER ---- #
optimizer = torch.optim.Adagrad(params=mdl.parameters(),lr=0.003)

#######################################
# ----- STEP 4: TRAIN THE MODEL ----- #
#######################################

n_epochs = 1200
m_batch = 5

tt_epoch = dt.now()
for ii in range(n_epochs):
    print('---------- Epoch %i of %i -----------' % (ii+1, n_epochs))
    tt_start = dt.now()
    # Randomly iterate over the training data
    np.random.seed(ii)
    lst_loss = []
    for jj, rr in df_train.sample(n=df_train.shape[0]).reset_index(drop=True).iterrows():
        # --- (0) Prep the data --- #
        file_jj = rr['file']
        tissue_jj = rr['tissue']
        ID_jj = rr['ID']
        fold_jj = os.path.join(dir_cropped, 'train', ID_jj, tissue_jj)
        stopifnot(os.path.exists(fold_jj), 'Folder does not exist')
        y_jj = np.array(rr[cn_lbls]) # Create the label
        # Load m_batch images
        fn_jj = np.random.choice(os.listdir(fold_jj), m_batch,replace=False)
        fn_jj = [os.path.join(fold_jj, fn) for fn in fn_jj]
        img_jj = io.imread_collection(fn_jj).concatenate().transpose([0,3,2,1])
        img_jj = torch.Tensor(img_jj / 255) # Convert to tensor and normalize

        # Clear gradients
        optimizer.zero_grad()
        # --- (1) FORWARD PASS --- #
        y_jj = torch.Tensor(np.repeat(y_jj,m_batch).astype(float))
        yhat_jj = mdl(img_jj).flatten()
        loss_jj = loss_fun(yhat_jj[~torch.isnan(y_jj)], y_jj[~torch.isnan(y_jj)])
        lst_loss.append(loss_jj.detach().numpy()+0)

        # (2) --- BACKWARD PASS --- #
        loss_jj.backward()
        optimizer.step()
        if jj % 10 == 0:
            print(mdl(img_jj))
            # Print the gradient variances
            se_conv1 = mdl.conv1.weight.grad.std()
            se_conv2 = mdl.conv2.weight.grad.std()
            se_fc1 = mdl.fc1.weight.grad.std()
            se_fc2 = mdl.fc2.weight.grad.std()
            se_fc4 = mdl.fc3.weight.grad.std()
            print('conv1: %0.4f\nconv2: %0.4f\nfc1: %0.4f\nfc2: %0.4f\nfc3: %0.4f' %
                  (se_conv1, se_conv2, se_fc1, se_fc2, se_fc4))
    # Get mean loss
    lst_loss = np.array(lst_loss)
    print('Average logistic loss: %0.3f, with SD: %0.3f' % (lst_loss.mean(),lst_loss.std()))
    tt_end = dt.now(); tt_seconds = (tt_end - tt_start).seconds
    print('Run time: %i seconds' % tt_seconds)

total_seconds = (tt_end - tt_epoch).seconds
print('Total run-time: %0.3f minutes' % (total_seconds / 60))

# Save model at current datetime
fname = 'cnn_binary_' + dt.now().strftime('%Y_%m_%d_%M:%H') + '.pt'
torch.save(mdl.state_dict(),os.path.join(dir_base,'saved_networks',fname))
print('end of script!')

