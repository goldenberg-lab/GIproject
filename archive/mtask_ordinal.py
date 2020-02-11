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
dir_data = os.path.join(dir_base,'..','data')
dir_train = os.path.join(dir_data,'cropped','train')
dir_test = os.path.join(dir_data,'cropped','test')

# labels
df_lbls = pd.read_csv(os.path.join(dir_data,'df_lbls_anon.csv'))

# Split train/test
df_train = df_lbls[df_lbls.type == 'train'].reset_index(drop=True)
df_test = df_lbls[df_lbls.type == 'test'].reset_index(drop=True)

########################################
# ----- STEP 3: SET UP THE MODEL ----- #
########################################

# 1) Load the binary conv-net model
from models import mdls_torch
torch.manual_seed(12345)
mdl = mdls_torch.CNN_ordinal()
# Load pre-trained network
pretrained = torch.load(os.path.join(dir_base,'saved_networks','cnn_binary_2019_10_24_48:21.pt'))
mdl_dict = mdl.state_dict()
pretrained_dict = {k: v for k, v in pretrained.items() if k in mdl_dict}
mdl_dict.update(pretrained_dict)
mdl.load_state_dict(mdl_dict)

# ---- STEP 2: DEFINE LOSS FUNCTION ---- #
loss_fun = nn.BCELoss(reduction='mean')

# ---- STEP 3: DEFINE THE OPTIMIZER ---- #
optimizer = torch.optim.Adagrad(params=mdl.parameters(),lr=0.002)

#######################################
# ----- STEP 4: TRAIN THE MODEL ----- #
#######################################

cn_robarts = ["robarts_CII", "robarts_EOU", "robarts_LPN"]
cn_nancy = ["nancy_CII", "nancy_AIC"]

n_epochs = 1500
m_batch = 10

tt_epoch = dt.now()
for ii in range(n_epochs):
    print('---------- Epoch %i of %i -----------' % (ii+1, n_epochs))
    tt_start = dt.now()
    # Randomly iterate over the training data
    np.random.seed(ii)
    lst_loss = []
    # Sample m_batch rows

    for jj, rr in df_train.sample(n=df_train.shape[0]).reset_index(drop=True).iterrows():
        # --- (0) Prep the data --- #
        file_jj = rr['file']
        ID_jj = rr['ID']
        tissue_jj = rr['file'].split('_')[-1].replace('.png','')
        fold_jj = os.path.join(dir_train, ID_jj,tissue_jj)
        stopifnot(os.path.exists(fold_jj), 'Folder does not exist')

        rr[cn_robarts + cn_nancy]

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
    print('Average absolute loss: %0.3f, with SD: %0.3f' % (lst_loss.mean(),lst_loss.std()))
    tt_end = dt.now(); tt_seconds = (tt_end - tt_start).seconds
    print('Run time: %i seconds' % tt_seconds)

total_seconds = (tt_end - tt_epoch).seconds
print('Total run-time: %0.3f minutes' % (total_seconds / 60))

# Save model at current datetime
fname = 'cnn_ordinal_' + dt.now().strftime('%Y_%m_%d_%M:%H') + '.pt'
torch.save(mdl.state_dict(),os.path.join(dir_base,'saved_networks',fname))
print('end of script!')

