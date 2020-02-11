import os
import shutil
import numpy as np
import pandas as pd

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

# Folder with the cropped images
dir_cropped = os.path.join(dir_data, 'cropped')

# data
df_robarts = pd.read_csv(os.path.join(dir_data,'df_lbls_robarts.csv'))
df_nancy = pd.read_csv(os.path.join(dir_data,'df_lbls_nancy.csv'))

# Rename columns
cn_nancy = ['CII','AIC']
cn_robarts = ['CII','LPN','NIE']
df_robarts = df_robarts[['ID','tissue','file'] + cn_robarts]
df_robarts.columns = np.where(df_robarts.columns.isin(cn_robarts),'robarts_' + df_robarts.columns,df_robarts.columns)
df_nancy = df_nancy[['ID','tissue','file'] + cn_nancy]
df_nancy.columns = np.where(df_nancy.columns.isin(cn_nancy),'nancy_' + df_nancy.columns,df_nancy.columns)
# merge
df_merge = df_robarts.merge(df_nancy,on=['ID','tissue','file'])

#################################
# ----- STEP 2: ANONYMIZE ----- #
#################################

import string

u_ID = df_merge.ID.unique()
slist = list(string.ascii_uppercase) + list(range(10))
np.random.seed(1234)
q_ID = [''.join(list(np.random.choice(slist,8))) for x in range(len(u_ID))]
di_ID = dict(zip(u_ID,q_ID))
# Create maps
df_merge.insert(1,'QID',[di_ID[x] for x in df_merge.ID])
df_merge.insert(4,'file2',[x[0] + '_' + di_ID[x[1]] + '_' + x[2] for x in df_merge.file.str.split('_')])
# Add train/test type
df_merge.insert(0,'type',np.where(df_merge.ID.str.contains('S18'),'test','train'))
# Save anonymized labels
df_anon = df_merge[['type','QID','file2'] + list(df_merge.columns[df_merge.columns.str.contains('\\_')])]
df_anon.rename(columns={'QID':'ID','file2':'file'},inplace=True)
df_anon.to_csv(os.path.join(dir_data,'df_lbls_anon.csv'),index=False)
# Save the code-breaker if needed
df_merge[['ID','QID','file','file2']].to_csv(os.path.join(dir_data,'df_codebreaker.csv'),index=False)


#########################################
# ----- STEP 3: ANONYMIZE FOLDERS ----- #
#########################################

dir_train = os.path.join(dir_cropped, 'train')
dir_test = os.path.join(dir_cropped, 'test')
if not os.path.exists(dir_train):
    print('making training folder'); os.mkdir(dir_train)
else:
    print('training folder already exists')
if not os.path.exists(dir_test):
    print('making test folder'); os.mkdir(dir_test)
else:
    print('test folder already exists')

for ii, rr in df_merge.iterrows():
   fold1 = rr['ID']
   fold2 = rr['QID']
   file1 = rr['file']
   file2 = rr['file2']
   val = rr['type']
   path1 = os.path.join(dir_cropped, fold1)
   path2 = os.path.join(dir_cropped, val, fold2)
   if os.path.exists(path1):
       print('Moving %s' %(file1))
       shutil.move(path1, path2)
       tissues = os.listdir(path2)
       for tt in tissues:
           path3 = os.path.join(path2,tt)
           fn1 = os.listdir(path3)
           fn2 = [x[0] + '_' + di_ID[x[1]] + '_' + x[2] for x in pd.Series(fn1).str.split('_', n=2)]
           for f1, f2 in zip(fn1,fn2):
               shutil.move(os.path.join(path3,f1), os.path.join(path3,f2))
