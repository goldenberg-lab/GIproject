import os
import shutil
import numpy as np
import pandas as pd
import string

from funs_support import find_dir_GI, makeifnot

########################################
# ----- STEP 1: LOAD IN THE DATA ----- #
########################################

# directories
dir_base = find_dir_GI()
dir_data = os.path.join(dir_base, 'data')
dir_cleaned = os.path.join(dir_data, 'cleaned')
dir_cropped = os.path.join(dir_data, 'cropped')
assert all([os.path.exists(fold) for fold in [dir_data, dir_cropped, dir_cleaned]])

# data
df_robarts = pd.read_csv(os.path.join(dir_data, 'df_lbls_robarts.csv'))
df_nancy = pd.read_csv(os.path.join(dir_data, 'df_lbls_nancy.csv'))

# Rename columns
cn_nancy = ['CII', 'AIC']
cn_robarts = ['CII', 'LPN', 'NIE']
df_robarts = df_robarts[['ID', 'tissue', 'file'] + cn_robarts]
df_robarts.columns = np.where(df_robarts.columns.isin(cn_robarts), 'robarts_' + df_robarts.columns, df_robarts.columns)
df_nancy = df_nancy[['ID', 'tissue', 'file'] + cn_nancy]
df_nancy.columns = np.where(df_nancy.columns.isin(cn_nancy), 'nancy_' + df_nancy.columns, df_nancy.columns)
# merge
df_merge = df_robarts.merge(df_nancy, on=['ID', 'tissue', 'file'])

#################################
# ----- STEP 2: ANONYMIZE ----- #
#################################

u_ID = df_merge.ID.unique()
slist = list(string.ascii_uppercase) + list(range(10))
seed = 1234
np.random.seed(seed)
q_ID = [''.join(list(np.random.choice(slist, 8))) for x in range(len(u_ID))]
di_ID = dict(zip(u_ID, q_ID))
# Create maps
df_merge.insert(1, 'QID', [di_ID[x] for x in df_merge.ID])
df_merge.insert(4, 'file2', [x[0] + '_' + di_ID[x[1]] + '_' + x[2] for x in df_merge.file.str.split('_')])
# Add train/test type
pat_test = '^S[H]?18'
df_merge.insert(0, 'type', np.where(df_merge.ID.str.contains(pat_test), 'test', 'train'))
# Save anonymized labels
df_anon = df_merge[['type', 'QID', 'file2'] + list(df_merge.columns[df_merge.columns.str.contains('\\_')])]
df_anon.rename(columns={'QID': 'ID', 'file2': 'file'}, inplace=True)
df_anon.to_csv(os.path.join(dir_data, 'df_lbls_anon.csv'), index=False)
print(df_merge.type.value_counts())
print(df_merge.query('ID == "S16-1847"').QID.unique()[0])

# Repeat for the new patients within Nancy/Robarts score
fn_cropped = pd.Series(os.listdir(dir_cropped))
fn_cropped = fn_cropped[fn_cropped.str.contains('\\-')].reset_index(None, True)
u_ID_new = list(np.setdiff1d(fn_cropped, u_ID))
df_new = pd.DataFrame({'ID': u_ID_new})
df_new = df_new.assign(type=lambda x: np.where(x.ID.str.contains(pat_test), 'test', 'train'),
                       tissue='Rectum')
df_new = df_new.assign(QID=lambda x: [''.join(list(np.random.choice(slist, 8))) for x in range(len(x))])
di_ID_new = dict(zip(df_new.ID, df_new.QID))
files_all = sum([os.listdir(os.path.join(dir_cleaned, fold)) for fold in os.listdir(dir_cleaned)], [])
files_new = pd.Series(np.setdiff1d(files_all, df_merge.file))
q1 = pd.DataFrame({'ID': files_new.str.split('\\_|\\.', 2, True).iloc[:, 1], 'file': files_new})
# q2 = pd.DataFrame({'ID':files_new})
df_new = df_new.merge(q1, 'left', 'ID')
df_new.insert(df_new.shape[1], 'file2', [x[0] + '_' + di_ID_new[x[1]] + '_' + x[2] for x in df_new.file.str.split('_')])
df_merge = pd.concat([df_merge, df_new], 0).reset_index(None, True)
print(df_merge.isnull().sum(0))
# Save the code-breaker if needed
cn_code = ['ID', 'tissue', 'type', 'QID', 'file', 'file2']
df_codebreaker = df_merge[cn_code].copy()
df_codebreaker.apply(lambda x: x.duplicated().sum(), 0)
assert np.all(df_codebreaker[['file', 'file2']].apply(lambda x: x.duplicated().sum(), 0) == 0)
df_codebreaker.to_csv(os.path.join(dir_data, 'df_codebreaker.csv'), index=False)

#########################################
# ----- STEP 3: ANONYMIZE FOLDERS ----- #
#########################################

dir_train = os.path.join(dir_cropped, 'train')
makeifnot(dir_train)
dir_test = os.path.join(dir_cropped, 'test')
makeifnot(dir_test)

di_ID_both = {**di_ID, **di_ID_new}

for ii, rr in df_merge.iterrows():
    fold1 = rr['ID']
    fold2 = rr['QID']
    file1 = rr['file']
    file2 = rr['file2']
    val = rr['type']
    path1 = os.path.join(dir_cropped, fold1)
    path2 = os.path.join(dir_cropped, val, fold2)
    if os.path.exists(path1):
        print('Moving %s' % file1)
        shutil.move(path1, path2)
        tissues = os.listdir(path2)
        for tt in tissues:
            path3 = os.path.join(path2, tt)
            fn1 = os.listdir(path3)
            fn2 = [x[0] + '_' + di_ID_both[x[1]] + '_' + x[2] for x in pd.Series(fn1).str.split('_', n=2)]
            for f1, f2 in zip(fn1, fn2):
                shutil.move(os.path.join(path3, f1), os.path.join(path3, f2))
