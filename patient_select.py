"""
SCRIPT TO SELECT 43 RANDOM CROPS FROM 23 PATIENTS FOR THE CELL ANNOTATOR
"""

# Load in all necessary modules
import numpy as np
import pandas as pd
import os
import shutil
from funs_support import makeifnot

# Assign data directory
dir_base = os.getcwd()
dir_data = os.path.join(dir_base, '..', 'data')
dir_cell = os.path.join(dir_data, 'cell_counter')

assert os.path.exists(dir_data)
makeifnot(dir_cell)


#################################################################
# -------------------------- JAZZ 36 -------------------------- #
#################################################################

# Load excel file
dat = pd.read_excel(os.path.join(dir_data, 'UC_TO_all_histo.xlsx'))
raw = dat.copy()
dat.rename(columns={'subject_id': 'id', 'gender_n': 'sex',
                    'Final_Eth_Label': 'ethnicity','MRN No#': 'MRN'}, inplace=True)
dat.replace({'sex': {1: 'M', 2: 'F'}}, inplace=True)  # Recode sex
dat['yr'] = dat.dt_dx.dt.strftime('%Y').astype(int)

n0 = dat.shape[0]
# (i) Keep only extensive disease
dat = dat[dat.DisPheno_max.isin(['E4'])]
n1 = dat.shape[0]
# (ii) Moderate/severe Mayo score
dat = dat[dat.BL_Mayo >= 2]
n2 = dat.shape[0]
# (iii) severe clinical disease index
dat = dat[dat.PUCAI_scr > 55]
n3 = dat.shape[0]
# (iv) Physician assessment
dat = dat[dat.PGA.isin(['Moderate', 'Severe'])]
n4 = dat.shape[0]
# Remove two white caucasian males to get to 50 patients
dat = dat[~((dat.ethnicity == 'Caucasian') & (dat.sex == 'M') & (dat.BL_Mayo == 2) & (dat.yr == 2014))]
n5 = dat.shape[0]
# Reset
dat.reset_index(drop=True, inplace=True)

print('Number of patients: %i\nAfter DisPheno: %i\nAfter MayoScore: %i\n'
      'After PUCAI: %i\nAfter PGA: %i' % (n0, n1, n2, n3, n4))

# Summary statistics
print('--- age at diagnosis (months) ---')
print(np.round((dat.Age_Dx_mths / 12).describe()))
print('--- sex ---')
print(dat.sex.value_counts())
print('--- ethnicity ---')
print(dat.ethnicity.value_counts())
print('--- year ---')
print(dat.yr.value_counts())
print('--- PGA ---')
print(dat.PGA.value_counts())
print('--- BL_Mayo ---')
print(dat.BL_Mayo.value_counts())

# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.distplot(dat.yr)
# sns.distplot(dat.Age_Dx_mths)

print(dat.MRN.duplicated().any())
print(dat.MRN.astype(str).str.cat(sep=', '))

#################################################################
# -------------------------- CROP 50 -------------------------- #
#################################################################

# df_robarts = pd.read_csv(os.path.join(dir_data, 'df_lbls_robarts.csv'))
# df_nancy = pd.read_csv(os.path.join(dir_data, 'df_lbls_nancy.csv'))
df_code = pd.read_csv(os.path.join(dir_data, 'df_codebreaker.csv'))
#
# # Average across scores for robarts
# df_robarts['score'] = df_robarts[['CII', 'EOU', 'LPN', 'NIE']].apply(np.nanmean, axis=1)
# df_robarts = df_robarts[df_robarts.score.notnull()].reset_index(drop=True)
#
# # Select patients from each unique score
# u_score = df_robarts.score.unique()
# n_patients = 50
# pps = n_patients // len(u_score)  # patients per score
#
# np.random.seed(1234)
# holder = []
# for ss in u_score:
#     tmp = df_robarts[df_robarts.score == ss]
#     if tmp.shape[0] > pps:
#         holder.append(tmp.sample(n=pps))
#     else:
#         holder.append(tmp)
# df_sub = pd.concat(holder).reset_index(drop=True)

demo = pd.read_csv(os.path.join(dir_data, 'df_lbls_nancy.csv')).drop(columns=['CII','AIC','ULC'])
nancy = pd.read_excel(os.path.join(dir_data, 'nancy_score_histo.xlsx'))
nancy = nancy.drop(columns=['ID code','PATH ID.1']).rename(columns={'PATH ID':'ID'}).melt('ID',None,'tissue','score')
nancy['score'] = nancy.score.astype(str).str.replace('\\.[0-9]','').str.replace('[^0-9]','')
nancy['score'] = np.where(nancy.score.str.len()==0,np.NaN, nancy.score).astype(float)
nancy = nancy[nancy.score.notnull()].assign(score = lambda x: x.score.astype(int)).reset_index(None, True)
demo = demo.merge(nancy,'left',['ID','tissue'])
print(demo.groupby(['score_x','score_y']).size().reset_index())
demo = demo.drop(columns=['score_x']).rename(columns={'score_y':'score'})
df_sub = demo[demo.score >= 2].reset_index()
print(df_sub.score.value_counts())

# # Inflammation
# dat_inf = df_sub.melt('file', ['CII', 'LPN', 'NIE', 'EOU']).groupby(['variable', 'value']).size().reset_index().rename(
#     columns={0: 'n', 'value': 'lbl'})
# dat_inf.lbl = dat_inf.lbl.astype(int)
# print(dat_inf)
# Distribution of metric
print('A total of %i uniquie patients out of %i' % (df_sub.ID.unique().shape[0], demo.ID.unique().shape[0]))
print(df_sub.ID.value_counts().reset_index())
# Sex
print(df_sub.sex.value_counts(normalize=True))
# Age range
np.round((df_sub.age_lab / 365).describe()).astype(int)

old_images = ['49TJHRED_Descending', '49TJHRED_Rectum', '6EAWUIY4_Cecum', '6EAWUIY4_Rectum', '8HDFP8K2_Transverse', '8ZYY45X6_Ascending', '8ZYY45X6_Descending', '8ZYY45X6_Sigmoid', '9U0ZXCBZ_Cecum', 'BCN3OLB3_Descending', 'BCN3OLB3_Transverse', 'BLROH2RX_Ascending', 'BLROH2RX_Cecum', 'E9T0C977_Sigmoid', 'ESZOXUA8_Ascending', 'J6QR55KL_Descending', 'J6QR55KL_Rectum', 'J6QR55KL_Sigmoid', 'J6QR55KL_Transverse', 'LALQDCTM_Descending', 'LALQDCTM_Rectum', 'LALQDCTM_Sigmoid', 'MARQQRM5_Rectum', 'MARQQRM5_Transverse', 'MM6IXZVW_Cecum', 'MM6IXZVW_Rectum', 'MM6IXZVW_Transverse', 'N6MF55ZU_Rectum', 'PZUZFPUN_Rectum', 'QF0TMM7V_Rectum', 'QF0TMM7V_Sigmoid', 'QIGW0TSV_Transverse', 'R9I7FYRB_Transverse', 'RADS40DE_Ascending', 'RADS40DE_Rectum', 'SQ8ICUXK_Transverse', 'TMYPX044_Transverse', 'TRS8XIRT_Cecum', 'Y4UFTIIO_Ascending', 'Y4UFTIIO_Cecum', 'Y7CXU9SM_Rectum', 'Y7CXU9SM_Transverse']
old_images = pd.Series(old_images).str.split('_',expand=True).rename(columns={0:'qid', 1:'tissue'})

[os.remove(os.path.join(dir_cell, ff)) for ff in os.listdir(dir_cell)]
# Loop through each patient and find a random crop associated with them
np.random.seed(1234)
for ii, rr in df_sub.iterrows():
    file = rr['file']
    tissue = rr['tissue']
    id = rr['ID']
    tt = np.where(id.split('-')[0] == 'S18', ['test'], ['train'])[0]
    qid = df_code[df_code.ID == id].QID.unique()[0]
    idx_old = (old_images.qid == qid) & (old_images.tissue == tissue)
    if idx_old.any():
        print('Skipping image, ID-tissue from old batch')
        continue
    else:
        print('New image, will sample')
    path = os.path.join(dir_data, 'cropped', tt, qid, tissue)
    crop = np.random.choice(os.listdir(path), 1)[0]
    shutil.copyfile(os.path.join(path, crop), os.path.join(dir_cell, crop))
    # assert len(os.listdir(dir_cell)) == (ii + 1)
