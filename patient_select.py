"""
SCRIPT TO SELECT 43 RANDOM CROPS FROM 23 PATIENTS FOR THE CELL ANNOTATOR
"""

# Load in all necessary modules
import numpy as np
import pandas as pd
import os
import sys
import shutil

# Assign data directory
dir_base = os.getcwd()
dir_data = os.path.join(dir_base,'..','data')
dir_cell = os.path.join(dir_data,'cell_counter')

def stopifnot(arg, msg):
    if not arg:
        sys.exit(msg)

#################################################################
# -------------------------- JAZZ 36 -------------------------- #
#################################################################

# Load excel file
dat = pd.read_excel(os.path.join(dir_data,'UC_TO_all_histo.xlsx'))
raw = dat.copy()
dat.rename(columns={'subject_id':'id','gender_n':'sex','Final_Eth_Label':'ethnicity','MRN No#':'MRN'},inplace=True)
dat.replace({'sex':{1:'M',2:'F'}},inplace=True) # Recode sex
dat['yr'] = dat.dt_dx.dt.strftime('%Y').astype(int)

# (i) Keep only extensive disease
dat = dat[dat.DisPheno_max.isin(['E4'])]
# (ii) Moderate/severe Mayo score
dat = dat[dat.BL_Mayo >= 2]
# (iii) severe clinical disease index
dat = dat[dat.PUCAI_scr > 55]
# (iv) Physician assessment
dat = dat[dat.PGA.isin(['Moderate','Severe'])]
# Remove two white caucasian males to get to 50 patients
dat = dat[~((dat.ethnicity == 'Caucasian') & (dat.sex=='M') & (dat.BL_Mayo==2) & (dat.yr == 2014))]
# Reset
dat.reset_index(drop=True,inplace=True)

# Summary statistics
print('--- age at diagnosis (months) ---')
print(np.round((dat.Age_Dx_mths/12).describe()))
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

df_robarts = pd.read_csv(os.path.join(dir_data,'df_lbls_robarts.csv'))
df_nancy = pd.read_csv(os.path.join(dir_data,'df_lbls_nancy.csv'))
df_code = pd.read_csv(os.path.join(dir_data,'df_codebreaker.csv'))

# Average across scores for robarts
df_robarts['score'] = df_robarts[['CII','EOU','LPN','NIE']].apply(np.nanmean,axis=1)
df_robarts = df_robarts[df_robarts.score.notnull()].reset_index(drop=True)

# Select patients from each unique score
u_score = df_robarts.score.unique()
n_patients = 50
pps = n_patients // len(u_score)# patients per score

np.random.seed(1234)
holder = []
for ss in u_score:
    tmp = df_robarts[df_robarts.score == ss]
    if tmp.shape[0] > pps:
        holder.append(tmp.sample(n=pps))
    else:
        holder.append(tmp)
df_sub = pd.concat(holder).reset_index(drop=True)

# Inflammation
dat_inf=df_sub.melt('file',['CII','LPN','NIE','EOU']).groupby(['variable','value']).size().reset_index().rename(columns={0:'n','value':'lbl'})
dat_inf.lbl = dat_inf.lbl.astype(int)
print(dat_inf)
# Distribution of metric
print('A total of %i uniquie patients out of %i' % (df_sub.ID.unique().shape[0], df_sub.shape[0]) )
print(df_sub.ID.value_counts().reset_index())
# Sex
print(df_sub.sex.value_counts(normalize=True))
# Age range
np.round((df_sub.age_lab / 365).describe()).astype(int)

[os.remove(os.path.join(dir_cell,ff)) for ff in os.listdir(dir_cell)]
# Loop through each patient and find a random crop associated with them
np.random.seed(1234)
for ii, rr in df_sub.iterrows():
    file = rr['file']
    tissue = rr['tissue']
    id = rr['ID']
    tt = np.where(id.split('-')[0] == 'S18', ['test'], ['train'])[0]
    qid = df_code[df_code.ID == id].QID.unique()[0]
    path = os.path.join(dir_data, 'cropped', tt, qid, tissue)
    crop = np.random.choice(os.listdir(path),1)[0]
    shutil.copyfile(os.path.join(path,crop), os.path.join(dir_cell,crop))
    stopifnot(len(os.listdir(dir_cell)) == (ii+1),'error2')



