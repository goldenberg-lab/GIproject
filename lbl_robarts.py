"""
Script to convert the Robarts score in a CSV format
"""

# Load in all necessary modules
import numpy as np
import pandas as pd
import datetime as dt
import os
import sys
import seaborn as sns
from matplotlib import pyplot as plt

from funs_support import find_dir_GI, makeifnot, no_diff, stopifnot

# Assign data directory
dir_base = find_dir_GI()
dir_data = os.path.join(dir_base, 'data')
dir_20x = os.path.join(dir_data, '20X')
dir_images = os.path.join(dir_data, 'cleaned')
dir_output = os.path.join(dir_base, 'output')
makeifnot(dir_output)

# Define the file names
fn_clin = os.path.join(dir_data,'annot_histo_stratifier.csv')
fn_code = os.path.join(dir_data,'code_breaker_histo.csv')
fn_robarts = os.path.join(dir_data,'robarts_score_histo.xlsx')

for ff in [dir_data, dir_images, fn_clin, fn_code, fn_robarts]:
    stopifnot(os.path.exists(ff),'Error! ' + ff + ' path not found!')

########################################
# ----- STEP 1: LOAD IN THE DATA ----- #
########################################

# New patients (won't have demographic data)
fn_20x = pd.Series(os.listdir(dir_20x))
idt_20x = fn_20x.str.split('\\s',1,True).iloc[:,0]

# Different tissue types
tissues = ['Cecum', 'Ascending', 'Transverse', 'Descending', 'Sigmoid', 'Rectum']

# --- (1) Get a list of IDs/Scans from the image folder --- #
df_scans = pd.DataFrame([])
for ff in os.listdir(dir_images):
    df_scans = pd.concat([df_scans,pd.DataFrame({'ID':ff,'tissue':os.listdir(os.path.join(dir_images,ff))})])
df_scans.reset_index(drop=True,inplace=True)
df_scans['file'] = df_scans.tissue.copy()
# Replace Splenic_Flexure with Descending
df_scans['tissue'] = df_scans.tissue.str.replace('Splenic_Flexure','Descending')
df_scans['tissue'] = df_scans.tissue.str.split('\\_',2,True).iloc[:,2]
# Remove .png or -001
df_scans['tissue'] = df_scans.tissue.str.split('\\.|\\-',1,True)[0]
df_scans_long = df_scans.groupby(['ID','tissue']).size().reset_index().rename(columns={0:'n'})
df_scans_long = df_scans_long.pivot(index='ID',columns='tissue',values='n').reset_index()
df_scans_long = df_scans_long.melt('ID',None,None,'n').assign(n=lambda x: x.n.fillna(0).astype(int))
df_scans_wide = df_scans_long.pivot(index='ID',columns='tissue',values='n').reset_index()
# Patients which we have an entire .svs for
idt_scans = pd.Series(df_scans.ID.unique())
idt_scans_orig = pd.Series(np.setdiff1d(idt_scans, idt_20x))
print('We have %i patients with a .svs file, and %i original' % (len(idt_scans), len(idt_scans_orig)))
df_scans_long_orig = df_scans_long.query('ID.isin(@idt_scans_orig)',engine='python').reset_index(None, True)
df_scans_orig = df_scans.query('ID.isin(@idt_scans_orig)',engine='python').reset_index(None, True)
df_scans_wide_orig = df_scans_wide.query('ID.isin(@idt_scans_orig)',engine='python').reset_index(None, True)

# --- (2) Load in the clinical data --- #

# Columns: sex [obvious], dob [date of birth],
    #       ibd_dx_dt [IBD diagnosis date],
    #       lab_dt [date of lab procedure]
df_clin = pd.read_csv(fn_clin).rename(columns={'path_id':'ID'})
df_clin.replace({'sex':{1:'M',2:'F'}},inplace=True) # Recode sex
# Remove empty string spaces from path ID
df_clin['ID'] = df_clin.ID.str.strip()
# All of the variables are identical so we can subset them
df_clin = df_clin[df_clin.groupby('ID').cumcount()==0].reset_index(drop=True)
# Transform dates
cn_date = ['dob', 'ibd_dx_dt', 'lab_dt']
if not all(df_clin[cn_date].dtypes == 'int64'):
    sys.exit('Error! The date columns are not integers')
for cn in cn_date:
    print('--------- Column : %s ---------' % cn)
    df_clin[cn] = pd.to_datetime([dt.date(1582,10,14) + dt.timedelta(seconds=x) for x in df_clin[cn].to_list()])
# Re-order
df_clin = df_clin[['ID','sex'] + cn_date]
# Calculate age at diagnosies
df_clin['age_diag'] = (df_clin.ibd_dx_dt - df_clin.dob).dt.days.values
df_clin['age_lab'] = (df_clin.lab_dt - df_clin.dob).dt.days.values
# Patients that have a lab date before or around their diagnosis date are new patients
df_clin['patient'] = np.where(df_clin.age_lab - df_clin.age_diag <= 30,'new','existing')
assert ~df_clin.ID.duplicated().any()
idt_demo = df_clin.ID.copy()
# Subset the clinical data to line up with the data we actually have scans for
assert len(np.setdiff1d(idt_scans_orig,idt_demo))==0
df_clin = df_clin[df_clin.ID.isin(idt_scans_orig)].reset_index(drop=True)

# --- (3) Load in the code breaker tabulations --- #
df_code = pd.read_csv(fn_code).rename(columns={'ID code':'breaker','PATH ID':'ID'})
df_code.rename(columns=dict(zip(df_code.columns[2:],df_code.columns[2:].str.capitalize())),inplace=True)
df_code.columns = df_code.columns.str.strip()
assert no_diff(idt_scans_orig, df_code.ID)
# Check that the code-breaker tabulation lines up
df_code_long = df_code.melt(id_vars=['breaker','ID'],var_name='tissue',value_name='n')
tmp_check = df_code_long.merge(df_scans_long_orig,how='outer',on=['ID','tissue'])
cond1 = tmp_check.shape[0] == df_scans_long_orig.shape[0]
cond2 =  np.all(tmp_check.n_x == tmp_check.n_y)
assert cond1 and cond2

# ---- (4) Load in the Robarts histological scores --- #
df_robarts = pd.read_excel(fn_robarts).rename(columns={'Code':'breaker'})
stopifnot(np.all([np.any(x == df_robarts.columns) for x in tissues]),'Columns do not align')
# Find the associated columns that match the tissues order
cidx_tissues = np.array([df_robarts.columns.to_list().index(x) for x in tissues])
cidx_CII = cidx_tissues+1 # Chronic Inflammatory infiltrate
cidx_LPN = cidx_tissues+2 # Lamina propria neutrophils
cidx_NIE = cidx_tissues+3 # Neutrophils in epithelium
cidx_EOU = cidx_tissues+4 # Erosion or ulceration
# Ensure column names are all the same
cidx_lbls = [cidx_CII, cidx_LPN, cidx_NIE, cidx_EOU]
for ii in cidx_lbls:
    check_ii = pd.Series(df_robarts.columns[ii].values).str.split('\\,|\\.[0-9]$',expand=True,n=1).iloc[:,0].unique()
    stopifnot(check_ii.shape[0] == 1,'Error! non-duplicate columns')
# Assign to a Pandas dataframe the score
cn_tissues = df_robarts.columns[cidx_tissues]
cn_lbls = ['CII','LPN','NIE','EOU']
di_cidx = dict(zip(cn_lbls, cidx_lbls))
df_holder = pd.DataFrame([])
for cn in cn_lbls:
    print(cn)
    tmp_cn = df_robarts.iloc[:, di_cidx[cn]]
    tmp_cn.columns = cn_tissues
    tmp_cn = tmp_cn.astype(str).apply(func=lambda x: np.where(x.str.contains('^[0-9]'),x,np.NaN), axis=0).astype(float)
    tmp_cn = pd.concat([pd.DataFrame({'breaker':df_robarts.breaker,'lbl':cn}),tmp_cn],axis=1)
    df_holder = pd.concat([df_holder, tmp_cn],axis=0)
df_holder.reset_index(drop=True,inplace=True)
df_lbls = df_code[['breaker','ID']].merge(df_holder,how='right',on='breaker').drop(columns='breaker')
# Merge with file IDs too
df_lbls = df_lbls.melt(id_vars=['ID','lbl'],var_name='tissue')
df_lbls = df_lbls.pivot_table(index=['ID','tissue'],columns='lbl',aggfunc=lambda x: x).reset_index()
df_lbls.columns = ['ID','tissue'] + [x[1] for x in df_lbls.columns][2:]
df_lbls = df_scans_orig.merge(df_lbls,how='left',on=['ID','tissue'])

# Add on the patient meta-data
df_lbls = df_clin[['ID','lab_dt','sex','age_lab']].merge(df_lbls,how='right',on='ID')

# Save for later
df_lbls.to_csv(os.path.join(dir_data,'df_lbls_robarts.csv'),index=False)

##########################################################
# ----- STEP 2: PRINT OFF SOME BASIC SUMMARY STATS ----- #
##########################################################

# Sex distribution
print('Sex table')
print(df_clin.sex.value_counts().reset_index())
# New/Old patient
print('Patient type')
print(df_clin.patient.value_counts().reset_index())
# Age range
print('Age range')
print((df_clin.age_lab / 365.25).describe())
tmp_df = df_clin[['ID','age_diag','age_lab']].melt(id_vars='ID')
tmp_df.value = tmp_df.value / 365.25
g = sns.FacetGrid(tmp_df,row='variable',margin_titles=True)
g.map(plt.hist, 'value', color='blue', bins=np.arange(0,18,1))
g.savefig(os.path.join(dir_output,'age_dist.png'))
plt.close()

# Correlation between scores
ax = sns.heatmap(df_lbls[cn_lbls].corr())
ax.set_ylim(4, 0)
ax.set_title('Correlation by score',size=18)
ax.figure.savefig(fname=os.path.join(dir_output,'corr_score.png'))
plt.close()

# Correlation between tissues
tmp_df = df_lbls.melt(id_vars=['ID','tissue'],value_vars=cn_lbls,var_name='scores')
tmp_df = tmp_df.pivot_table(index=['ID','scores'],columns='tissue',values='value').reset_index().iloc[:,2:].corr()
tmp_df.index.name=''
tmp_df.columns.name = ''
plt.figure(figsize=(10,8))
ax = sns.heatmap(tmp_df,square=True); ax.set_ylim(6, 0)
ax.set_title('Correlation by tissue',size=18)
ax.figure.savefig(fname=os.path.join(dir_output,'corr_tissue.png'))
plt.close()

# Tabular frequency of scores
df_tab = df_lbls.melt(id_vars='ID',value_vars=cn_lbls,var_name='score').groupby(['score','value']).size().reset_index()
df_tab = df_tab.rename(columns={0:'n'}).pivot(index='score',columns='value',values='n').reset_index()
df_tab = df_tab.melt(id_vars='score',value_name='n')
df_tab.n = np.where(df_tab.n.isnull(),0,df_tab.n).astype(int)
df_tab.rename(columns={'value':'ordinal'},inplace=True)
df_tab = df_tab.groupby('score')['n'].sum().reset_index().rename(columns={'n':'tot'}).merge(df_tab,how='right',on='score')
df_tab['share'] = df_tab.n / df_tab.tot
# plot it
ax = sns.barplot(x='score',y='share',hue='ordinal',data=df_tab)
ax.set_xlabel('Score type'); ax.set_ylabel('Percent of labels')
ax.figure.savefig(fname=os.path.join(dir_output,'dist_lbls.png'))


