# Compare what files we have compared to cidscan manifest

import numpy as np
import pandas as pd
import os
from funs_support import find_dir_GI

# Existing folder
dir_base = find_dir_GI()
dir_data = os.path.join(dir_base, 'data')
dir_20x = os.path.join(dir_data, '20X')
dir_images = os.path.join(dir_data, 'images')

os.listdir(dir_data)


#####################################
# ------ STEP 1: FIND FILES ------- #

cn_scan = ['MRNNo#', 'Cidscann_no', 'dt_dx', 'dateofhisto', 'Accession_No']
df_scan = pd.read_excel(os.path.join(dir_data, 'UC_TO_cidscann_final_August2021.xlsx'),usecols=cn_scan)
df_scan = df_scan.dropna()
df_scan['in_xlsx'] = True

fn_images = pd.Series(os.listdir(dir_images)).append(pd.Series(os.listdir(dir_20x)))
fn_images = fn_images[fn_images.str.contains('^S')]
df_images = pd.DataFrame({'Accession_No':fn_images.str.split('\\s',1,True)[0].unique(),'in_folder':True})

df_both = df_scan.merge(df_images,'outer')[['Accession_No','in_xlsx','in_folder']]
df_both = df_both.fillna(False).sort_values(['in_xlsx','in_folder','Accession_No']).reset_index(None,True)
df_both.to_csv(os.path.join(dir_data,'cidscann_match.csv'))
df_both.groupby(['in_xlsx','in_folder']).size()


