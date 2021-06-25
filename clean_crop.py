# Import necessary modules
import sys
import os
import re
import shutil
import numpy as np
import pandas as pd
import cv2
import gc
import pathlib
from datetime import datetime
from funs_support import find_dir_GI, makeifnot, random_crop, listfiles

######################################
# ------ STEP 1 CODE SET UP ------- #

# Existing folder
dir_base = find_dir_GI()
dir_data = os.path.join(dir_base, 'data')
dir_20x = os.path.join(dir_data, '20X')
dir_images = os.path.join(dir_data, 'images')
lst_dir = [dir_data, dir_images, dir_20x]
assert all([os.path.exists(ff) for ff in lst_dir])
# New folders
dir_cleaned = os.path.join(dir_data, 'cleaned')
makeifnot(dir_cleaned)
dir_cropped = os.path.join(dir_data, 'cropped')
makeifnot(dir_cropped)
fold_cleaned = listfiles(dir_cleaned)


#########################################
# ------ STEP 2: SORT NEW FILES ------- #
#########################################

# Load in the patient annotations
dat_20x = pd.read_excel(os.path.join(dir_data, 'RI_NI_additional AI_moderate_severe_histotool.xlsx'))
dat_20x = dat_20x.melt('idx', None, 'tissue', 'idt').dropna().drop(columns='idx')
assert not dat_20x.duplicated().any()
di_tissue = {'Lower GI': 'Rectum', 'Upper GI': 'Upper'}
dat_20x = dat_20x.assign(tissue=lambda x: x.tissue.map(di_tissue)).query('tissue=="Rectum"')
# Get the list of svs files
fn_20x = pd.Series(listfiles(dir_20x))
idt_20x = fn_20x.str.split('\\s', 1, True).iloc[:, 0]
assert not idt_20x.duplicated().any()
print('Dropping non-matching patient: %s' % (np.setdiff1d(idt_20x, dat_20x.idt)))
idt_20x = list(np.intersect1d(idt_20x, dat_20x.idt))

print('We have %i new rectal images' % (len(idt_20x)))

# Loop through NEW IMAGES and make a folder in the ~/data/images folder
for idt in idt_20x:
    print('Patient: %s' % idt)
    dir_idt = os.path.join(dir_images, idt)
    makeifnot(dir_idt)
    fn_idt = fn_20x[fn_20x.str.contains(idt)].to_list()[0]
    fn_new = fn_idt.split(' ')[0] + ' (Rectum).svs'
    path_old = os.path.join(dir_20x, fn_idt)
    path_new = os.path.join(dir_idt, fn_new)
    if os.path.exists(path_new):
        fn_new = fn_new.split('.')[0] + '-001.svs'
    if not os.path.exists(path_new):
        print('Transfering file')
        shutil.copy(src=path_old, dst=path_new)
    else:
        print('File is already there')

fold_images = listfiles(dir_images)
print('There are %i total patient folders' % len(fold_images))

all_images = pd.concat([pd.DataFrame({'date': [
    datetime.fromtimestamp(pathlib.Path(os.path.join(dir_images, idt, z)).stat().st_mtime).strftime('%Y-%m-%d') for z in
    listfiles(os.path.join(dir_images, idt))], 'idt': idt}) for idt in fold_images])
all_images = all_images.assign(date=lambda x: pd.to_datetime(x.date)).sort_values(['idt', 'date']).reset_index(None,
                                                                                                               True)
all_images = all_images.assign(year=lambda x: x.date.dt.strftime('%Y').astype(int))
assert np.all(all_images.groupby('idt').year.var().fillna(0) == 0)


########################################
# ------ STEP 3: IMAGE PROCESS ------- #
########################################

# Different tissue types
tissues = ['Cecum', 'Ascending', 'Transverse', 'Descending', 'Sigmoid', 'Rectum']

# Loop through each and move to a specific class folder
for ii, fold in enumerate(fold_images):
    print('----- Folder: %s (%i of %i) -----' % (fold, ii + 1, len(fold_images)))
    fold_path = os.path.join(dir_images, fold)
    tmp_fns = pd.Series(listfiles(fold_path))
    tmp_split = tmp_fns.str.split('\\s\\(', expand=True)
    tmp_id = tmp_split.iloc[:, 0].unique()[0]
    if not tmp_id == fold:
        sys.exit('Error! ID in folder does not align')
    tmp_loc = tmp_split.iloc[:, 1].str.split('\\)', expand=True).iloc[:, 0]
    # Make patient folder if not exists
    fold_out = os.path.join(dir_cleaned, fold)
    makeifnot(fold_out)
    # Loop through each image and load
    for ii, fn in enumerate(tmp_fns):
        lbl_ii = tmp_loc[ii]
        print('Image %s (%i of %i)' % (lbl_ii, ii + 1, len(tmp_fns)))
        fn_out = 'cleaned_' + re.sub('\\s', '_', re.sub('\\(|\\)', '', fn)).replace('.svs', '.png')
        path_out = os.path.join(dir_cleaned, fold_out, fn_out)
        if not os.path.exists(path_out):
            print('Does not exist')
            # --- Step 1: Load in the gray-scale image and crop --- #
            gray_ii = cv2.imread(os.path.join(dir_images, fold, fn), cv2.IMREAD_GRAYSCALE)
            gray_small_ii = cv2.resize(src=gray_ii, dsize=None, fx=0.25, fy=0.25)
            n, p = gray_small_ii.shape
            mpix = max(n, p)
            # Apply a two stage gaussian filter
            stride = int(np.ceil(mpix * 0.01) + np.where(np.ceil(mpix * 0.01) % 2 == 0, 1, 0))
            stride2 = stride * 10 + np.where(stride * 10 % 2 == 0, 1, 0)
            blurry = cv2.GaussianBlur(cv2.GaussianBlur(gray_small_ii, (stride, stride), 0), (stride2, stride2), 0)
            mi, mx = int(blurry.min()), int(blurry.max())
            cu = int(np.floor((mi + mx) / 2))
            cidx = np.setdiff1d(np.arange(blurry.shape[1]), np.where(np.sum(blurry < cu, axis=0) == 0)[0])
            ridx = np.setdiff1d(np.arange(blurry.shape[0]), np.where(np.sum(blurry < cu, axis=1) == 0)[0])
            rmi, rma = int(np.min(ridx)) - 1, int(np.max(ridx)) + 1
            cmi, cma = int(np.min(cidx)) - 1, int(np.max(cidx)) + 1
            # Add on 4% of the pixels for a buffer
            nstride = 4
            rmi, rma = max(rmi - nstride * stride, 0), min(rma + nstride * stride, n)
            cmi, cma = max(cmi - nstride * stride, 0), min(cma + nstride * stride, n)
            # Get the scaling coordinates (r1/r2 & c1/c2
            ratio_r, ratio_c = gray_ii.shape[0] / n, gray_ii.shape[1] / p
            r1, r2 = int(np.floor(rmi * ratio_r)), int(np.ceil(rma * ratio_r))
            c1, c2 = int(np.floor(cmi * ratio_c)), int(np.ceil(cma * ratio_c))

            # --- Step 2: Load colour image and remove artifacts --- #
            col_ii = cv2.imread(os.path.join(dir_images, fold, fn), cv2.IMREAD_COLOR)[r1:r2, c1:c2]
            # Shrink for faster calculations
            var_ii = col_ii.var(axis=2)
            for kk in range(3):
                col_ii[:, :, kk] = np.where(var_ii > 5, col_ii[:, :, kk], col_ii[:, :, kk].max())

            # --- Step 3: Save image for later --- #
            cv2.imwrite(path_out, col_ii, [cv2.IMWRITE_PNG_COMPRESSION, 2])

############################################
# ------ STEP 4: APPLY RANDOM CROP ------- #
############################################

n_crop = 100
size_crop = 500

# Load the existing index
old_dat_idx_crops = pd.read_csv(os.path.join(dir_data, 'dat_idx_crops.csv'))
old_idt = list(old_dat_idx_crops.idt.unique())
# Add on any IDs
new_idt = list(np.setdiff1d(fold_cleaned, old_idt))
if len(new_idt) > 0:
    print('Adding new IDs to the dat_idx_crop')
    idt_all = old_idt + new_idt
else:
    print('Not adding IDs to the dat_idx_crop')
    idt_all = old_idt

# Loop through each and move to a specific class folder
holder = []
for ii, fold in enumerate(idt_all):
    fold_ii = os.path.join(dir_cleaned, fold)
    out_ii = os.path.join(dir_cropped, fold)
    if not os.path.exists(out_ii):
        print('----- Folder: %s (%i of %i) -----' % (fold, ii + 1, len(idt_all)))
        makeifnot(out_ii)
        # Get the tissue order (needs to be sorted to ensure identical results across systems)
        fn_ii = pd.Series(listfiles(fold_ii))
        if fold in old_idt:
            print('Getting previous tissue order')
            tmp_ii = list(old_dat_idx_crops.query('idt == @fold').tissue.unique())
            tmp2_ii = [np.where(fn_ii.str.contains(t))[0][0] for t in tmp_ii]
            fn_ii = list(fn_ii[tmp2_ii])
        for jj, fn in enumerate(fn_ii):
            tissue_jj = fn.split('_')[2].replace('.png', '')
            out_jj = os.path.join(out_ii, tissue_jj)
            makeifnot(out_jj)
            # Load image
            img_jj = cv2.imread(os.path.join(fold_ii, fn), cv2.IMREAD_COLOR)
            height, width, channels = img_jj.shape
            gc.collect()
            kk, rr = 0, 0
            while kk <= n_crop:
                rr += 1
                img_kk, yidx_kk, xidx_kk = random_crop(img_jj, height, width, size_crop, ii + jj + rr)
                mu_kk = img_kk.mean() / 255
                if mu_kk <= 0.95:
                    fn_kk = fn.replace('.png', '_' + str(kk) + '.png')
                    cv2.imwrite(os.path.join(out_jj, fn_kk), img_kk, [cv2.IMWRITE_PNG_COMPRESSION, 2])
                    tmp = pd.DataFrame({'idt': fold, 'tissue': tissue_jj, 'sample': kk,
                                        'yidx': yidx_kk, 'xidx': xidx_kk}, index=[0])
                    kk += 1
                    holder.append(tmp)
                    if kk % 25 == 0:
                        print('Crop %i of %i' % (kk, n_crop))

# Get positions
if len(holder) > 0:
    new_dat_idx_crops = pd.concat(holder)
    # New with old files
    idt_new = new_dat_idx_crops.idt.unique()
    idt_existing = np.setdiff1d(idt_all, idt_new)
    dat_idx_crops = pd.concat([new_dat_idx_crops, old_dat_idx_crops.query('idt.isin(@idt_existing)',engine='python')]).reset_index(None, True)
else:
    dat_idx_crops = old_dat_idx_crops.copy().reset_index(None, True)

# Make sure it is all unique
assert not dat_idx_crops.duplicated().any()
dat_idx_crops.to_csv(os.path.join(dir_data, 'dat_idx_crops.csv'), index=False)

# End of script
print('------ END OF SCRIPT: clean_crop.py -------')
