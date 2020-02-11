# Import necessary modules
import sys
import os
import re
import numpy as np
import pandas as pd
import cv2
import gc

######################################
# ------ STEP 0: CODE SET UP ------- #

dir_base = os.getcwd()
dir_data = os.path.join(dir_base,'..','data')
dir_images = os.path.join(dir_data,'images')
dir_cleaned = os.path.join(dir_data,'cleaned')

if not os.path.exists(dir_data) & os.path.exists(dir_images):
    sys.exit('Error! data and image directory is not where it is expected')
if not os.path.exists(dir_cleaned):
    print('Creating cleaned folder')
    os.mkdir(dir_cleaned)

fold_images = os.listdir(dir_images)
print('There are %i total patient folders' % len(fold_images) )

########################################
# ------ STEP 1: IMAGE PROCESS ------- #
########################################

# Different tissue types
tissues = ['Cecum', 'Ascending', 'Transverse', 'Descending', 'Sigmoid', 'Rectum']

# Loop through each and move to a specific class folder
for ii, fold in enumerate(fold_images):
    print('----- Folder: %s (%i of %i) -----' % (fold, ii+1, len(fold_images)))
    fold_path = os.path.join(dir_images,fold)
    tmp_fns = pd.Series(os.listdir(fold_path))
    tmp_split = tmp_fns.str.split('\\s\\(',expand=True)
    tmp_id = tmp_split.iloc[:,0].unique()[0]
    if not tmp_id == fold:
        sys.exit('Error! ID in folder does not align')
    tmp_loc = tmp_split.iloc[:,1].str.split('\\)',expand=True).iloc[:,0]
    # Make patient folder if not exists
    fold_out = os.path.join(dir_cleaned,fold)
    if not os.path.exists(fold_out):
        print('Cleaned folder does not exist, creating')
        os.mkdir(fold_out)
    # Loop through each image and load
    for ii, fn in enumerate(tmp_fns):
        lbl_ii = tmp_loc[ii]
        print('Image %s (%i of %i)' % (lbl_ii,ii+1,len(tmp_fns)) )

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
        fn_out = 'cleaned_' + re.sub('\\s', '_', re.sub('\\(|\\)', '', fn)).replace('.svs', '.png')
        cv2.imwrite(os.path.join(dir_cleaned, fold_out, fn_out), col_ii, [cv2.IMWRITE_PNG_COMPRESSION, 2])

############################################
# ------ STEP 2: APPLY RANDOM CROP ------- #
############################################

from matplotlib import pyplot as plt

fold_cleaned = os.listdir(dir_cleaned)
dir_cropped = os.path.join(dir_data,'cropped')
if not os.path.exists(dir_cropped):
    print('Making cropped folder'); os.mkdir(dir_cropped)

n_crop = 100
size_crop = 500

def random_crop(img,height,width,crop_size,ss):
    np.random.seed(ss)
    yidx = np.random.choice(np.arange(height-crop_size))
    xidx = np.random.choice(np.arange(width-crop_size))
    cropped = img[yidx:(yidx+crop_size+1),xidx:(xidx+crop_size)+1 ].copy()
    return(cropped)

# Loop through each and move to a specific class folder
for ii, fold in enumerate(fold_cleaned):
    print('----- Folder: %s (%i of %i) -----' % (fold, ii+1, len(fold_cleaned)))
    fold_ii = os.path.join(dir_cleaned, fold)
    fn_ii = os.listdir(fold_ii)
    out_ii = os.path.join(dir_cropped, fold)
    if not os.path.exists(out_ii):
        os.mkdir(out_ii)
    for jj, fn in enumerate(fn_ii):
        print('Image %i of %i' % (jj+1, len(fn_ii)))
        tissue_jj = fn.split('_')[2].replace('.png','')
        out_jj = os.path.join(out_ii, tissue_jj)
        if not os.path.exists(out_jj):
            os.mkdir(out_jj)
        # Load image
        img_jj = cv2.imread(os.path.join(fold_ii, fn), cv2.IMREAD_COLOR)
        height, width, channels = img_jj.shape
        gc.collect()
        kk, rr = 0, 0
        while kk <= n_crop:
            rr += 1
            img_kk = random_crop(img_jj, height, width, size_crop, ii + jj + rr)
            mu_kk = img_kk.mean() / 255
            if mu_kk <= 0.95:
                fn_kk = fn.replace('.png','_'+str(kk)+'.png')
                cv2.imwrite(os.path.join(out_jj,fn_kk),img_kk, [cv2.IMWRITE_PNG_COMPRESSION, 2])
                kk += 1
                if kk % 25 == 0:
                    print('Crop %i of %i' % (kk, n_crop))

# End of script
print('------ END OF SCRIPT: image_clean.py -------')