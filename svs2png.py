# Script to convert the svs files to png for transfer to CCHMC

import os
import cv2
import shutil
import pandas as pd
from PIL import Image
from funs_support import find_dir_GI

# Get data directory
dir_base = find_dir_GI()
dir_data = os.path.join(dir_base, 'data')
dir_WSI = os.path.join(dir_data, 'cidscann_WSI')
dir_png = os.path.join(dir_WSI, 'png')

# Loop over each folder
jj = 0
for fold in os.listdir(dir_WSI):
    dir_fold = os.path.join(dir_WSI, fold)
    for mag in os.listdir(dir_fold):
        dir_mag = os.path.join(dir_fold, mag)
        fn_mag = pd.Series(os.listdir(dir_mag))
        fn_mag = fn_mag[fn_mag.str.contains('\\.svs$',regex=True)]
        for fn in fn_mag:
            jj += 1
            path_svs = os.path.join(dir_mag, fn)
            fn_png = fn.replace('.svs','.png')
            path_png = os.path.join(dir_mag, fn_png)
            assert os.path.exists(path_svs)
            print('folder: %s, mag: %s, fn: %s (Iteration %i)' % (fold, mag, fn, jj))
            if not os.path.exists(path_png):
                # Convert to png
                img = cv2.imread(path_svs, cv2.IMREAD_COLOR)
                img = Image.fromarray(img)
                img.save(path_png)
            else:
                # Move to png folder
                fold_new = os.path.join(dir_png, fold, mag)
                if not os.path.exists(fold_new):
                    os.makedirs(fold_new, exist_ok=True)
                path_png_new = os.path.join(fold_new, fn_png)
                shutil.move(src=path_png, dst=path_png_new)
