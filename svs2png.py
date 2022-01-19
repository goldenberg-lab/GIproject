# Script to convert the svs files to png for transfer to CCHMC

import os
import cv2
import shutil
import pandas as pd
from PIL import Image
from funs_support import find_dir_GI, makeifnot, listfolds

# Get data directory
dir_base = find_dir_GI()
dir_data = os.path.join(dir_base, 'data')
dir_WSI = os.path.join(dir_data, 'cidscann_WSI')
dir_png = os.path.join(dir_WSI, 'png')
makeifnot(dir_png)
dir_jpg = os.path.join(dir_WSI, 'jpg')
makeifnot(dir_jpg)
di_fmt = {'png':dir_png, 'jpg':dir_jpg}

# -- CONVERT EACH TO A PNG AND A JPG -- #
lst_fmt = ['png', 'jpg']
# Loop over each folder
jj = 0
for fold in os.listdir(dir_WSI):
    dir_fold = os.path.join(dir_WSI, fold)
    for mag in listfolds(dir_fold):
        dir_mag = os.path.join(dir_fold, mag)
        fn_mag = pd.Series(os.listdir(dir_mag))
        fn_mag = fn_mag[fn_mag.str.contains('\\.svs$',regex=True)]
        for fn in fn_mag:
            jj += 1
            path_svs = os.path.join(dir_mag, fn)
            assert os.path.exists(path_svs), 'svs file cannot be found'
            for fmt in lst_fmt:
                dir_fmt = di_fmt[fmt]
                fn_fmt = fn.replace('.svs','.'+fmt)
                path_fmt = os.path.join(dir_mag, fn_fmt)
                print('folder: %s, mag: %s, fn: %s (Iteration %i)' % (fold, mag, fn, jj))
                if not os.path.exists(path_fmt):
                    # Convert to png
                    img = cv2.imread(path_svs, cv2.IMREAD_COLOR)
                    img = Image.fromarray(img)
                    img.save(path_fmt)
                else:
                    # Move to png folder
                    fold_new = os.path.join(dir_fmt, fold, mag)
                    if not os.path.exists(fold_new):
                        os.makedirs(fold_new, exist_ok=True)
                    path_fmt_new = os.path.join(fold_new, fn_fmt)
                    shutil.move(src=path_fmt, dst=path_fmt_new)



