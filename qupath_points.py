# -*- coding: utf-8 -*-
"""
SCRIPT TO PARSE THE QU-PATH POINTS
"""

import sys
import os
from zipfile import ZipFile
import shutil
import numpy as np
import pandas as pd

os.chdir('C:\\Users\\erik drysdale\\Documents\\projects\\GI')
dir_base = os.getcwd()
dir_images = os.path.join(dir_base, 'images')
dir_points = os.path.join(dir_base, 'points')

def stopifnot(cond):
    if not cond:
        sys.exit('error!')

valid_cells = ['eosinophil','neutrophil','plasma','enterocyte','other','lymphocyte']

# Function to parse the zipped file
def zip_points_parse(fn):
    valid_files = ['Points '+str(k+1)+'.txt' for k in range(6)]
    with ZipFile(file=fn,mode='r') as zf:
        names = pd.Series(zf.namelist())
        stopifnot(names.isin(valid_files).all())
        zf.extractall('tmp')
    # Loop through and parse files
    holder = []
    for pp in names:
        s_pp = pd.read_csv(os.path.join(dir_base,'tmp',pp),sep='\t',header=None)
        stopifnot(s_pp.loc[0,0] == 'Name')
        cell_pp = s_pp.loc[0,1].lower()
        stopifnot(cell_pp in valid_cells)
        df_pp = pd.DataFrame(s_pp.loc[3:].values.astype(float),columns=['x','y'])
        stopifnot(df_pp.shape[0] == int(s_pp.loc[2,1])) # number of coords lines up
        df_pp.insert(0,'cell',cell_pp)
        holder.append(df_pp)
    df = pd.concat(holder).reset_index(drop=True)
    shutil.rmtree('tmp', ignore_errors=True) # Get rid of temporary folder    
    return(df)


##################################
## --- (1) LOAD IN THE DATA --- ##

fn_points = os.listdir(dir_points)
fn_images = os.listdir(dir_images)
raw_points = pd.Series(fn_points).str.split('\\.',expand=True).iloc[:,0]
raw_images = pd.Series(fn_images).str.split('\\.',expand=True).iloc[:,0]
stopifnot(raw_points.isin(raw_images).all())

import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image

fig, axes = plt.subplots(nrows=len(fn_points),ncols=1,figsize=(10,20))

for ii, fn in enumerate(fn_points):
    print('file %s (%i of %i)' % (fn, ii+1, len(fn_points)))
    path = os.path.join(dir_points, fn)
    df_ii = zip_points_parse(path)
    df_ii.cell = pd.Categorical(df_ii.cell,categories=valid_cells)
    path_img = os.path.join(dir_images,fn.split('-')[0])
    img_vals =  np.array(Image.open(path_img))
    axes[ii].imshow(img_vals)
    #axes.imshow(img_vals)
    sns.scatterplot(x='x',y='y',hue='cell',data=df_ii,ax=axes[ii])
    

    
