# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 09:32:44 2020

@author: user
"""

import numpy as np
import pandas as pd
import os
# import matplotlib
# import matplotlib.pyplot as plt
# import plotly.express as px
# from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# import time
# lbl is used twice in the df, as a column name for the histological lbl
# and to denote the labels in the y column

dir_base = os.getcwd()

df1 = pd.read_csv(os.path.join(dir_base, 'df_ordinal_score.csv'))
df1 = df1[~df1['lbl'].isna()].copy()
df_score = df1[df1['y'] == 'score']
df_lbl = df1[df1['y'] == 'lbl']

df = df_score.merge(df_lbl, on = ['ID', 'tissue', 'type', 'lbl', 'histo']).groupby(['ID', 'tissue', 'histo', 'lbl']).agg(np.mean)
df = df.reset_index()
df = df[~df['value_y'].isna()].copy()

counter =0
fig_multi = make_subplots(rows = 1, cols = 5, subplot_titles = (
    'nancy AIC', 'nancy CII', 'robarts CII', 'robarts LPN', 'robarts NIE'))
colors_list = ['#ff0000', '#ff8000', '#ffff00', '#0000ff']

results_hists = []

#append graphs for each histological measure to results_hists
for histo in df['histo'].unique():
    for lbl in df[df['histo'] == histo]['lbl'].unique():
        counter += 1
        fig = go.Figure()

            
        fig.update_layout(
            title = histo+lbl)
        
        counter2 = 0
        for score in df[(df['histo'] == histo) & (df['lbl'] == lbl)]['value_y'].unique():
            data = df[(df['histo'] == histo) & (df['lbl'] == lbl) & \
                      (df['value_y'] == score)]['value_x'].values.tolist()
            minihist = go.Histogram(x = data, marker_color = colors_list[counter2], name = str(score))

            fig.add_trace(minihist)
            fig.update_layout(barmode = 'overlay')
            fig.update_traces(opacity = 0.75)

            fig_multi.add_trace(minihist, row = 1, col = counter)
            counter2 += 1
        results_hists.append(fig)

    


