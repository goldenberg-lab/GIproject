#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:07:36 2020

@author: user
"""

import base64
import datetime
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State 
from PIL import Image
import os
from skimage import io
from io import BytesIO
import plotly.graph_objects as go
import cv2
import json
import torch
from models import mdls_torch
import plotly.express as px
dir_base = os.getcwd()
dir_networks = os.path.join(dir_base,'saved_networks')


    
app = dash.Dash()


app.layout = html.Div(
    [

    dcc.Store(id = 'memory',
              # storage_type = 'session'
              ),
    dcc.Upload(
            id = 'upload-image',
            children = html.Div([
                    'drag and drop or ',
                    html.A('Select Files')
                    ]),
            style = {
                    'width':'50%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'},
            multiple = True
        ),
    dcc.Dropdown(
        id = 'classifier-selector',
        options=[
            {'label': 'Robarts CII', 'value':'robarts_CII'},
            {'label': 'Robarts LPN', 'value': 'robarts_LPN'},
            {'label': 'Robarts NIE', 'value': 'robarts_NIE'},
            {'label': 'Nancy CII', 'value': 'nancy_CII'},
            {'label': 'Nancy NIE', 'value': 'nancy_NIE'}],
        style = {
            'width': '60%'
            },
        placeholder = 'Select a classification scheme (default Robarts CII)'
        ),

    dcc.Graph(
        id = 'output-image-upload'
        ),
    
    dcc.Graph(
        id = 'zoom-window'
        ),
    html.Div(
        id = 'test-div'),
    html.Div(
        id = 'test-div-2')
    
    ]
    )



##################################
    
def b64_to_pil(string):
    decoded = base64.b64decode(string)
    buffer = BytesIO(decoded)
    im = Image.open(buffer)
    return im

##################################

def load_model():
    torch.manual_seed(12345)
    mdl = mdls_torch.CNN_ordinal()
    mdl_inf = mdls_torch.CNN_ordinal()
    # fn_pre = pd.Series(os.listdir(dir_networks))
    # fn = fn_pre[fn_pre.str.split('epoch|\\.',expand=True).iloc[:,1].astype(int).idxmax()]
    # mdl_inf.load_state_dict(torch.load(os.path.join(dir_networks,fn)))
    mdl_inf.load_state_dict(torch.load(os.path.join(dir_networks, 'cnn_conc_epoch10000.pt')))
    
    return mdl_inf

##################################


    


def take_crops(img):
    n_crops = 50
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    crop_locations = pd.DataFrame(columns = ['x','y','robarts_CII', 'robarts_LPN','robarts_NIE', 'nancy_CII', 'nancy_AIC'])

    counter = 0
    test_counter = 0
    crop_size = 500
    height = img.size[1]
    width = img.size[0]
    mdl_inf = load_model()
    while counter < n_crops:
        test_counter += 1
        yidx = np.random.choice(np.arange(height - crop_size))
        xidx = np.random.choice(np.arange(width - crop_size))
        crop = image[yidx:(yidx+crop_size+1),xidx:(xidx+crop_size+1)]
        crop_mean = crop.mean()/255.0
        if crop_mean <= .95:
            colour_crop = np.array(img)[yidx:(yidx+crop_size+1),xidx:(xidx+crop_size+1)]
            # crop_locations.loc[counter] = [xidx, yidx, crop_mean]
            counter += 1
            crop_normal = crop/255.0
            # img_tensor = torch.Tensor(crop_normal.concatenate().transpose([0,3,2,1]))
            img_tensor = torch.Tensor(colour_crop/255.0).unsqueeze(0).transpose(1,3)
            score = mdl_inf(img_tensor).detach().numpy()
            crop_locations.loc[counter] = [xidx, yidx] + score.tolist()[0]
        else:
            counter = counter

         
   
        # crop_locations.loc[counter] = [xidx, yidx]
        # counter += 1

    

    crop_locations.loc[counter+1] = [width, height, -11, 0,0,0,0]

    return crop_locations
            
        
        
        

#######################

def InteractiveImage(image):
    #set up dummy graph the size of the image
    x_data = np.array([0, image.size[0]])
    y_data = np.array([0, image.size[1]])
    scaling_factor = 1
    
    crop_locations = take_crops(image)
    
    
    # return dcc.Graph(
    #     id='main-image',
    figure={
        'data': [
            {
                'x': crop_locations['x'],
                'y': [image.size[1] - i for i in crop_locations['y']],
                'name': 'dummy_trace',
                'mode': 'markers',
                'marker' : {'color' : crop_locations['robarts_CII']+11,
                            'opacity': 1,
                            'colorbar': {'thickness':10,
                                         'tickmode': 'array',
                                         'ticktext': ['low', 'high']}
                            }
                }],
        'layout': {
            'autosize': False,
            'margin': go.Margin(l = 40, b= 40, t = 26, r= 10),
            'xaxis': {
                'range': [0, x_data[1]*scaling_factor],
                # 'scaleanchor': 'y',
                'scaleratio': scaling_factor,
                
            },
            'yaxis': {
                'range': [0, y_data[1]*scaling_factor],
                # 'scaleanchor': 'x',
                'scaleratio': scaling_factor,
                # 'dticks': 100,
                'autorange': True,
                # 'rangemode': 'tozero',
                'constrain': 'domain'
                # 'fixedrange': True
                
            },

            'width': int(1000*x_data[1]/y_data[1]),
            'height': 1000,
            'images': [{
                'xref': 'x',
                'yref': 'y',
                'x': 0,
                'y': 0,
                'yanchor': 'bottom',
                'sizex': x_data[1]*scaling_factor,
                'sizey': y_data[1]*scaling_factor,
                # 'sizing': 'contain',
                'layer': 'below',
                'source': image
            }],
            'dragmode': 'select',
            'hovermode': 'closest'
        }
    }
    

    # return fig

    return figure
    # )
###################################
                    
                    
def crop_image(img):
    numpy_image = np.array(img)
    gray_ii = cv2.cvtColor(numpy_image,cv2.COLOR_RGB2GRAY)
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
    col_ii = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)[r1:r2, c1:c2]
    # Shrink for faster calculations
    var_ii = col_ii.var(axis=2)
    for kk in range(3):
        col_ii[:, :, kk] = np.where(var_ii > 5, col_ii[:, :, kk], col_ii[:, :, kk].max())
    return col_ii


###############################################


                    
@app.callback([Output('output-image-upload', 'figure'),
               Output('test-div', 'children')],
              [Input('upload-image', 'contents')]
              )

def update_output(contents):
    if contents is not None:

        
        datatype, string = contents[0].split(',')
        img = b64_to_pil(string)
        gray_ii = crop_image(img)
        return InteractiveImage(Image.fromarray(gray_ii)), 'asdf'
        
    else:
        pass

@app.callback([Output('zoom-window', 'figure'),
               Output('test-div-2', 'children')],
              [Input('output-image-upload', 'clickData'),
                Input('output-image-upload', 'figure')
                ])
# def update_zoom(zoom_location, image):
def update_zoom(zoom_location, img):
    
    if zoom_location is not None:
        scaling_factor = 1
        x = int(json.dumps(zoom_location['points'][0]['x']))
        y = int(json.dumps(zoom_location['points'][0]['y']))
        # image = Image.fromarray(np.array(json.loads(img), dtype = 'uint8'))
        image = img

        image['layout']['autosize'] = False
        image['layout']['xaxis'] = {
            'range': [x, x+500],
            'scale_ratio':1}
        image['layout']['yaxis'] = {
            'range': [y-500, y],
            'scale_ratio': 1,
            'autorange': False,
            'constrain': 'domain'
            }
        image['layout']['images'][0]['sizex'] = 3300
        image['layout']['images'][0]['sizey'] = 5000
        image['layout']['images'][0]['xref'] = 'x'
        image['layout']['images'][0]['yref'] = 'y'
        image['layout']['images'][0]['x'] = 0
        image['layout']['images'][0]['y'] = 0
        image['layout']['images'][0]['sizing'] = 'stretch'
        image['layout']['width'] = 1000
        image['layout']['height'] = 1000


                # }
                

        return image, 'asdf'

    else:
        pass

# @app.callback(Output('output-image-upload', 'figure'),
#               [Input('classifier-selector', 'value')])
# def change_evaluator(x):
#     pass
    


        
    
if __name__ == '__main__':
    app.run_server(debug=True)