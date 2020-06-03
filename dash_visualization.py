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

dir_base = os.getcwd()


    
app = dash.Dash()

@profile
def the_whole_thing():  
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
    
        dcc.Graph(
            id = 'output-image-upload'
            ),
        dcc.Graph(
            id = 'zoom-window'
            ),
        html.Div(
            id = 'test-div')
        
        ]
        )
    
    
    
    ##################################
        
    def b64_to_pil(string):
        decoded = base64.b64decode(string)
        buffer = BytesIO(decoded)
        im = Image.open(buffer)
        return im
    
    ##################################
    
    def take_crops(img):
        n_crops = 1
        image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
        crop_locations = pd.DataFrame(columns = ['x','y'])
        np.random.seed(0)
        counter = 0
        test_counter = 0
        crop_size = 500
        height = img.size[1]
        width = img.size[0]
        while counter < n_crops:
            test_counter += 1
            yidx = np.random.choice(np.arange(height - crop_size))
            xidx = np.random.choice(np.arange(width - crop_size))
            crop = image[yidx:(yidx+crop_size+1),xidx:(xidx+crop_size+1)]
            if np.mean(crop)/255.0 <= .95:
                crop_locations.loc[counter] = [xidx, yidx]
                counter += 1
            else:
                counter = counter
    
                
            # crop_locations.loc[counter] = [xidx, yidx]
            # counter += 1
        crop_locations.loc[counter+1] = [width, height]
        crop_locations.loc[counter+2] = [0, test_counter]
        return crop_locations
                
            
            
            
    
    #######################
    
    def InteractiveImage(image):
        #set up dummy graph the size of the image
        x_data = np.array([0, image.size[0]])
        y_data = np.array([0, image.size[1]])
        scaling_factor = 1
        
        crop_locations = take_crops(image)
        
        
        # return dcc.Graph(
            # id='main-image',
        figure={
            'data': [
                {
                    'x': crop_locations.x*scaling_factor,
                    'y': crop_locations['y']*scaling_factor,
                    'name': 'dummy_trace',
                    'mode': 'markers',
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
                    'sizing': 'contain',
                    'layer': 'below',
                    'source': image
                }],
                'dragmode': 'select',
                'hovermode': 'closest'
            }
        }
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
    def zoom_image(image):
        pass
        return dcc.graph(
            id = 'zoom',
            figure = {
                })
    
    
                        
    @app.callback([Output('output-image-upload', 'figure'),
                    Output('memory', 'data')],
                  [Input('upload-image', 'contents')]
                  )
    
    def update_output(contents):
        if contents is not None:
    
            
            datatype, string = contents[0].split(',')
            img = b64_to_pil(string)
            gray_ii = crop_image(img)
            
    
            
            return InteractiveImage(Image.fromarray(gray_ii)), json.dumps(np.array(gray_ii).tolist())
            # return InteractiveImage(Image.fromarray(gray_ii)), gray_ii
        
    # @app.callback(Output('zoom-window', 'figure'),
    #               [Input('output-image-upload', 'clickData'),
    #                Input('memory', 'data')])
    @app.callback([Output('test-div', 'children'),
                   Output('zoom-window', 'figure')],
                  [Input('output-image-upload', 'clickData'),
                   Input('memory', 'data')
                   ])
    # def update_zoom(zoom_location, image):
    def update_zoom(zoom_location, img):
        
        if zoom_location is not None:
            scaling_factor = 1
            x = int(json.dumps(zoom_location['points'][0]['x']))
            y = int(json.dumps(zoom_location['points'][0]['y']))
            image = Image.fromarray(np.array(json.loads(img), dtype = 'uint8'))
            figure ={
                'layout': {
                    'autosize': False,
                    'margin': go.Margin(l = 40, b= 40, t = 26, r= 10),
                    'xaxis': {
                        'range': [x, x+500],
                        # 'scaleanchor': 'y',
                        'scaleratio': scaling_factor,
                        
                    },
                    'yaxis': {
                        'range': [y, y+500],
                        # 'scaleanchor': 'x',
                        'scaleratio': scaling_factor,
                        # 'dticks': 100,
                        'autorange': False,
                        # 'rangemode': 'tozero',
                        'constrain': 'domain'
                        # 'fixedrange': True
                    },
                    'width': 500,
                    'height': 500,
                    'images': [{
                        'xref': 'x',
                        'yref': 'y',
                        'x': 0,
                        'y': 0,
                        'yanchor': 'bottom',
                        'sizex': image.size[0],
                        'sizey': image.size[1],
                        'sizing': 'stretch',
                        'layer': 'below',
                        'source': image
                    }],
                    'dragmode': 'select',
                    'hovermode': 'closest'
                }
                }
            test_figure = {
                'data': [{
                    'x': [1,2,3],
                    'y': [1,2,3],
                    'type': 'bar'
                    }],
                'layout': {
                    'title': 'testy mctest graph'}
    
                    }
            return str([x,y]),figure
                
    
the_whole_thing()
    


        
    
if __name__ == '__main__':
    app.run_server(debug=True)