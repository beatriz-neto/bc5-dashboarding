# --------------------------------------------------  PACKAGES
from main import app
import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
from datetime import datetime
from dash import Input, Output, dcc, html

#-----------------------------------LAYOUT-------------------------

home_layout = dbc.Container([
    html.Div([
        html.Div([
            html.Img(src='/assets/home_page.png', height='60px'),
        ], style={'width': '13%'}),
        html.Div([
            html.H1('Home',
                    className='text-left mb-4'),
            ], style = {'width': '30%'})
], id = '3th row', style = {'display': 'flex', 'width': '50%'}),
    html.Hr(),

html.Div([
    html.Iframe(width="1100" ,height="60", src="https://rss.app/embed/v1/ticker/SyOSq6tyEWmV1RpA"),
    html.Iframe(width="1100" ,height="1600", src="https://rss.app/embed/v1/imageboard/f9amuUT8nQIpSWss")
])])


