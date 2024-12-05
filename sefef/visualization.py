# -*- coding: utf-8 -*-
"""
sefef.visualization
-------------------

This is a helper module for visualization.

:copyright: (c) 2024 by Ana Sofia Carmo
:license: BSD 3-clause License, see LICENSE for more details.
"""

# third-party
import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go
import pandas as pd

COLOR_PALETTE = ['#4179A0', '#A0415D', '#44546A', '#44AA97', '#FFC000', '#0F3970', '#873C26']


def hex_to_rgba(h, alpha):
    '''Converts color value in hex format to rgba format with alpha transparency'''
    return tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha])

def _color_fader(prob, thr=0.5, ll='#FFFFC7', lh='#FFC900', hl='#FF9300', hh='#FF0000'):
    ''' Fade (interpolate) from color c1 to c2 with a non-linear transformation, according to the provided threshold.
    
    Parameters
    ---------- 
    ll_color, lh_color, hl_color, hh_color : any format supported by matplotlib, e.g., 'blue', '#FF0000'
    prob : float64
        Value between 0 and 1 corresponding to the probability of a seizure happening.
    thr : float64
        Value between 0 and 1 corresponding to the threshold 
        
    Returns
    -------
        A hex string representing the blended color.
    '''
    ll_color = np.array(mpl.colors.to_rgb(ll))
    lh_color = np.array(mpl.colors.to_rgb(lh))
    hl_color = np.array(mpl.colors.to_rgb(hl))
    hh_color = np.array(mpl.colors.to_rgb(hh))

    if prob <= thr:
        return mpl.colors.to_hex((1 - prob/thr) * ll_color + (prob/thr) * lh_color)
    else:
        return mpl.colors.to_hex((1 - ((prob-thr)/(1-thr))) * hl_color + ((prob-thr)/(1-thr)) * hh_color)
                          


def plot_forecasts(forecasts, ts, sz_onsets, high_likelihood_thr):
    ''' Provide visualization of forecasts
    
    Parameters
    ---------- 
    forecasts : array-like, shape (#forecasts, ), dtype "float64"
        Contains the predicted probabilites of seizure occurrence for the period with duration "forecast_horizon" and starting at the timestamps in "result2".
    ts : array-like, shape (#forecasts, ), dtype "int64"
        Contains the Unix timestamps, in seconds, for the start of the period for which the forecasts (in "result1") are valid. 
    sz_onsets : array-like, shape (#sz onsets, )
        Contains the unix timestamps (in seconds) of the onsts of seizures. 
    high_likelihood_thr : float64
        Value between 0 and 1 corresponding to the threshold of high-likelihood.
    ''' 
    fig = go.Figure()

    y_values = np.linspace(start=0, stop=1, num=100)
    y_values[np.argmin(np.abs(y_values - high_likelihood_thr))] = high_likelihood_thr

    for i in range(len(y_values) - 1):
        y0 = y_values[i]
        y1 = y_values[i + 1]
        color = _color_fader((y0 + y1) / 2, thr=high_likelihood_thr)  # Color for the midpoint of the interval
        fig.add_shape(
            type="rect",
            x0=0, x1=1,  # Full chart width (relative to 'paper')
            y0=y0, y1=y1,
            xref='paper', yref='y',
            fillcolor=color,
            line_width=0,  # No border
            layer="below",
        )
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(ts, unit='s'), y=forecasts, 
        mode='lines',
        line_color=COLOR_PALETTE[2],
        line_width=3
    ))

    fig.add_trace(go.Scatter(
        x=pd.to_datetime(sz_onsets, unit='s'), y=[(np.max(forecasts)-np.min(forecasts))/2] * len(sz_onsets),
        mode='text',
        text=['ϟ'] * len(sz_onsets),
        textfont=dict(
            size=16,
            color='white'  # Set the color of the Unicode text here
        ) 
    ))

    fig.add_hline(y=high_likelihood_thr, line_width=1, line_color='#FF0000')
    
    fig.update_yaxes(
        gridcolor='lightgrey',
        tickfont=dict(size=12),
        range=[0, np.min([1, np.max(forecasts)+0.1])],
    )
    fig.update_layout(
        title='Daily event likelihood',
        showlegend=False,
        plot_bgcolor='white')
    fig.show()


