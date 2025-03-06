# -*- coding: utf-8 -*-
"""
sefef.visualization
-------------------

This is a helper module for visualization.

:copyright: (c) 2024 by Ana Sofia Carmo
:license: BSD 3-clause License, see LICENSE for more details.
"""
# built-in
import os
from datetime import datetime

# third-party
from plotly.subplots import make_subplots
import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express.colors as pc
import plotly.express as px


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


def plot_forecasts(forecasts, ts, sz_onsets, high_likelihood_thr, forecast_horizon, title='Seizure probability', folder_path=None, filename=None, show=True, return_plot=False, n_points=100):
    ''' Provide visualization of forecasts.

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

    # get only sz_onsets for which there are forecasts
    ts_forecast_end = ts + forecast_horizon
    sz_onsets = sz_onsets[np.logical_and((sz_onsets[:, np.newaxis] >= ts[np.newaxis, :]),
                                         (sz_onsets[:, np.newaxis] < ts_forecast_end[np.newaxis, :])).any(axis=1)]
    sz_onsets_forecasts_ind = np.argwhere(np.logical_and(
        (sz_onsets[:, np.newaxis] >= ts[np.newaxis, :]), (sz_onsets[:, np.newaxis] < ts_forecast_end[np.newaxis, :])))[:, 1]

    # Draw background
    color_gradient = px.colors.sequential.RdBu[1:-1][::-1]
    color_gradient.remove('rgb(247,247,247)')
    low_prob_values = np.linspace(0, high_likelihood_thr, int(n_points/2))
    high_prob_values = np.linspace(high_likelihood_thr, 1, int(n_points/2))
    low_prob_colors = [color_gradient[int(
        val / high_likelihood_thr * (len(color_gradient)//2 - 1))] for val in low_prob_values]
    high_prob_colors = [color_gradient[-len(color_gradient)//2:][int((val - high_likelihood_thr) / (1 - high_likelihood_thr) * (len(color_gradient)//2 - 1))]
                        for val in high_prob_values]

    fig = _get_gradient(fig, low_prob_values, low_prob_colors, x0=pd.to_datetime(ts[0], unit='s'), x1=pd.to_datetime(
        ts_forecast_end[-1], unit='s'))
    fig = _get_gradient(fig, high_prob_values, high_prob_colors, x0=pd.to_datetime(ts[0], unit='s'), x1=pd.to_datetime(
        ts_forecast_end[-1], unit='s'))

    df_start = pd.DataFrame({'ts': ts, 'forecasts': forecasts})
    df_end = df_start.copy()
    df_end['ts'] += forecast_horizon - 1
    df = pd.concat([df_start, df_end]).sort_values('ts').reset_index(drop=True)

    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df['ts'], unit='s'), y=df['forecasts'],
        mode='lines',
        line_color=COLOR_PALETTE[2],
        line_width=3,
        fill='tozeroy',
        fillcolor='white'
    ))

    fig.add_trace(go.Scatter(
        x=pd.to_datetime(sz_onsets, unit='s'), y=forecasts[sz_onsets_forecasts_ind],
        mode='text',
        text=['ϟ'] * len(sz_onsets),
        textfont=dict(
            size=25,
            color=COLOR_PALETTE[1]
        )
    ))

    fig.add_hline(y=high_likelihood_thr, line_width=1, line_color='#FF0000')

    non_nan_forecasts = forecasts[~np.isnan(forecasts)]
    fig.update_yaxes(
        title='Probability',
        gridcolor='lightgrey',
        tickfont=dict(size=12),
        range=[np.max([0, np.min(non_nan_forecasts) - np.std(non_nan_forecasts)]),
               np.min([1, np.max(non_nan_forecasts) + np.std(non_nan_forecasts)])],
    )
    fig.update_xaxes(
        title='Time',
    )
    fig.update_layout(
        title=title,
        showlegend=False,
        plot_bgcolor='white',)

    if folder_path is not None:
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        fig.write_image(os.path.join(folder_path, filename))

    if show:
        fig.show()

    if return_plot:
        return fig


def _get_gradient(fig, values, colors, x0, x1):
    for i in range(len(colors)-1):
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=values[i], y1=values[i+1],
            fillcolor=colors[i],
            yref='y',
            line_width=0,  # No border
            layer="below",)
    return fig


def aggregate_plots(figs, folder_path=None, filename=None, show=True,):
    ''' Receives go.Figure objects created using "plot_forecasts" and aggregates them into a single Figure.

    Parameters
    ---------- 
    figs : go.Figure
        Figures to aggregate into a single plot. 
    '''
    fig = make_subplots(rows=1, cols=len(
        figs), shared_yaxes=True, horizontal_spacing=0.005)
    forecasts = []
    for ifig, figure in enumerate(figs):
        print(f'Aggregating forecast plots ({ifig+1}/{len(figs)})', end='\r')
        x0, x1 = datetime.max, datetime.min
        for trace in figure.data:
            fig.add_trace(trace, row=1, col=(ifig+1))
            x0, x1 = min(x0, min(trace['x'])), max(x1, max(trace['x']))
        for shape in figure.layout.shapes or []:
            new_shape = shape.to_plotly_json()
            new_shape["xref"], new_shape["yref"] = f"x{ifig+1}", f"y{ifig+1}"
            fig.add_shape(new_shape)
        for annotation in figure.layout.annotations or []:
            annotation["xref"], annotation["yref"] = f"x{ifig+1}", f"y{ifig+1}"
            fig.add_annotation(annotation)
        fig.update_xaxes(
            range=[x0, x1], row=1, col=(ifig+1))
        fig.update_yaxes(
            showticklabels=(ifig == 0),
            row=1, col=(ifig+1))
        forecasts += [figure['data'][0]['y']]

    # Update layout
    forecasts = np.concat(forecasts)
    fig.update_layout(
        title_text=figure['layout']['title']['text'],
        showlegend=False, plot_bgcolor='white',
        yaxis_title='Probability',
        annotations=[
            dict(
                text="Time",
                x=0.5, y=-0.15,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14)
            )],)
    fig.update_yaxes(
        range=[np.max([0, np.min(forecasts) - np.std(forecasts)]),
               np.min([1, np.max(forecasts) + np.std(forecasts)])])

    if folder_path is not None:
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        fig.write_image(os.path.join(folder_path, filename))

    if show:
        fig.show()


def html_modelcard_formating(contents):
    '''Courtesy of ChatGPT'''
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        /* Resetting some default browser styles */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        /* A4 page formatting */
        @page {{
            size: A4;
            margin: 20mm;
        }}

        body {{
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
            line-height: 1.6;
            font-size: 12pt;
            color: #333333;
            background-color: #f9f9f9;
            padding-top: 10mm;
            padding-left: 20mm;
            padding-right: 20mm;
        }}

        h2, h3, h4 {{
            margin-top: 10mm;
            font-weight: 700;
            color: #2c3e50;
        }}

        h2 {{
            font-size: 18pt;
            border-bottom: 2px solid #ccc;
            padding-bottom: 5px;
        }}

        h3 {{
            font-size: 16pt;
            margin-top: 8mm;
            border-bottom: 1px solid #ccc;
            padding-bottom: 5px;
        }}

        h4 {{
            font-size: 14pt;
            margin-top: 5mm;
            color: #34495e;
        }}

        ul {{
            padding-left: 20px;
            margin-top: 5mm;
        }}

        li {{
            margin-bottom: 5mm;
            font-size: 12pt;
        }}

        p {{
            margin-top: 5mm;
            font-size: 12pt;
            line-height: 1.6;
        }}

        /* Styling for bullet points */
        ul li {{
            list-style-type: disc;
        }}

        /* Add some padding and margins to structure content */
        .content-wrapper {{
            margin-bottom: 20mm;
        }}

        .content-wrapper h2 {{
            margin-top: 10mm;
        }}

        /* Page-specific styles */
        @media print {{
            body {{
                width: 210mm;
                height: 297mm;
                padding: 20mm;
                margin: 0;
            }}

            h2 {{
                font-size: 16pt;
                page-break-before: always;
            }}

            h3 {{
                font-size: 14pt;
            }}

            ul {{
                list-style-type: disc;
            }}

            /* Ensure content is spaced out correctly across pages */
            .content-wrapper {{
                page-break-inside: avoid;
                margin-bottom: 10mm;
            }}

            .content-wrapper:last-child {{
                page-break-after: auto;
            }}
        }}
    </style>
</head>
<body>
    <div class="content-wrapper">
        {contents}
    </div>
</body>
</html>
"""
