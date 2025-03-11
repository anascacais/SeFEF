# third-party
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# sefef
from sefef.visualization import COLOR_PALETTE

polar_layout = {
    'bgcolor': 'white',
    'angularaxis': {'showgrid': True, 'gridcolor': 'lightgrey'},
    'radialaxis': {'showticklabels': True, 'gridcolor': 'lightgrey'},
    'barmode': 'overlay',
}


def plot_event_phase_dist(bin_edges, event_counts, sample_counts, pdf_func, cycles_duration):
    ''' Polar plot of events' phase distribution for the provided cycles. Plots both a density histogram (bin frequency divided by the bin width, so that the area under the histogram integrates to 1 (np.sum(density * np.diff(bins)) == 1)) and the provided PDF function.

    Parameters
    ---------- 
    bin_edges : array-like, shape (#num_bins+1, )
        Bin edges (in radians).
    freq : list of array-like, shape (#num_bins,)
        Bin's event count divided by the total number of counts.
    pdf_func : list of function
        Funtions that receives an array of phases (in radians) and returns the corresponding PDF.
    cycle_duration : list of str
        Duration of cycles as it should appear in plot's title (e.g. "30 days").
    '''

    fig = make_subplots(
        rows=1, cols=len(cycles_duration),
        subplot_titles=cycles_duration,
        specs=[[{'type': 'polar'}]*len(cycles_duration)]*1)

    for i in range(len(cycles_duration)):
        fig.add_trace(_get_frequency_hist_samples(bin_edges[i], sample_counts[i]/np.amax(sample_counts[i]), i),
                      row=1, col=i+1)
        fig.add_trace(_get_frequency_hist_events(bin_edges[i], event_counts[i]/np.amax(sample_counts[i]), i),
                      row=1, col=i+1)
        fig.add_trace(_get_polar_dist(pdf_func[i], i),
                      row=1, col=i+1)

        p = ['hour', 'day']['day' in cycles_duration[i]]
        polar_layout['angularaxis']['tickvals'] = np.rad2deg(bin_edges[i][:-1])
        polar_layout['angularaxis']['ticktext'] = [
            f'{p} {period}' for period in range(1, len(bin_edges[i][:-1])+1)]
        fig.update_layout({f"polar{i+1}": polar_layout})
        fig.layout.annotations[i].update(y=1.03)

    fig.update_layout(
        title_text="Event phase distribution for significant cycles")

    fig.show()


def _get_frequency_hist_events(bin_edges, counts, i=0):
    # density = freq / np.diff(bin_edges)
    return go.Barpolar(
        theta=np.rad2deg((bin_edges[:-1] + bin_edges[1:]) / 2),
        r=counts,
        width=np.rad2deg(np.diff(bin_edges)),
        name='Histogram seizures',
        marker_color=COLOR_PALETTE[0],
        legendgroup='freq_events',
        showlegend=not bool(i),
    )


def _get_frequency_hist_samples(bin_edges, counts, i=0):
    # density = freq / np.diff(bin_edges)
    return go.Barpolar(
        theta=np.rad2deg((bin_edges[:-1] + bin_edges[1:]) / 2),
        r=counts,
        width=np.rad2deg(np.diff(bin_edges)),
        name='Histogram samples',
        marker_color=COLOR_PALETTE[0],
        opacity=0.2,
        legendgroup='freq_samples',
        showlegend=not bool(i),
    )


def _get_polar_dist(pdf_func, i=0):
    x = np.linspace(0, 2*np.pi, num=501)
    pdf = pdf_func(x)
    return go.Scatterpolar(
        theta=np.rad2deg(x),
        r=pdf,
        name='Von Mises PDF (approx.)',
        line_color=COLOR_PALETTE[1],
        legendgroup='group2',
        showlegend=not bool(i),
    )
