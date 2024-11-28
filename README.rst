Welcome to ``SeFEF``
======================

.. image:: https://raw.githubusercontent.com/anascacais/sefef/main/docs/logo/sefef-logo.png
    :align: center
    :alt: SeFEF logo

|

``SeFEF`` is a Seizure Forecast Evaluation Framework written in Python.
The framework standardizes the development, evaluation, and reporting of individualized algorithms for seizure likelihood forecast. 
``SeFEF`` aims to decrease development time and minimize implementation errors by automating key procedures within data preparation, training/testing, and computation of evaluation metrics. 

Highlights:
-----------

- ``evaluation`` module: implements time series cross-validation.
- ``labeling`` module: automatically labels samples according to the desired pre-ictal duration and prediction latency.
- ``postprocessing`` module: processes individual predicted probabilities into a unified forecast according to the desired forecast horizon.
- ``scoring`` module: computes both deterministic and probabilistic metrics according to the horizon of the forecast.  



Installation
------------

Installation can be easily done with ``pip``:

.. code:: bash

    $ pip install sefef

Simple Example
--------------

The code below loads the metadata from an existing dataset from the ``examples`` folder, create a ``Dataset`` instance, and creates an adequate split for a time series cross-validation.

.. code:: python

    import json
    import pandas as pd
    from sefef import evaluation

    # read example files
    files_metadata = pd.read_csv('examples/files_metadata.csv')
    with open('examples/sz_onsets.txt', 'r') as f:
         sz_onsets = json.load(f)
   
    # create Dataset instance and perform TSCV
    dataset = evaluation.Dataset(files_metadata, sz_onsets, sampling_frequency=128)
    tscv = evaluation.TimeSeriesCV()
    tscv.split(dataset, iteratively=False, plot=True)
