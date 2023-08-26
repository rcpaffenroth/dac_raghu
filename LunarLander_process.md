---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,py:percent
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Setup and libraries


## Load the needed libraries

These are the libraries I will be using for this notebook

```python
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import json

%matplotlib inline
```

# Write files

```python
# This section produces the data for a generative model of the Lunar Lander
# Create a single dataframe with all the data
# each row is a single run of a single model
# each column is a single timestep of a single variable

# NOTE:  There needs to be some thinking here.  I mean, while the position,
# velocity, and angle are all continuous, the thrust is not.  So, we need to
# thinkg about how to interpolate the thrust. I think the data in this case
# needs to be "ragged" in the sense that each row has a different number of
# entries.  However, perhaps we can also just look at the "shortest" run and
# truncate all the other runs to that length. 


# TODO:  This is getting close, but is not there yet.  I want things like
# 'x,x,x' to be something like 'x1,x2,x3' so that I can use the autoencoder
# more easily.  Is that a matter of combining the columns?  I think so.  
# How about keeping a dict to map times to indices?  That would work I think.

def uniform_data_for_autoencoder(info, entries_per_run=100):
    all_data = []
    for model_name in info['models']:
        for run_idx in range(info['number_of_trajectories']):
            df = pd.read_parquet(f'data/lander/{model_name}_{run_idx}_trajectory.parquet')  

            # index plays the role of timestep
            df['timestamp'] = pd.to_datetime(df.index, unit='s')
            df['idx'] = df.index
            df.set_index('timestamp', inplace=True)

            # We now compute the delta t that gives us 100 total sample points for each run
            # We do this by taking the total time of the run and dividing by 100
            total_time = df.index[-1] - df.index[0]
            delta_t = total_time / entries_per_run
            df = df.resample(delta_t).interpolate()

            df = pd.melt(df, 
                        value_vars=['x', 'y', 'vx', 'vy', 'theta', 'vtheta'], 
                        var_name='variable', 
                        ignore_index=False, 
                        value_name='value')
            # How to add a few additional rows to the dataframe
            df.loc[df.index[0]] = ['model_name', model_name]
            df.loc[df.index[-1]] = ['total_time', total_time]
            all_data.append(df)

    # for i,df in enumerate(all_data):
    #     if i == 0:
    #         all_data = pd.DataFrame(df).T
    #     df['run_idx'] = i    
    # all_data = pd.concat(all_data)
    # all_data.to_parquet(filename)
    return all_data
info = json.load(open('data/lander/info.json', 'r'))
all_data = uniform_data_for_autoencoder(info)
```

```python
entries_per_run=100
df = pd.read_parquet(f'data/lander/better_0_trajectory.parquet')  

# index plays the role of timestep
df['timestamp'] = pd.to_datetime(df.index, unit='s')
df.set_index('timestamp', inplace=True)

# We now compute the delta t that gives us 100 total sample points for each run
# We do this by taking the total time of the run and dividing by 100
total_time = df.index[-1] - df.index[0]
delta_t = total_time / entries_per_run
df = df.resample(delta_t).interpolate()
multi_index = pd.MultiIndex.from_arrays([np.arange(len(df)), df.index], names=('idx', 'timestamp'))
df.index = multi_index
df
```

```python
df = pd.melt(df, 
            value_vars=['x', 'y', 'vx', 'vy', 'theta', 'vtheta'], 
            var_name='variable', 
            ignore_index=False, 
            value_name='value',
            col_level=0)
df
```

```python
df.T
```

```python

```
