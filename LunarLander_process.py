# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Setup and libraries

# ## Load the needed libraries
#
# These are the libraries I will be using for this notebook

# +
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import json

# %matplotlib inline
# -

# # Write files

# +
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
            # There is a nice way to resample the data in pandas, but it requires a datetime index
            df['timestamp'] = pd.to_datetime(df.index, unit='s')
            df.set_index('timestamp', inplace=True)

            # However, we just want the number of seconds since the start of the run
            # so we just keep that as a column
            df['time_seconds'] = (df.index - df.index[0]).total_seconds()

            # We now compute the delta t that gives us 100 total sample points for each run
            # We do this by taking the total time of the run and dividing by 100
            total_time = df.index[-1] - df.index[0]
            delta_t = total_time / entries_per_run
            df = df.resample(delta_t).interpolate()

            # Ok, now things are resampled and interpolated, but we need to get rid of the
            # datetime index and replace it with a simple integer index and the number of seconds.
            #multi_index = pd.MultiIndex.from_arrays([np.arange(len(df))], names=('idx',))
            df.index = np.arange(len(df))

            # Melt makes a mutli-column dataframe into a single column dataframe (well, actually
            # a pair of columns, one for the variable name and one for the value).  
            df_melt = pd.melt(df, 
                  value_vars=['x', 'y', 'vx', 'vy', 'theta', 'vtheta', 'time_seconds'], 
                  var_name='parameter', 
                  ignore_index=False, 
                  value_name=(model_name, run_idx),
                  col_level=0)

            # We now have a dataframe with a single column, but we want to make the index
            # better for later slicing.  In particular, we want to make the index a multi-index
            # with the first index being the row number and the second index being the parameter
            # name.  This will make it easy to slice out all the x values, for example.
            df_melt.index = pd.MultiIndex.from_arrays([df_melt.index, df_melt['parameter']],names=('idx', 'parameter'))
            df_melt.drop(columns=['parameter'], inplace=True)

            # We now have a dataframe with a single column, but we want each experiment to be
            # a single row.  
            experiment = df_melt.T

            # Last but not least we want to add the model name and run index to the dataframe
            experiment.index = pd.MultiIndex.from_tuples(experiment.index, names=('run_idx', 'experiment'))
            
            all_data.append(experiment)
    all_data = pd.concat(all_data)
    return all_data
info = json.load(open('data/lander/info.json', 'r'))
all_data = uniform_data_for_autoencoder(info)
# -

all_data.to_parquet('data/lander/all_data.parquet')

# Example of slicing out x,y values for time stepss 1..4 for all the runs of all the models
all_data.loc[:, (range(1,5),('x','y'))]


#  Just the better runs, but all the x values
all_data.loc[('better', slice(None)), (slice(None),('x',))]


