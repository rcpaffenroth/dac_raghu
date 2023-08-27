---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,py:percent,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<a target="_blank" href="https://colab.research.google.com/github/rcpaffenroth/dac_raghu/blob/main/LunarLander.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# Setup and libraries


## Colab specific stuff

This is some code to make sure that the notebook works in Colab. It's not used when you are running things locally.

```python
import sys
IN_COLAB = 'google.colab' in sys.modules
```

```python
if IN_COLAB:
  ! apt-get install swig
  ! pip install stable-baselines3[extra] gymnasium[box2d] huggingface_sb3
  pass
else:
  # Otherwise, install locally and you need the following
  # NOTE: Need "gym" and "gymnasium" installed, since we use "gymnasium" for the LunarLander environment
  #       and "gym" is for huggingface_sb3.
  # sudo apt install swig ffmpeg
  # pip install stable-baselines3[extra] gymnasium[box2d] huggingface_sb3 imageio[ffmpeg] gym ipywidgets ipykernel pandas pyarrow
  pass
```

## Load the needed libraries

These are the libraries I will be using for this notebook

```python
import gymnasium as gym
import matplotlib.pylab as plt
import numpy as np

import imageio
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub

import pandas as pd

import pathlib

from IPython.display import display
from IPython.display import HTML
from ipywidgets import interact, widgets
from base64 import b64encode

import json
import gc
%matplotlib inline
```

# Parameters

```python
# For 16 trajectories, this takes about 40 seconds on a RTX 4090
number_of_trajectories = 1024
```

# The Lander environment.  

It is trying to land softly between the two flags

```python
# Make the environment
env = gym.make("LunarLander-v2", render_mode='rgb_array')
observation = env.reset()
```


There is the top level link to the library

https://gymnasium.farama.org/

Here is the Lunar Lander specific link

https://gymnasium.farama.org/environments/box2d/lunar_lander/




### Action Space
There are four discrete actions available:

0: do nothing

1: fire left orientation engine

2: fire main engine

3: fire right orientation engine




### Observation Space

The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.

```python
obs_names = ['x', 'y', 'vx', 'vy', 'theta', 'vtheta', 'leg1', 'leg2']
```

### Rewards

After every step a reward is granted. The total reward of an episode is the sum of the rewards for all the steps within that episode.

For each step, the reward:

is increased/decreased the closer/further the lander is to the landing pad.

is increased/decreased the slower/faster the lander is moving.

is decreased the more the lander is tilted (angle not horizontal).

is increased by 10 points for each leg that is in contact with the ground.

is decreased by 0.03 points each frame a side engine is firing.

is decreased by 0.3 points each frame the main engine is firing.

The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

An episode is considered a solution if it scores at least 200 points.


### Starting State

The lander starts at the top center of the viewport with a random initial force applied to its center of mass.


### Episode Termination

The episode finishes if:

the lander crashes (the lander body gets in contact with the moon);

the lander gets outside of the viewport (x coordinate is greater than 1);

the lander is not awake. From the Box2D docs, a body which is not awake is a body which doesn’t move and doesn’t collide with any other body:

When Box2D determines that a body (or group of bodies) has come to rest, the body enters a sleep state which has very little CPU overhead. If a body is awake and collides with a sleeping body, then the sleeping body wakes up. Bodies will also wake up if a joint or contact attached to them is destroyed.


# Training and downloading models

```python
models = {}
```

```python
class RandomModel(object):
  def __init__(self, env):
    self.env = env

  def predict(self, obs):
    return env.action_space.sample(), None # The second return value is the state value, which the random model does not use

random_model =  RandomModel(env)
models['random'] = {}
models['random']['model'] = random_model
models['random']['runs'] = []
```

```python
# This is an trained model that has a good architecture and loss function, but is not trained very much.  This takes about 30 sec on
# a RTX 4090
trained_model = PPO("MlpPolicy", env)
trained_model.learn(total_timesteps=20000)

models['trained'] = {}
models['trained']['model'] = trained_model
models['trained']['runs'] = []
```

```python
# This is a model from huggingface.co at https://huggingface.co/sb3/a2c-LunarLander-v2
# Mean reward: 181.08 +/- 95.35
checkpoint = load_from_hub(
    repo_id="sb3/a2c-LunarLander-v2",
    filename="a2c-LunarLander-v2.zip",
)

good_model = PPO.load(checkpoint)

models['good'] = {}
models['good']['model'] = good_model
models['good']['runs'] = []
```

```python
# This is a model from huggingface.co at https://huggingface.co/araffin/ppo-LunarLander-v2
# Mean reward:  283.49 +/- 13.74
checkpoint = load_from_hub(
    repo_id="araffin/ppo-LunarLander-v2",
    filename="ppo-LunarLander-v2.zip",
)

better_model = PPO.load(checkpoint)
models['better'] = {}
models['better']['model'] = better_model
models['better']['runs'] = []
```

# Evaluate models

```python
def evaluate_model(model_name, run_idx, models=models, env=env, movie=True):
   gc.collect()
   # Make a movie of a trained agent
   obs = env.reset()[0]

   # Get the model
   model = models[model_name]['model']
   images = []
   all_obs = []
   all_actions = []
   all_rewards = []
   done = False
   while not done:
      # This rendering mode puts an image into a numpy array
      images += [env.render()]
      action, _state = model.predict(obs)
      all_obs.append(obs)
      all_actions.append(int(action))
      obs, reward, done, trunc, info = env.step(action)
      all_rewards.append(reward)
   env.close()

   # Save the trajectory
   df = pd.DataFrame(all_obs, columns=obs_names)
   df['action'] = all_actions
   df['reward'] = all_rewards
   df.to_parquet(f'data/lander/{model_name}_{run_idx}_trajectory.parquet')

   if movie:
      # Save the movie
      imageio.mimsave(f'data/lander/{model_name}_{run_idx}_trajectory.mp4', images, fps=15)
```

```python

```

```python
pathlib.Path('data/lander').mkdir(exist_ok=True, parents=True)

info = {'models': ['random', 'trained', 'good', 'better'],
        'number_of_trajectories': number_of_trajectories,
}

json.dump(info, open('data/lander/info.json', 'w'))

for i in range(number_of_trajectories):
    print(f'Generating trajectory {i} of {number_of_trajectories}')
    evaluate_model('random', i)
    evaluate_model('trained', i)
    evaluate_model('good', i)
    evaluate_model('better', i)
```

# Basic visualization and analysis


## Rewards

```python
info = json.load(open('data/lander/info.json', 'r'))
for model_name in info['models']:
    total_rewards = []
    for run_idx in range(info['number_of_trajectories']):
        data = pd.read_parquet(f"data/lander/{model_name}_{run_idx}_trajectory.parquet")  
        # Print the total reward for each model
        total_rewards.append(np.sum(data['reward']))
    print(f"{model_name}: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
```


## Cool little visualization tool

```python
info = json.load(open('data/lander/info.json', 'r'))

@interact(model_name=info['models'], 
          run_idx=widgets.IntSlider(min=0, max=info['number_of_trajectories']-1, 
                                    step=1, value=0))
def show_video(model_name, run_idx):
      name = f'data/lander/{model_name}_{run_idx}_trajectory.mp4'
      mp4 = open(name,'rb').read()
      data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
      # This puts the video in the notebook
      display(HTML("""
      <video width=400 controls>
            <source src="%s" type="video/mp4">
      </video>
      """ % data_url))
      # plot various data for the run 
      _, ax = plt.subplots(1, 3)
      data = pd.read_parquet(f"data/lander/{model_name}_{run_idx}_trajectory.parquet")
      ax[0].plot(data['reward'])
      ax[0].set_title('reward')
      ax[1].plot(data['x'], data['y'])
      ax[1].set_title('position')
      ax[2].plot(data['vx'], data['vy'])
      ax[2].set_title('velocity')
      plt.tight_layout()
```
```python

```

