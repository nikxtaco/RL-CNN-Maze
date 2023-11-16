## Setup

Install required packages with:
```bash
pip install gymnasium matplotlib opencv-python pygame stable_baselines3
```
To set up the maze environment run:
```bash
python setup.py install
```

## Environment Details
- One agent attempts to get to the fixed end of the maze (green) while the other attempts to get to the randomly spawning object (red).
- Each env generated per agent, goes through a series of mazes that are identical across the two agents.
- The maze is only reset when the agent has reached its goal.
<!--- The maze is reset either when (i) the episode has ended or (ii) the agent has reached its goal. -->

## Stats Recorded
- In log_agent#.csv file: For each agent, separate log files are recorded with the episode number and environment number. The number of timesteps when the agent reaches the goal state (since the last maze reset) is recorded.
<!-- should the penalty be explicitly removed at the end of the episode if atleast once the goal has been reached within the episode?-->
- In performance_agent#.csv file: For each agent, separate performance files are recorded with the episode number and the following metrics for that episode - pi loss, value loss, entropy loss.

## Current Functionality Summary

- run train_agent#.py to train
- models are saved into saved_models/agent#/model_{timestep}.pt
- log data is saved into log_agent#.csv
- performance data is saved into performance_agent#.csv
<!-- - run cnn_plot_agent#.py to plot performance graph #todo: mean rewards -->
<!-- - cnn_test.py is incomplete -->
<!-- yet to check equivalence of environments, attempt render side by side?-->