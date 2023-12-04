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
- The maze is reset either when (i) the episode length has reached max timesteps (500) or (ii) the agent has reached its goal.

<!-- ## Reward Details
- Agent gets +1 for attaining the goal state.
- Agent gets -0.1/(maze_size[0]*maze_size[1]) penalty for every timestep that it doesn't achieve the goal state.
- So if the agent doesn't attain the goal state even once during an episode, the cumulative reward will be approx. penalty*n_steps*n_envs. If it attains the goal state exactly once, the cumulative reward will be approx. (penalty*n_steps*n_envs)+(number of times the goal was attained). -->

## Stats Recorded
- In log_agent#.csv file: For each agent, separate log files are recorded with the episode number and environment number. The number of timesteps when the agent reaches the goal state (since the last maze reset) is recorded.
<!-- should the penalty be explicitly removed at the end of the episode if atleast once the goal has been reached within the episode?-->
- In performance_agent#.csv file: For each agent, separate performance files are recorded with the episode number and the following metrics for that episode - cumulative reward, pi loss, value loss, entropy loss.

<!--add checkpointing frequency-->

## Current Functionality Summary

- Run train_agent#.py to train
- Models are saved into saved_models/agent#/{timestep}.pt
- Log data is saved into log_agent#.csv
- Performance data is saved into performance_agent#.csv
- Run plot_agent#.py to visualize performance the mean reward and loss graphs.
<!-- - test_agent#.py is incomplete -->