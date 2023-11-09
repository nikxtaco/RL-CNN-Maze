## Setup

Install required packages with:
```bash
pip install gymnasium matplotlib opencv-python pygame stable_baselines3
```
To set up the maze environment run:
```bash
python setup.py install
```

## Current Functionality Summary

- run train_agent#.py to train
- models are saved into saved_models/agent#/model_{timestep}.pt
- logs and performance data are saved into performance_agent#.csv
<!-- - run cnn_plot_agent#.py to plot performance graph #todo: mean rewards -->
<!-- - cnn_test.py is incomplete -->