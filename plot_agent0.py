import csv
import matplotlib.pyplot as plt
import numpy as np

LOG_FILE = "log_agent0.csv"
PERFORMANCE_FILE = "performance_agent0.csv"

window_size = 5
episode_rewards = {}

with open(LOG_FILE, 'r') as csvfile:
    log_reader = csv.reader(csvfile)
    next(log_reader)  # skip header
    for row in log_reader:
        episode = int(row[0])
        reward = float(row[3])

        # Calculate the starting and ending episodes for the current data point
        start_episode = (episode // window_size) * window_size
        end_episode = start_episode + window_size
        print(f'start: {start_episode} and end: {end_episode}')

        # Check if the episode is within the specified range
        if start_episode <= episode < end_episode:
            if start_episode in episode_rewards:
                episode_rewards[start_episode].append(reward)
            else:
                episode_rewards[start_episode] = [reward]

# Calculate mean rewards for each episode within the specified range
mean_episodes = sorted(episode_rewards.keys())
mean_rewards = [np.mean(episode_rewards[episode]) for episode in mean_episodes]

plt.plot(mean_episodes, mean_rewards, label=f'Mean Rewards (Episodes {mean_episodes[0]}-{mean_episodes[-1] + window_size-1})')
plt.xlabel('Episodes')
plt.ylabel('Mean Rewards')
plt.legend()
plt.show()
