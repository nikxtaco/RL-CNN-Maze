# import csv
# import matplotlib.pyplot as plt
# import numpy as np

# LOG_FILE = "log_agent0.csv"
# PERFORMANCE_FILE = "performance_agent0.csv"

# window_size = 20
# episode_rewards = {}

# with open(LOG_FILE, 'r') as csvfile:
#     log_reader = csv.reader(csvfile)
#     next(log_reader)  # skip header
#     for row in log_reader:
#         episode = int(row[0])
#         reward = float(row[3])

#         # Calculate the starting and ending episodes for the current data point
#         start_episode = (episode // window_size) * window_size
#         end_episode = start_episode + window_size
#         print(f'start: {start_episode} and end: {end_episode}')

#         # Check if the episode is within the specified range
#         if start_episode <= episode < end_episode:
#             if start_episode in episode_rewards:
#                 episode_rewards[start_episode].append(reward)
#             else:
#                 episode_rewards[start_episode] = [reward]

# # Calculate mean rewards for each episode within the specified range
# mean_episodes = sorted(episode_rewards.keys())
# mean_rewards = [np.mean(episode_rewards[episode]) for episode in mean_episodes]

# plt.plot(mean_episodes, mean_rewards, label=f'Mean Rewards (Episodes {mean_episodes[0]}-{mean_episodes[-1] + window_size-1})')
# plt.xlabel('Episodes')
# plt.ylabel('Mean Rewards')
# plt.legend()
# plt.show()


import csv
import matplotlib.pyplot as plt
import numpy as np

LOG_FILE = "log_agent0.csv"
PERFORMANCE_FILE = "performance_agent0.csv"

window_size = 250
episode_rewards = {}

with open(PERFORMANCE_FILE, 'r') as csvfile:
    performance_reader = csv.reader(csvfile)
    next(performance_reader)  # skip header
    for row in performance_reader:
        episode = int(row[0])
        reward = float(row[1])

        # Calculate the starting and ending episodes for the current data point
        start_episode = (episode // window_size) * window_size
        end_episode = start_episode + window_size

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

episode_pi_loss = {}

with open(PERFORMANCE_FILE, 'r') as csvfile:
    performance_reader = csv.reader(csvfile)
    next(performance_reader)  # skip header
    for row in performance_reader:
        episode = int(row[0])
        pi_loss = float(row[2])

        # Calculate the starting and ending episodes for the current data point
        start_episode = (episode // window_size) * window_size
        end_episode = start_episode + window_size

        # Check if the episode is within the specified range
        if start_episode <= episode < end_episode:
            if start_episode in episode_pi_loss:
                episode_pi_loss[start_episode].append(pi_loss)
            else:
                episode_pi_loss[start_episode] = [pi_loss]

# Calculate mean pi_loss for each episode within the specified range
mean_episodes = sorted(episode_pi_loss.keys())
mean_pi_loss = [np.mean(episode_pi_loss[episode]) for episode in mean_episodes]

plt.plot(mean_episodes, mean_pi_loss, label=f'Mean Pi Loss (Episodes {mean_episodes[0]}-{mean_episodes[-1] + window_size-1})')
plt.xlabel('Episodes')
plt.ylabel('Mean Pi_loss')
plt.legend()
plt.show()


episode_value_loss = {}

with open(PERFORMANCE_FILE, 'r') as csvfile:
    performance_reader = csv.reader(csvfile)
    next(performance_reader)  # skip header
    for row in performance_reader:
        episode = int(row[0])
        value_loss = float(row[3])

        # Calculate the starting and ending episodes for the current data point
        start_episode = (episode // window_size) * window_size
        end_episode = start_episode + window_size

        # Check if the episode is within the specified range
        if start_episode <= episode < end_episode:
            if start_episode in episode_value_loss:
                episode_value_loss[start_episode].append(value_loss)
            else:
                episode_value_loss[start_episode] = [value_loss]

# Calculate mean value_loss for each episode within the specified range
mean_episodes = sorted(episode_value_loss.keys())
mean_value_loss = [np.mean(episode_value_loss[episode]) for episode in mean_episodes]

plt.plot(mean_episodes, mean_value_loss, label=f'Mean Value Loss (Episodes {mean_episodes[0]}-{mean_episodes[-1] + window_size-1})')
plt.xlabel('Episodes')
plt.ylabel('Mean value_loss')
plt.legend()
plt.show()


episode_entropy_loss = {}

with open(PERFORMANCE_FILE, 'r') as csvfile:
    performance_reader = csv.reader(csvfile)
    next(performance_reader)  # skip header
    for row in performance_reader:
        episode = int(row[0])
        entropy_loss = float(row[4])

        # Calculate the starting and ending episodes for the current data point
        start_episode = (episode // window_size) * window_size
        end_episode = start_episode + window_size

        # Check if the episode is within the specified range
        if start_episode <= episode < end_episode:
            if start_episode in episode_entropy_loss:
                episode_entropy_loss[start_episode].append(entropy_loss)
            else:
                episode_entropy_loss[start_episode] = [entropy_loss]

# Calculate mean entropy_loss for each episode within the specified range
mean_episodes = sorted(episode_entropy_loss.keys())
mean_entropy_loss = [np.mean(episode_entropy_loss[episode]) for episode in mean_episodes]

plt.plot(mean_episodes, mean_entropy_loss, label=f'Mean Entropy Loss (Episodes {mean_episodes[0]}-{mean_episodes[-1] + window_size-1})')
plt.xlabel('Episodes')
plt.ylabel('Mean entropy_loss')
plt.legend()
plt.show()
