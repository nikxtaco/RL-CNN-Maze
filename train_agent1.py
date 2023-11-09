import os
import torch
import gymnasium as gym
import gym_maze
from common.storage import Storage
from common.model import ImpalaModel
from common.policy import CategoricalPolicy
from agents.ppo import PPO as AGENT
from stable_baselines3.common.vec_env import DummyVecEnv


def make_env(rank):
    def init():
        env = gym.make("maze-random-10x10-v0", seed=42+rank, target=1) # Set target to random location
        return env
    return init

if __name__=='__main__':
    MAX_T = 2000
    N_ENVS = 2
    N_STEPS = 500
    RENDER_MAZE = True # Make False
    CHECKPOINT_PATH = 'saved_models/agent1/model_'
    LOAD_FROM_TIMESTEP = str(1000)
    PERFORMANCE_FILE = "performance_agent1.csv"
    N_CHECKPOINTS = 2

    # Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Make Vectorized Env
    env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])

    # Set Env Spaces
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    in_channels = observation_shape[2]
    action_space = env.action_space

    # Model & Policy
    model = ImpalaModel(in_channels=in_channels).to(device)
    if isinstance(action_space, gym.spaces.Discrete):
        recurrent = False
        action_size = action_space.n
        policy = CategoricalPolicy(model, recurrent, action_size)
    else:
        raise NotImplementedError
    policy.to(device)

    # Initialize Storage
    print('Initializing Storage...')
    hidden_state_dim = model.output_dim
    storage = Storage(observation_shape, hidden_state_dim, N_STEPS, N_ENVS, device=device)
    storage_valid = Storage(observation_shape, hidden_state_dim, N_STEPS, N_ENVS, device=device)
    print(f'Storage Valid...:{storage_valid}')

    agent = AGENT(env, policy, logger=None, storage=storage, device=device, n_checkpoints=N_CHECKPOINTS, n_steps=N_STEPS, n_envs=N_ENVS)
    
    # Load Model
    if os.path.exists(CHECKPOINT_PATH + LOAD_FROM_TIMESTEP + '.pt'):
        print("Loading file...")
        checkpoint = torch.load(CHECKPOINT_PATH + LOAD_FROM_TIMESTEP + '.pt')
        agent.policy.load_state_dict(checkpoint['model_state_dict']) #, map_location=torch.device(device))
        print("Model policy loaded.")
        print("Loading file...")
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) #, map_location=torch.device(device))
        print("Model optimizer loaded.")

    print('START TRAINING...')
    agent.train(MAX_T, CHECKPOINT_PATH, PERFORMANCE_FILE)