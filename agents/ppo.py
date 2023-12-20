from .base_agent import BaseAgent
from common.misc_util import adjust_lr, get_n_params
import torch
import torch.optim as optim
import numpy as np
import os
import csv
import gymnasium as gym


class PPO(BaseAgent):
    def __init__(self,
                 env,
                 policy,
                 logger,
                 storage,
                 device,
                 n_checkpoints,
                 env_valid=None,
                 storage_valid=None,
                 n_steps=500,
                 n_envs=4,
                 epoch=1,
                 mini_batch_per_epoch=1,
                 mini_batch_size=4*1,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=2.5e-4,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 normalize_adv=True,
                 normalize_rew=True,
                 use_gae=True,
                 **kwargs):

        super(PPO, self).__init__(env, policy, logger, storage, device,
                                  n_checkpoints, env_valid, storage_valid)

        self.n_steps = n_steps
        self.n_envs = n_envs
        self.epoch = epoch
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae

    def preprocess_observation(self, obs): # Added
        obs = obs.astype(np.float32) / 255.0  # Normalize to the range [0, 1]
        obs = obs.transpose((0, 3, 1, 2))  # Transpose channels to match policy network's input format
        return obs

    def map_int_to_char(self, action_value): # Added
        DISCRETE_ACTIONS = ["N", "S", "E", "W"]
        discrete_actions = np.empty(len(action_value), dtype=np.dtype('U1'))

        for i in range(len(action_value)):
            action = action_value[i]
            if action == 0:
                discrete_actions[i] = DISCRETE_ACTIONS[0]
            elif action == 1:
                discrete_actions[i] = DISCRETE_ACTIONS[1]
            elif action == 2:
                discrete_actions[i] = DISCRETE_ACTIONS[2]
            else:
                discrete_actions[i] = DISCRETE_ACTIONS[3]

        return discrete_actions
    
    def predict(self, obs, hidden_state, done):
        with torch.no_grad():
            obs = self.preprocess_observation(obs) # Added
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1 - done).to(device=self.device)
            dist, value, hidden_state = self.policy(obs, hidden_state, mask)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)
        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy()

    def predict_w_value_saliency(self, obs, hidden_state, done):
        obs = torch.FloatTensor(obs).to(device=self.device)
        obs.requires_grad_()
        obs.retain_grad()
        hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
        mask = torch.FloatTensor(1-done).to(device=self.device)
        dist, value, hidden_state = self.policy(obs, hidden_state, mask)
        value.backward(retain_graph=True)
        act = dist.sample()
        log_prob_act = dist.log_prob(act)

        return act.detach().cpu().numpy(), log_prob_act.detach().cpu().numpy(), value.detach().cpu().numpy(), hidden_state.detach().cpu().numpy(), obs.grad.data.detach().cpu().numpy()

    def optimize(self):
        print("Optimizing...")
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.policy.train()
        for e in range(self.epoch):
            recurrent = self.policy.is_recurrent()
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                           recurrent=recurrent)
            # track = 0
            for sample in generator:
                obs_batch, hidden_state_batch, act_batch, done_batch, \
                    old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample
                mask_batch = (1-done_batch)
                # track += 1
                # print(f'Sample: {track}')

                obs_batch = self.preprocess_observation(obs_batch.cpu().numpy()) # Added
                obs_batch = torch.FloatTensor(obs_batch).to(device=self.device) # Added
                dist_batch, value_batch, _ = self.policy(obs_batch, hidden_state_batch, mask_batch)

                # Clipped Surrogate Objective
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                # Clipped Bellman-Error
                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip, self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                # Policy Entropy
                entropy_loss = dist_batch.entropy().mean()
                loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                loss.backward()

                # Let model to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                pi_loss_list.append(-pi_loss.item())
                value_loss_list.append(-value_loss.item())
                entropy_loss_list.append(entropy_loss.item())

        summary = {'Loss/pi': np.mean(pi_loss_list),
                   'Loss/v': np.mean(value_loss_list),
                   'Loss/entropy': np.mean(entropy_loss_list)}
        # print(summary)
        return summary

    def train(self, num_timesteps, checkpoint_path, log_file, performance_file):
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        obs = self.env.reset()
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)

        if self.env_valid is not None:
            obs_v = self.env_valid.reset()
            hidden_state_v = np.zeros((self.n_envs, self.storage.hidden_state_size))
            done_v = np.zeros(self.n_envs)

        # for i in range(self.n_envs):
        #     self.env = gym.wrappers.RecordEpisodeStatistics(self.env) # Added

        # Added below blocks to determine the starting episode number by reading from the CSV file
        cur_episode = 0
        if os.path.exists(log_file):
            with open(log_file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
                if len(rows) > 1: # Check if there's at least one row of data (excluding the header)
                    cur_episode = int(rows[-1][0]) + 1
                else:
                    with open(log_file, 'w', newline='') as csvfile:
                        fieldnames = ['Episode', 'Env', 'Steps', 'Reward']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
        else:
            with open(log_file, 'w', newline='') as csvfile:
                fieldnames = ['Episode', 'Env', 'Steps', 'Reward']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        
        csvfile = open(log_file, 'a', newline='')
        fieldnames = ['Episode', 'Env', 'Steps', 'Reward']
        log_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if os.path.exists(performance_file):
            with open(performance_file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
                if len(rows) > 1: # Check if there's at least one row of data (excluding the header)
                    cur_episode = int(rows[-1][0]) + 1
                else:
                    with open(performance_file, 'w', newline='') as csvfile:
                        fieldnames = ['Episode', 'CumulativeReward', 'Loss/pi', 'Loss/v', 'Loss/entropy']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
        else:
            with open(performance_file, 'w', newline='') as csvfile:
                fieldnames = ['Episode', 'CumulativeReward', 'Loss/pi', 'Loss/v', 'Loss/entropy']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        csvfile = open(performance_file, 'a', newline='')
        fieldnames = ['Episode', 'CumulativeReward', 'Loss/pi', 'Loss/v', 'Loss/entropy']
        performance_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        cur_time = 0 # Added
        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()
            print('Reseting cumulative reward per episode to 0.')
            cumulative_rew_per_ep = 0 # Added
            print(self.n_steps)
            for i in range(self.n_steps):
                self.env.envs[0].render() # Added
                act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done)
                act_compass = self.map_int_to_char(act) # Added compass
                next_obs, rew, done, info = self.env.step(act_compass) # Added compass
                self.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
                obs = next_obs
                hidden_state = next_hidden_state
                # print(f'Step: {i}, done: {done}, rew: {rew}')
                with np.nditer(np.array(rew), flags=['multi_index']) as it:
                    for rew_step in it:
                        cumulative_rew_per_ep += rew_step
                # Added an iterator to print episode stats
                with np.nditer(np.array(done), flags=['multi_index']) as it:
                    # print(f'Dones: {done}')
                    # print(f'Rewards: {rew}')
                    # print(f'Info: {info}')
                    for done_step in it:
                        if done_step and info[it.multi_index[0]]['TimeLimit.truncated']!=True:
                            print("Episode {} for env {} completed successfully after {} time steps with total reward = {}.".format(cur_episode, it.multi_index[0], cur_time, rew[it.multi_index]))
                            log_writer.writerow({'Episode': cur_episode, 'Env': it.multi_index[0], 'Steps': cur_time, 'Reward': rew[it.multi_index]})                   
                            cur_time = 0 # Added
                cur_time +=1 # Added
            print(f'Episode {cur_episode} ended with a cumulative reward of {cumulative_rew_per_ep}.')
            cur_episode += 1 # Added
            value_batch = self.storage.value_batch[:self.n_steps]
            _, _, last_val, hidden_state = self.predict(obs, hidden_state, done)
            self.storage.store_last(obs, hidden_state, last_val)

            # Compute advantage estimates
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            #valid
            if self.env_valid is not None:
                for _ in range(self.n_steps):
                    act_v, log_prob_act_v, value_v, next_hidden_state_v = self.predict(obs_v, hidden_state_v, done_v)
                    act_compass_v = self.map_int_to_char(act) # Added compass
                    next_obs_v, rew_v, done_v, info_v = self.env_valid.step(act_compass_v) # Added compass
                    self.storage_valid.store(obs_v, hidden_state_v, act_v,
                                             rew_v, done_v, info_v,
                                             log_prob_act_v, value_v)
                    obs_v = next_obs_v
                    hidden_state_v = next_hidden_state_v
                _, _, last_val_v, hidden_state_v = self.predict(obs_v, hidden_state_v, done_v)
                self.storage_valid.store_last(obs_v, hidden_state_v, last_val_v)
                self.storage_valid.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # Optimize policy & valueq
            summary = self.optimize()           
            performance_writer.writerow({'Episode': cur_episode-1, 'CumulativeReward': cumulative_rew_per_ep, 'Loss/pi': summary['Loss/pi'], 'Loss/v': summary['Loss/v'], 'Loss/entropy': summary['Loss/entropy']})                   
    

            # Log the training-procedure (Removed)

            self.t += self.n_steps * self.n_envs

            self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, num_timesteps)
            
            # Added below block to reset maze after n_steps
            if cur_time > self.n_steps - 1:
                obs = self.env.reset()
                hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
                done = np.zeros(self.n_envs)
                cur_time = 0
                print("Maze was reset after {} time steps.".format(cur_time))

            # Save the model
            if self.t == ((checkpoint_cnt+1) * save_every):
                print("Saving model.")
                # torch.save({'model_state_dict': self.policy.state_dict(),
                #             'optimizer_state_dict': self.optimizer.state_dict()},
                #              checkpoint_path + str(self.t) + '.pt') # Changed logdir
                torch.save({'model_state_dict': self.policy.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                             checkpoint_path + str((cur_episode)*self.n_envs*self.n_steps) + '.pt') # Changed logdir
                checkpoint_cnt += 1
        
        csvfile.close() # Added
        self.env.close()
        if self.env_valid is not None:
            self.env_valid.close()