"""
[CSCI-GA 3033-090] Special Topics: Deep Decision Making & Reinforcement Learning

Homework - 2, DAgger
Deadline: March 8, 2024 11:59 PM.

Complete the code template provided in dagger_template.py, with the right 
code in every TODO section, to implement DAgger. Attach the completed 
file in your submission.
"""

import tqdm
import hydra
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import gym
import particle_envs

from utils import weight_init, ExpertBuffer
from video import VideoRecorder

from matplotlib import pyplot as plt

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        # Define a network with three hidden layers and Tanh output activation.
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.apply(weight_init)

    def forward(self, obs, goal):
        # Concatenate observation and goal, then pass through the network.
        x = torch.cat([obs, goal], dim=1)
        action = self.model(x)
        return action

def initialize_model_and_optim(cfg):
    # Create an Actor model and its Adam optimizer.
    input_dim = cfg.obs_dim * 2
    action_dim = cfg.action_dim
    hidden_dim = cfg.hidden_dim
    model = Actor(input_dim, action_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    return model, optimizer

class Workspace:
    def __init__(self, cfg):
        self._work_dir = os.getcwd()
        print(f'workspace: {self._work_dir}')
        self.cfg = cfg

        self.device = torch.device(cfg.device)
        self.train_env = gym.make('particle-v0', height=cfg.height, width=cfg.width, step_size=cfg.step_size, reward_type='dense')
        self.eval_env = gym.make('particle-v0', height=cfg.height, width=cfg.width, step_size=cfg.step_size, reward_type='dense')

        self.expert_buffer = ExpertBuffer(cfg.experience_buffer_len, 
                                          self.train_env.observation_space.shape,
                                          self.train_env.action_space.shape)
        
        self.model, self.optimizer = initialize_model_and_optim(cfg)

        # Define the loss function (Mean Squared Error)
        self.loss_function = nn.MSELoss()

        # Initialize the video recorder.
        self.video_recorder = VideoRecorder(self._work_dir)
        
    def eval(self, ep_num):
        # Evaluate the current model.
        self.model.eval()

        avg_eval_reward = 0.
        avg_episode_length = 0.
        successes = 0
        for ep in range(self.cfg.num_eval_episodes):
            eval_reward = 0.
            ep_length = 0.
            obs_np = self.eval_env.reset(reset_goal=True)
            goal_np = self.eval_env.goal
            if ep == 0:
                self.video_recorder.init(self.eval_env, enabled=True)
            # Convert observation and goal to torch tensors.
            obs = torch.from_numpy(obs_np).float().to(self.device).unsqueeze(0)
            goal = torch.from_numpy(goal_np).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                action = self.model(obs, goal)
            done = False
            while not done:
                action = action.squeeze().detach().cpu().numpy()
                obs_np, reward, done, info = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                obs = torch.from_numpy(obs_np).float().to(self.device).unsqueeze(0)
                with torch.no_grad():
                    action = self.model(obs, goal)
                eval_reward += reward
                ep_length += 1.
            avg_eval_reward += eval_reward
            avg_episode_length += ep_length
            if info.get('is_success', False):
                successes += 1
        avg_eval_reward /= self.cfg.num_eval_episodes
        avg_episode_length /= self.cfg.num_eval_episodes
        success_rate = successes / self.cfg.num_eval_episodes
        self.video_recorder.save(f'eval_{ep_num}.mp4')
        return avg_eval_reward, avg_episode_length, success_rate

    def model_training_step(self):
        # Optimize the model using the aggregated expert data.
        self.model.train()
        avg_loss = 0.
        iterable = tqdm.trange(self.cfg.num_training_steps)
        for _ in iterable:
            num_samples = self.cfg.batch_size
            obs, goal, action = self.expert_buffer.sample(num_samples)
            # Convert data from numpy to torch tensors.
            obs = torch.from_numpy(obs).float().to(self.device)
            goal = torch.from_numpy(goal).float().to(self.device)
            action = torch.from_numpy(action).float().to(self.device)
            self.optimizer.zero_grad()
            predicted_action = self.model(obs, goal)
            loss = self.loss_function(predicted_action, action)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()
        avg_loss /= self.cfg.num_training_steps
        return avg_loss

    
    def run(self):
        prev_reward = None
        reward = None
        train_loss, eval_reward, episode_length = None, 0, 0
        # Lists for plotting: total expert queries vs. success rate.
        expert_queries_list = []
        success_rate_list = []
        
        iterable = tqdm.trange(self.cfg.total_training_episodes)
        for ep_num in iterable:
            iterable.set_description('Collecting exp')
            self.model.eval()
            ep_train_reward = 0.
            ep_length = 0.
            
            initial_beta = self.cfg.get("initial_beta", 1.0)
            decay_rate = self.cfg.get("decay_rate", 0.95)
            min_beta = self.cfg.get("min_beta", 0.1)
            beta = initial_beta * (decay_rate ** ep_num)
            beta = max(beta, min_beta)
            
            obs_np = self.train_env.reset(reset_goal=True)
            goal_np = self.train_env.goal
            done = False
            while not done:
                # With probability beta, query the expert.
                # Here we use the concept of previously stored reward.
                if np.random.rand() < beta and (prev_reward is None or prev_reward != 0):
                    expert_action = self.train_env.get_expert_action()
                    # Insert the expert label into the buffer and use the expert action for execution.
                    self.expert_buffer.insert(obs_np, goal_np, expert_action)
                    action_np = expert_action
                else:
                    # Otherwise, use the model's predicted action.
                    obs_tensor = torch.from_numpy(obs_np).float().to(self.device).unsqueeze(0)
                    goal_tensor = torch.from_numpy(goal_np).float().to(self.device).unsqueeze(0)
                    with torch.no_grad():
                        action = self.model(obs_tensor, goal_tensor)
                    action_np = action.squeeze().detach().cpu().numpy()
                
                # Execute the chosen action in the environment.
                prev_reward = reward
                obs_next, reward, done, info = self.train_env.step(action_np)
                obs_np = obs_next
                ep_train_reward += reward
                ep_length += 1

            train_reward = ep_train_reward

            # Periodically train the model on the aggregated data.
            if (ep_num + 1) % self.cfg.train_every == 0:
                iterable.set_description('Training model')
                train_loss = self.model_training_step()

            # Periodically evaluate the current model.
            if ep_num % self.cfg.eval_every == 0:
                iterable.set_description('Evaluating model')
                eval_reward, episode_length, success_rate = self.eval(ep_num)
                
                expert_queries = self.train_env.expert_calls
                expert_queries_list.append(expert_queries)
                success_rate_list.append(success_rate)

            iterable.set_postfix({
                'Train loss': train_loss,
                'Train reward': train_reward,
                'Eval reward': eval_reward
            })
        
        # After training, plot expert queries vs. success rate.
        plt.figure()
        plt.plot(expert_queries_list, success_rate_list, marker='o')
        plt.xlabel("Number of Expert Queries")
        plt.ylabel("Success Rate")
        plt.title("Expert Queries vs. Success Rate with Beta Decay")
        plt.grid(True)
        plt.savefig(os.path.join(self._work_dir, "expert_queries_vs_success_rate.png"))
        plt.show()


@hydra.main(config_path='.', config_name='train')
def main(cfg):
    # In hydra, the configuration in train.yaml is loaded into cfg.
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
