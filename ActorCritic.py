import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from PartiallyObservableFrozenLake import PartiallyObservableFrozenLake

INPUT_DIM = 9
HIDDEN_DIM = 128
ACTION_DIM = 4

LEARNING_RATE = 1e-3
GAMMA = 0.99

MAX_STEPS_PER_EPISODE = 100
EPISODE_AVERAGE = 10
    
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, ACTION_DIM)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def preprocess_observation(observation):
    return torch.tensor(observation, dtype=torch.float32).flatten()
    
def train_actor_critic(env, actor, critic, actor_optimizer, critic_optimizer, num_episodes=1000):
    
    ep_rewards = []
    ep_successes = []
    ep_actor_losses = []
    ep_critic_losses = []
    ep_n_steps = []
    
    for episode in range(num_episodes):
        
        observation = preprocess_observation(env.reset())
        
        done = False
        steps = 0
        total_reward = 0
        
        while not done and steps < MAX_STEPS_PER_EPISODE:

            action_probs = actor(observation)
            action = torch.multinomial(action_probs, 1).item()
            
            next_observation, reward, done = env.step(action)
            next_observation = preprocess_observation(next_observation)

            value = critic(observation)
            next_value = critic(next_observation) if not done else 0

            td_target = reward + GAMMA * next_value
            td_error = td_target - value

            critic_loss = td_error.pow(2)
            actor_loss = -torch.log(action_probs[action]) * td_error.detach()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            observation = next_observation
            total_reward += reward
            steps += 1

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}")
        
        ep_rewards.append(total_reward)
        ep_successes.append(1 if env.agent_position == env.goal_state else 0)
        ep_n_steps.append(steps)
        ep_actor_losses.append(actor_loss.detach())
        ep_critic_losses.append(critic_loss.detach())

    results = {
        "ep_rewards": ep_rewards,
        "ep_successes": ep_successes,
        "ep_n_steps": ep_n_steps,
        "ep_actor_losses": ep_actor_losses,
        "ep_critic_losses": ep_critic_losses
    }
    
    return results

def test_actor_critic(env, actor, num_episodes=10, render = False):
    
    actor.eval()

    for episode in range(num_episodes):
        
        observation = preprocess_observation(env.reset())

        done = False
        steps = 0

        while not done and steps < MAX_STEPS_PER_EPISODE:
            
            if render:
                env.render()
            
            with torch.no_grad():
                action_probs = actor(observation)
            action = torch.multinomial(action_probs, 1).item()

            next_observation, reward, done = env.step(action)
            next_observation = preprocess_observation(next_observation)

            steps += 1
            observation = next_observation

def plot_raw_results(results_dict):
    fig, axs = plt.subplots(2,3)
    axs[0,0].plot(results["ep_rewards"])
    axs[0,0].set_xlabel('Episode')
    axs[0,0].set_ylabel('Reward')
    axs[0,0].set_title('Total Reward Per Episode')
    
    axs[0,1].plot(results["ep_successes"])
    axs[0,1].set_xlabel('Episode')
    axs[0,1].set_ylabel('Success')
    axs[0,1].set_title('Success Per Episode')
    
    axs[0,2].plot(results["ep_n_steps"])
    axs[0,2].set_xlabel('Episode')
    axs[0,2].set_ylabel('Steps Taken')
    axs[0,2].set_title('Steps Taken Per Episode')
    
    axs[1,0].plot(results["ep_actor_losses"])
    axs[1,0].set_xlabel('Episode')
    axs[1,0].set_ylabel('Actor Loss')
    axs[1,0].set_title('Actor Loss Per Episode')
    
    axs[1,1].plot(results["ep_critic_losses"])
    axs[1,1].set_xlabel('Episode')
    axs[1,1].set_ylabel('Critic Loss')
    axs[1,1].set_title('Critic Loss Per Episode')

    plt.tight_layout()
    plt.show()

def plot_avg_results(results_dict):

    avg_rewards = []
    avg_successes = []
    avg_steps = []
    avg_actor_losses = []
    avg_critic_losses = []
        
    for idx in range(1, len(results_dict["ep_rewards"])):
        start = max(0, idx - EPISODE_AVERAGE)
        divisor = min(EPISODE_AVERAGE, idx)

        avg_rewards.append(sum(results_dict["ep_rewards"][start:idx]) / divisor)
        avg_successes.append(sum(results_dict["ep_successes"][start:idx]) / divisor)
        avg_steps.append(sum(results_dict["ep_n_steps"][start:idx]) / divisor)
        avg_actor_losses.append(sum(results_dict["ep_actor_losses"][start:idx]) / divisor)
        avg_critic_losses.append(sum(results_dict["ep_critic_losses"][start:idx]) / divisor)  

    fig, axs = plt.subplots(2,3)
    
    axs[0,0].plot(avg_rewards)
    axs[0,0].set_xlabel('Episode')
    axs[0,0].set_ylabel('Average Reward')
    axs[0,0].set_title(f'Average Reward Over Prior {EPISODE_AVERAGE} Episodes')
    
    axs[0,1].plot(avg_successes)
    axs[0,1].set_xlabel('Episode')
    axs[0,1].set_ylabel('Average Success')
    axs[0,1].set_title(f'Average Success Over Prior {EPISODE_AVERAGE} Episodes')
    
    axs[0,2].plot(avg_steps)
    axs[0,2].set_xlabel('Episode')
    axs[0,2].set_ylabel('Average Steps')
    axs[0,2].set_title(f'Average Steps Over Prior {EPISODE_AVERAGE} Episodes')
    
    axs[1,0].plot(avg_actor_losses)
    axs[1,0].set_xlabel('Episode')
    axs[1,0].set_ylabel('Average Actor Reward')
    axs[1,0].set_title(f'Average Actor Reward Over Prior {EPISODE_AVERAGE} Episodes')
    
    axs[1,1].plot(avg_critic_losses)
    axs[1,1].set_xlabel('Episode')
    axs[1,1].set_ylabel('Average Critic Reward')
    axs[1,1].set_title(f'Average Critic Reward Over Prior {EPISODE_AVERAGE} Episodes')

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":    
    train_env = PartiallyObservableFrozenLake()
    test_env = PartiallyObservableFrozenLake(render_mode="human")
    
    actor = Actor()
    critic = Critic()
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    results = train_actor_critic(train_env, actor, critic, actor_optimizer, critic_optimizer, num_episodes=1000)
    # test_actor_critic(test_env, actor, num_episodes=1, render=True)
    
    plot_raw_results(results)
    plot_avg_results(results)

