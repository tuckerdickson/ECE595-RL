import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PartiallyObservableFrozenLake import PartiallyObservableFrozenLake

class DeepQNetwork(nn.Module):
    """Neural network architecture for Deep Q-Learning"""
    def __init__(self, state_size, action_size, hidden_layers=[256, 256]):
        """
        Initialize the Deep Q-Network
        
        Args:
            state_size (int): Dimension of the state space
            action_size (int): Number of possible actions
            hidden_layers (list): List of hidden layer sizes
        """
        super(DeepQNetwork, self).__init__()
        
        # create layers dynamically based on hidden_layers
        layer_sizes = [state_size] + hidden_layers + [action_size]
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        # activation functions
        self.activations = [nn.ReLU() for _ in range(len(hidden_layers))]

    def forward(self, x):
        """Forward pass through the network"""
        for layer, activation in zip(self.layers[:-1], self.activations):
            x = activation(layer(x))
        return self.layers[-1](x)

class ReplayBuffer:
    """Experience replay buffer for storing and sampling experiences"""
    def __init__(self, capacity, state_size):
        """
        Initialize the Replay Buffer
        
        Args:
            capacity (int): Maximum number of experiences to store
            state_size (int): Dimension of the state space
        """
        self.capacity = capacity
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.index = 0
        self.is_full = False

    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer"""
        idx = self.index % self.capacity
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        
        self.index += 1
        if self.index >= self.capacity:
            self.is_full = True

    def sample(self, batch_size):
        """Sample a batch of experiences"""
        max_index = self.capacity if self.is_full else self.index
        indices = np.random.choice(max_index, batch_size, replace=False)
        
        return (
            torch.FloatTensor(self.states[indices]),
            torch.LongTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.dones[indices])
        )

    def __len__(self):
        return self.index if not self.is_full else self.capacity

class DQNAgent:
    """Deep Q-Learning Agent"""
    def __init__(self, state_size, action_size, device='cuda', lr=1e-4):
        """
        Initialize the DQN Agent
        
        Args:
            state_size (int): Dimension of the state space
            action_size (int): Number of possible actions
            device (str): Device to run computations on (cuda/cpu)
            lr (float): Learning rate for the optimizer
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-Network and Target Network
        self.q_network = DeepQNetwork(state_size, action_size).to(self.device)
        self.target_network = DeepQNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000, state_size=state_size)
        
        # hyperparameters
        self.gamma = 0.99
        self.tau = 0.001
        self.batch_size = 256
        
        # exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9995

    def select_action(self, state):
        """
        Select an action using epsilon-greedy strategy
        
        Args:
            state (np.array): Current environment state
        
        Returns:
            int: Selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def train(self, state, action, reward, next_state, done):
        """
        Train the agent by storing experience and performing learning
        
        Args:
            state (np.array): Current state
            action (int): Selected action
            reward (float): Reward received
            next_state (np.array): Next state
            done (bool): Episode termination flag
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # sample batch and train
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # target Q-values
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # compute loss
        loss = F.mse_loss(current_q_values, target_q_values.detach())
        
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # update target network
        self._soft_update()
        
        # decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _soft_update(self):
        """Soft update of the target network"""
        for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)

def flatten_observation(obs):
    """
    Flatten the 3x3 observation into a 1D array
    
    Args:
        obs (list): 3x3 observation from PartiallyObservableFrozenLake
    
    Returns:
        np.array: Flattened observation
    """
    # replace None with a default value
    flat_obs = [cell if cell is not None else -1 for row in obs for cell in row]
    return np.array(flat_obs, dtype=np.float32)

def train_dqn(env_class, num_episodes, max_steps, render):
    """
    Train a DQN agent in a given environment
    
    Args:
        env_class (gym.Env): Environment class to instantiate
        num_episodes (int): Number of training episodes
        max_steps (int): Maximum steps per episode
        render (bool): Whether to render the environment
    
    Returns:
        DQNAgent: Trained agent
    """
    # create environment
    env = env_class(render_mode='human' if render else None)
    
    # determine state and action space
    initial_obs = env.reset()
    state_size = len(flatten_observation(initial_obs))
    action_size = env.env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = flatten_observation(env.reset())
        episode_reward = 0
        
        for step in range(max_steps):
            # select and execute action
            action = agent.select_action(state)
            next_obs, reward, done = env.step(action)
            next_state = flatten_observation(next_obs)
            
            # train agent
            agent.train(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.3f}, Epsilon: {agent.epsilon:.3f}")
    
    # plot rewards per episode
    plt.figure(figsize=(15, 5))
    
    # subplot 1: raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid()

    # subplot 2: average rewards every 100 episodes
    window_size = 100
    averaged_rewards = [np.mean(episode_rewards[i:i+window_size]) for i in range(0, len(episode_rewards), window_size)]
    
    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(episode_rewards), window_size), averaged_rewards)
    plt.title('Average Reward Every 100 Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid()
    
    plt.tight_layout()
    plt.savefig('reward_analysis.png')
    plt.close()
    
    return agent

if __name__ == '__main__':
    train_dqn(PartiallyObservableFrozenLake, num_episodes=10000, max_steps=30, render=False)