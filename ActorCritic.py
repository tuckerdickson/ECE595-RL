import time
import sys
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from PartiallyObservableFrozenLake import PartiallyObservableFrozenLake

INPUT_DIM = 9                   # dimension of the observability window (height * width)
HIDDEN_DIM = 128                # number of hidden dimensions in the actor and critic networks
ACTION_DIM = 4                  # output dimension of the actor network (number of actions)

LEARNING_RATE = 1e-3            # optimizer learning rate
GAMMA = 0.99                    # discount factor

MAX_STEPS_PER_EPISODE = 30     # maximum number of steps allowed before episode resets
EPISODE_AVERAGE = 10            # the number of past episodes to average over when computing results
EARLY_STOP_SUCCESSES = 80       # the minimum number of successes needed to stop early on a given layout
    
class Actor(nn.Module):
    """This class defines the "actor" portion of the Actor-Critic method.
    It is implemented as a simple neural network which takes the agent's observability window as input,
    and produces a probability distribution over the action space as an output."""
    
    def __init__(self):
        """Initializes the Actor instance by defining its two fully-connected layers.
        The first layer maps the input observability window to a HIDDEN_DIM-dimensional latent representation.
        The second layer maps the latent representation to a probability distribution over the action space."""
        
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)         # first fully connected layer (observability window -> hidden representation)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2)   # second fully connected layer (hidden representation -> hidden representation)
        self.fc3 = nn.Linear(HIDDEN_DIM // 2, ACTION_DIM)   # third fully connected layer (hidden representation -> action distribution)

    def forward(self, x):
        """Propagates the input observability window through the two network layers,
        producing an action probability distribution at the output.
        
        Args:
            x (torch.tensor(torch.float32)): The agent's (flattened) observability window.

        Returns:
            torch.tensor(torch.float32): The Actor's probability distribution over the action space.
        """
        
        # send the observability window (x) through the first layer, using ReLU as nonlinearity
        x = torch.relu(self.fc1(x))

        # send the hidden representation through the second layer
        x = torch.relu(self.fc2(x))

        # send the second hidden representation through the third layer and perform softmax to produce a probability distribution
        return torch.softmax(self.fc3(x), dim=-1)

class Critic(nn.Module):
    """This class defines the "critic" portion of the Actor-Critic method.
    It is implemented as a simple neural network which takes the agent's observability window as input,
    and produces an estimate of the value of the given state."""

    
    def __init__(self):
        """Initializes the Critic instance by defining its two fully-connected layers.
        The first layer maps the input observability window to a HIDDEN_DIM-dimensional latent representation.
        The second layer maps the latent representation to a scalar estimate of the value of the given state."""
        
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)         # first fully connected layer (observability window -> hidden representation)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2)   # second fully connected layer (hidden representation -> hidden representation)
        self.fc3 = nn.Linear(HIDDEN_DIM // 2, 1)            # third fully connected layer (hidden representation -> scalar value estimate)

    def forward(self, x):
        """Propagates the input observability window through the two network layers,
        producing a scalar estimate of the value of the given state.

        Args:
            x (torch.tensor(torch.float32)): The agent's (flattened) observability window.

        Returns:
            float: The Critic's estimate of the value of the given state.
        """

        # send the observability window (x) through the first layer, using ReLU as nonlinearity
        x = torch.relu(self.fc1(x))

        # send the hidden representation through the second layer
        x = torch.relu(self.fc2(x))

        # send the second hidden representation through the third layer and perform softmax to produce a probability distribution
        return self.fc3(x)

def preprocess_observation(observation):
    """Processes an observation before it can be passed through the Actor/Critic networks.

    Args:
        x [[int]]: An observation, represented as a 2D array of ints

    Returns:
        torch.tensor(torch.float32): A tensor representation of the observation (also flattened).
    """
    return torch.tensor(observation, dtype=torch.float32).flatten()



# * ========================================================== STATIC-ENVIRONMENT FUNCTIONS ========================================================== *



def train_actor_critic_static(env, actor, critic, actor_optimizer, critic_optimizer, num_episodes=1000):
    """The training function used during the first phase of the project (before the progress report).
    This function trains an agent to learn a static environment (i.e., one where the layout is not changing).

    Args:
        env (PartiallyObsevableFrozenLake): The environment for the simulation
        actor (Actor): The actor network
        critic (Critic): The critic network
        actor_optimizer (torch.optim.Optimizer): The actor's optimizer
        critic_optimizer (torch.optim.Optimizer): The critic's optimizer
        num_episodes (int): The number of episodes to train for

    Returns:
        {string: [float]}: A dictionary containing various performance metrics gathered during training
    """
    
    ep_rewards = []         # per-episode accumulated rewards
    ep_successes = []       # per-episode number of successes
    ep_actor_losses = []    # per-episode accumulated actor loss
    ep_critic_losses = []   # per-episode accumulated critic loss
    ep_n_steps = []         # per-episode number of steps taken

    
    for episode in range(num_episodes):
        # reset the environment, process the initial observation so that it can be passed through networks
        observation = preprocess_observation(env.reset())

        done = False        # indicates whether this episode is finished
        steps = 0           # tracks steps taken in this episode
        total_reward = 0    # tracks accumulated reward in this episode

        # keep iterating until the agent falls in a hole, reaches the reward, or exceeds the maximum number of steps
        while not done and steps < MAX_STEPS_PER_EPISODE:

            # get the action probabilities from the actor, use to sample an action
            action_probs = actor(observation)
            action = torch.multinomial(action_probs, 1).item()

            # perform the sampled action, get next observation and reward
            next_observation, reward, done = env.step(action)
            next_observation = preprocess_observation(next_observation)

            # use the critic to estimate the value of the current and next states
            value = critic(observation)
            next_value = critic(next_observation) if not done else 0

            # compute the target value, error between the target value and the estimated value of current state
            td_target = reward + GAMMA * next_value
            td_error = td_target - value

            # update actor and critic networks
            critic_loss = td_error.pow(2)
            actor_loss = -torch.log(action_probs[action]) * td_error.detach()

            # zeor actor gradients, backpropagate actor loss, update actor parameters
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            # zeor critic gradients, backpropagate critic loss, update critic parameters
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # move to next state
            observation = next_observation

            # accumulate rewards and steps
            total_reward += reward
            steps += 1

        if episode >= 100:
            start = episode - 100
            avg_reward = sum(ep_rewards[start:episode]) / (episode - start)
            n_successes = sum(ep_successes[start:episode])
                
            # optional: print out episode reward every 100 episodes
            if episode % 100 == 0:
                print(f"\tEpisodes {start}-{episode}, Average Reward: {avg_reward}")
                # print(f"Episode {episode}/{episodes_per_layout}, Total Reward: {total_reward}")

        # log accumulated reward, successes, steps, actor loss, and critic loss for this episode
        ep_rewards.append(total_reward)
        ep_successes.append(1 if env.agent_position == env.goal_state else 0)
        ep_n_steps.append(steps)
        ep_actor_losses.append(actor_loss.detach())
        ep_critic_losses.append(critic_loss.detach())

    # aggregate performance metric lists into a dictionary and return
    results = {
        "ep_rewards": ep_rewards,
        "ep_successes": ep_successes,
        "ep_n_steps": ep_n_steps,
        "ep_actor_losses": ep_actor_losses,
        "ep_critic_losses": ep_critic_losses
    }
    return results


def test_actor_critic_static(env, actor, num_episodes=10, render = False):
    """The testing function used during the first phase of the project (before the progress report).
    In theory, this function tests an agent's ability to navigate an unseen environment; however,
    since we focused on static environments during phase 1, it was not really used.

    Args:
        env (PartiallyObsevableFrozenLake): The environment for the simulation
        actor (Actor): The actor network
        num_episodes (int): The number of episodes to test for
        render (boolean): True to render a simulation window, False otherwise
    """

    # set actor network in evaluation mode
    actor.eval()

    for episode in range(num_episodes):
        # reset the environment, process the initial observation so that it can be passed through networks
        observation = preprocess_observation(env.reset())

        done = False        # indicates whether this episode is finished
        steps = 0           # tracks steps taken in this episode

        # keep iterating until the agent falls in a hole, reaches the reward, or exceeds the maximum number of steps
        while not done and steps < MAX_STEPS_PER_EPISODE:
            # if render is True, render the simulation window
            if render:
                env.render()

            # turn off gradients, get action probabilities from actor network
            with torch.no_grad():
                action_probs = actor(observation)

            # sample an action from the action probabilities
            action = torch.multinomial(action_probs, 1).item()

            # perform the sampled action, get next observation and reward
            next_observation, reward, done = env.step(action)
            next_observation = preprocess_observation(next_observation)

            # accumulate steps, move to next state
            steps += 1
            observation = next_observation

def plot_raw_results(results):
    """Plots five performance metrics (total rewards per episode, success per episode, steps taken per episode,
    actor loss per episode, and critic loss per episode) accumulated over a training run.

    Args:
        results_dict ({string: [float]}): A dictionary containing the per-episode performance metrics for a training run
    """
    # display results on a 2x3 subplot
    fig, axs = plt.subplots(2,3)

    # plot accumulated rewards per episode
    axs[0,0].plot(results["ep_rewards"])
    axs[0,0].set_xlabel('Episode')
    axs[0,0].set_ylabel('Reward')
    axs[0,0].set_title('Total Reward Per Episode')
    
    # plot number of successes per episode
    axs[0,1].plot(results["ep_successes"])
    axs[0,1].set_xlabel('Episode')
    axs[0,1].set_ylabel('Success')
    axs[0,1].set_title('Success Per Episode')

    # plot number of steps taken per episode
    axs[0,2].plot(results["ep_n_steps"])
    axs[0,2].set_xlabel('Episode')
    axs[0,2].set_ylabel('Steps Taken')
    axs[0,2].set_title('Steps Taken Per Episode')

    # plot accumulated actor loss per episode
    axs[1,0].plot(results["ep_actor_losses"])
    axs[1,0].set_xlabel('Episode')
    axs[1,0].set_ylabel('Actor Loss')
    axs[1,0].set_title('Actor Loss Per Episode')

    # plot accumulated critic loss per episode
    axs[1,1].plot(results["ep_critic_losses"])
    axs[1,1].set_xlabel('Episode')
    axs[1,1].set_ylabel('Critic Loss')
    axs[1,1].set_title('Critic Loss Per Episode')

    # format subplot and display
    plt.tight_layout()
    plt.show()

def plot_avg_results(results_dict):
    """Plots five performance metrics (average rewards, average success, average steps taken,
    average actor loss, and average critic loss), each averaged over the EPISODE_AVERAGE previous
    episodes during a training run. In other words, the ith value displayed is the average of the
    values of episodes [i-EPISODE_AVERAGE...i]. One corner case is the first EPISODE_AVERAGE
    episodes. Here, we just average the current episode and all previous episodes.

    This method results in much smoother graphs compared to plotting the raw values at each episode.

    Args:
        results_dict ({string: [float]}): A dictionary containing the per-episode performance metrics for a training run
    """

    # keeps track of the average metrics over the past EPISODE_AVERAGE number of episodes
    avg_rewards = []
    avg_successes = []
    avg_steps = []
    avg_actor_losses = []
    avg_critic_losses = []

    # compute the average performance metrics for each episode
    for idx in range(1, len(results_dict["ep_rewards"])):
        start = max(0, idx - EPISODE_AVERAGE)   # index for the earliest episode to consider when averaging
        divisor = min(EPISODE_AVERAGE, idx)     # divisor to use when computing the averages

        # compute averages and append to lists
        avg_rewards.append(sum(results_dict["ep_rewards"][start:idx]) / divisor)
        avg_successes.append(sum(results_dict["ep_successes"][start:idx]) / divisor)
        avg_steps.append(sum(results_dict["ep_n_steps"][start:idx]) / divisor)
        avg_actor_losses.append(sum(results_dict["ep_actor_losses"][start:idx]) / divisor)
        avg_critic_losses.append(sum(results_dict["ep_critic_losses"][start:idx]) / divisor)  

    # display results on a 2x3 subplot
    fig, axs = plt.subplots(2,3)

    # plot averaged rewards per episode
    axs[0,0].plot(avg_rewards)
    axs[0,0].set_xlabel('Episode')
    axs[0,0].set_ylabel('Average Reward')
    axs[0,0].set_title(f'Average Reward Over Prior {EPISODE_AVERAGE} Episodes')

    # plot averaged successes per episode
    axs[0,1].plot(avg_successes)
    axs[0,1].set_xlabel('Episode')
    axs[0,1].set_ylabel('Average Success')
    axs[0,1].set_title(f'Average Success Over Prior {EPISODE_AVERAGE} Episodes')

    # plot averaged steps taken per episode
    axs[0,2].plot(avg_steps)
    axs[0,2].set_xlabel('Episode')
    axs[0,2].set_ylabel('Average Steps')
    axs[0,2].set_title(f'Average Steps Over Prior {EPISODE_AVERAGE} Episodes')

    # plot averaged actor loss per episode
    axs[1,0].plot(avg_actor_losses)
    axs[1,0].set_xlabel('Episode')
    axs[1,0].set_ylabel('Average Actor Reward')
    axs[1,0].set_title(f'Average Actor Reward Over Prior {EPISODE_AVERAGE} Episodes')

    # plot averaged critic loss per episode
    axs[1,1].plot(avg_critic_losses)
    axs[1,1].set_xlabel('Episode')
    axs[1,1].set_ylabel('Average Critic Reward')
    axs[1,1].set_title(f'Average Critic Reward Over Prior {EPISODE_AVERAGE} Episodes')

    # format subplot and display
    plt.tight_layout()
    plt.show()



# * ========================================================== DYNAMIC-ENVIRONMENT FUNCTIONS ========================================================== *



def train_actor_critic_dynamic(env, actor, critic, actor_optimizer, critic_optimizer, num_layouts=10, episodes_per_layout=1000, show_results=False):
    """The training function used during the second phase of the project (after the progress report).
    This function trains an agent to learn a dynamic environment (i.e., one where the layout is changing).

    Args:
        env (PartiallyObsevableFrozenLake): The environment for the simulation
        actor (Actor): The actor network
        critic (Critic): The critic network
        actor_optimizer (torch.optim.Optimizer): The actor's optimizer
        critic_optimizer (torch.optim.Optimizer): The critic's optimizer
        num_episodes (int): The number of episodes to train for

    Returns:
        {string: [float]}: A dictionary containing various performance metrics gathered during training
    """

    for layout in range(num_layouts):
        print(f"* ======================= LAYOUT {layout} ======================= *")

        ep_rewards = []         # per-episode accumulated rewards
        ep_successes = []       # per-episode number of successes
        ep_actor_losses = []    # per-episode accumulated actor loss
        ep_critic_losses = []   # per-episode accumulated critic loss
        ep_n_steps = []         # per-episode number of steps taken

        # initialize a new random layout
        env.initialize_random_layout()

        epsilon = 1.0                   # initial exploration rate
        epsilon_min = 0.1               # minimum exploration rate
        epsilon_decay = 0.995           # epsilon decay rate
        
        stop_early = False      # set to True if the average reward over 100 episodes exceeds 0.8
        for episode in range(episodes_per_layout):
            if stop_early:
                break
            
            # reset the environment, process the initial observation so that it can be passed through networks
            observation = preprocess_observation(env.reset())

            done = False        # indicates whether this episode is finished
            steps = 0           # tracks steps taken in this episode
            total_reward = 0    # tracks accumulated reward in this episode

            # keep iterating until the agent falls in a hole, reaches the reward, or exceeds the maximum number of steps
            while not done and steps < MAX_STEPS_PER_EPISODE:

                # get the action probabilities from the actor, use to sample an action
                action_probs = actor(observation)
                action = torch.argmax(action_probs).item()

##                if random.random() < epsilon:
##                    action = random.randint(0, 3)
##                else:
##                    action = torch.argmax(action_probs).item()
##                epsilon = max(epsilon_min, epsilon_decay * epsilon)

                # perform the sampled action, get next observation and reward
                next_observation, reward, done = env.step(action)                    
                next_observation = preprocess_observation(next_observation)

                # use the critic to estimate the value of the current and next states
                value = critic(observation)
                next_value = critic(next_observation) if not done else 0

                # compute the target value, error between the target value and the estimated value of current state
                td_target = reward + GAMMA * next_value
                td_error = td_target - value

                # update actor and critic networks
                critic_loss = td_error.pow(2)
                actor_loss = -torch.log(action_probs[action]) * td_error.detach()

                # zeor actor gradients, backpropagate actor loss, update actor parameters
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                # zeor critic gradients, backpropagate critic loss, update critic parameters
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # move to next state
                observation = next_observation

                # accumulate rewards and steps
                total_reward += reward
                steps += 1

            # log accumulated reward, successes, steps, actor loss, and critic loss for this episode
            ep_rewards.append(total_reward)
            ep_successes.append(1 if env.agent_position == env.goal_state else 0)
            ep_n_steps.append(steps)
            ep_actor_losses.append(actor_loss.detach())
            ep_critic_losses.append(critic_loss.detach())

            # early stopping
            if episode >= 100:
                start = episode - 100
                avg_reward = sum(ep_rewards[start:episode]) / (episode - start)
                n_successes = sum(ep_successes[start:episode])
                
                # print out episode reward every 100 episodes
                if episode % 100 == 0:
                    print(f"\tEpisodes {start}-{episode}, Average Reward: {avg_reward}")

                # if the number of successes over the last 100 episodes exceeds 80, stop early
##                if n_successes > EARLY_STOP_SUCCESSES:
##                    print(f"\t\t{n_successes}/{100} over the last 100 episodes. Stopping early.")
##                    stop_early = True

        # aggregate performance metric lists into a dictionary and return
        result_dict = {
            "ep_rewards": ep_rewards,
            "ep_successes": ep_successes,
            "ep_n_steps": ep_n_steps,
            "ep_actor_losses": ep_actor_losses,
            "ep_critic_losses": ep_critic_losses
        }

        # plot the results
        if show_results:
            plot_avg_results_dynamic(result_dict)

def test_actor_critic_dynamic(env, actor, num_layouts=100, episodes_per_layout=100):
    """The testing function used during the second phase of the project (after the progress report).
    This function tests an agent's ability to navigate unseen environments. 

    Args:
        env (PartiallyObsevableFrozenLake): The environment for the simulation
        actor (Actor): The actor network
        num_episodes (int): The number of episodes to test for
        render (boolean): True to render a simulation window, False otherwise
    """
    
    test_rewards = []
    test_success_rates = []

    for lo in range(num_layouts):
        
        rewards = []
        total_successes = 0
        env.initialize_random_layout()
        
        for episode in range(episodes_per_layout):
            
            observation = preprocess_observation(env.reset())
            total_reward = 0
            done = False
            
            while not done:
                
                action_probs = actor(observation)
                action = torch.multinomial(action_probs, 1).item()
                
                next_observation, reward, done = env.step(action)
                observation = preprocess_observation(next_observation)
                
                total_reward += reward
                if reward == 10:
                    total_successes += 1
                
            rewards.append(total_reward)

        test_rewards.append(np.mean(rewards))
        test_success_rates.append(total_successes / episodes_per_layout)

    print("Test Rewards Per Layout:", test_rewards)
    print("Test Success Rates Per Layout:", test_success_rates)

def plot_avg_results_dynamic(results_dict):
    """Plots three performance metrics (average rewards, average success, average steps taken),
    each averaged over the EPISODE_AVERAGE previous
    episodes during a training run. In other words, the ith value displayed is the average of the
    values of episodes [i-EPISODE_AVERAGE...i]. One corner case is the first EPISODE_AVERAGE
    episodes. Here, we just average the current episode and all previous episodes.

    This method results in much smoother graphs compared to plotting the raw values at each episode.

    Args:
        results_dict ({string: [float]}): A dictionary containing the per-episode performance metrics for a training run
    """

    # keeps track of the average metrics over the past EPISODE_AVERAGE number of episodes
    avg_rewards = []
    avg_successes = []
    avg_steps = []

    # compute the average performance metrics for each episode
    for idx in range(1, len(results_dict["ep_rewards"])):
        start = max(0, idx - EPISODE_AVERAGE)   # index for the earliest episode to consider when averaging
        divisor = min(EPISODE_AVERAGE, idx)     # divisor to use when computing the averages

        # compute averages and append to lists
        avg_rewards.append(sum(results_dict["ep_rewards"][start:idx]) / divisor)
        avg_successes.append(sum(results_dict["ep_successes"][start:idx]) / divisor)
        avg_steps.append(sum(results_dict["ep_n_steps"][start:idx]) / divisor)

    # display results on a 2x3 subplot
    fig, axs = plt.subplots(1,3)

    # plot averaged rewards per episode
    axs[0].plot(avg_rewards)
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Average Reward')
    axs[0].set_title(f'Average Reward Over Prior {EPISODE_AVERAGE} Episodes')

    # plot averaged successes per episode
    axs[1].plot(avg_successes)
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Average Success')
    axs[1].set_title(f'Average Success Over Prior {EPISODE_AVERAGE} Episodes')

    # plot averaged steps taken per episode
    axs[2].plot(avg_steps)
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Average Steps')
    axs[2].set_title(f'Average Steps Over Prior {EPISODE_AVERAGE} Episodes')

    # format subplot and display
    plt.tight_layout()
    plt.show()



# * ========================================================== MAIN FUNCTION ========================================================== *



if __name__ == "__main__":
    error_message = "Please pass exactly one argument: \"static\" or \"dynamic\""

    # make sure there are exactly 2 command line arguments (program name and static/dynamic)
    assert len(sys.argv) == 2, error_message

    # static environment case (i.e., layout does not change)
    if sys.argv[1].lower() == "static":
        # initialize environment
        train_env = PartiallyObservableFrozenLake(is_slippery=False, custom_reward=True)
        test_env = PartiallyObservableFrozenLake(is_slippery=False, desc=train_env.env.unwrapped.desc, render_mode="human")
        
        # initialize actor and critic networks
        actor = Actor()
        critic = Critic()

        # initialize optimizers
        actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
        critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

        # train the agent, collect the performance results
        results = train_actor_critic_static(train_env, actor, critic, actor_optimizer, critic_optimizer, num_episodes=1000)
        test_actor_critic_static(test_env, actor, num_episodes=1, render = True)
        
        # display the results
        plot_raw_results(results)
        plot_avg_results(results)

    # dynamic environment case (i.e., layout does change)
    elif sys.argv[1].lower() == "dynamic":
        # initialize environment
        train_env = PartiallyObservableFrozenLake(is_slippery=False, custom_reward=True)
        test_env = PartiallyObservableFrozenLake(is_slippery=False, custom_reward=True)
        
        # initialize actor and critic networks
        actor = Actor()
        critic = Critic()

        # initialize optimizers
        actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
        critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

        # train the agent, collect the performance results
        train_actor_critic_dynamic(train_env, actor, critic, actor_optimizer, critic_optimizer, num_layouts=10, episodes_per_layout=1000, show_results=True)
        test_actor_critic_dynamic(test_env, actor)
        
    # any other input is not acceptable
    else:
        print(error_message)
        
