import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F

# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.fc2 = nn.Linear(h1_nodes, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions) # output layer w

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = F.relu(self.fc2(x))
        return self.out(x)         # Calculate output

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# trainer
class SafePedestrianDQL():
    # Hyperparameters
    learning_rate_a = 0.001         # learning rate (alpha)
    discount_factor_g = 0.9         # discount rate (gamma)    
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000       # size of replay memory
    mini_batch_size = 64  #  32     # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    def train(self, episodes): #, render=False, is_slippery=False):
        # create instance
        env = gym.make('PedestrianCrossing-v0')
        num_states = env.observation_space.shape[0] # should be 6
        num_actions = env.action_space.n #should be 3
        
        epsilon = 1.0 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # create policy & target network.
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict())# make the target & policy networks the same

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_dqn = policy_dqn.to(device)
        target_dqn = target_dqn.to(device)

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        rewards_per_episode = np.zeros(episodes)# list to keep track of rewards collected per episode. Init list to 0's.
        epsilon_history = []        #list to keep track of epsilon decay
        step_count=0 # track number of steps taken. Used for syncing policy => target network.
        
        for i in range(episodes):
            state, _ = env.reset() # Init to state 0
            state = self.state_to_tensor(state, device)
            terminated = False # <- new flag for when any terminating condition is met for pedestrian.

            total_reward = 0

            while not terminated:
                # select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                else: # select best action
                    with torch.no_grad():
                        action = policy_dqn(state).argmax().item()

                # execute action
                new_state, reward, terminated, truncated, _ = env.step(action)
                next_state = self.state_to_tensor(next_state, device) if not terminated else None

                # save experience into memory
                memory.append((state, action, new_state, reward, terminated)) 
                
                state = new_state # move to the next state
                total_reward+= reward

            # check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory)>self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn, device)

                # copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0

            rewards_per_episode[i] = total_reward
            # decay epsilon
            epsilon = max(epsilon - 1/episodes, 0.05)
            epsilon_history.append(epsilon)
            print(f"Episode{i+1}/{episodes} finished with reward {total_reward}")
        env.close()

        torch.save(policy_dqn.state_dict(), "pedestrian_crossing_dql.pt") # save policy
        
        plt.figure(figsize=(12,5))
        plt.subplot(121)
        plt.plot(rewards_per_episode)
        plt.title("Total Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        plt.subplot(122)
        plt.plot(epsilon_history)
        plt.title("Epsilon Decay")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
            
        plt.tight_layout()
        plt.savefig('pedestrian_crossing_dql_training.png')
        plt.show()

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn, device):
        states, actions, next_states, rewards, terminateds = zip(*mini_batch)

        states = torch.cat(states).to(device)
        actions = torch.tensor(actions, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        
        non_final_mask = torch.tensor([s is not None for s in next_states], device = device, dtype = torch.bool)
        non_final_next_states = torch.cat([s for s in next_states if s is not None]).to(device)

        current_q_values = policy_dqn(states).gather(1, actions)

        next_q_values = torch.zeros(self.mini_batch_size, device=device) # target values
        next_q_values[non_final_mask] = target_dqn(non_final_next_states).max(1)[0].detach()
        
        expected_q_values = (next_q_values.unsqueeze(1)*self.discount_factor_g) + rewards
        #       compute the loss
        loss = self.loss_fn(current_q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_to_tensor(self, state:int, device):
        # convert numpy array state to torch tensor
        return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)


if __name__ == '__main__':

    pedestrian_crossing = SafePedestrianDQL()
    pedestrian_crossing.train(episodes=1000)
    #safe_pedestrian.evaluate() # <- this will include things like: safety rate, average wait time, etc.