import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
from gymnasium.envs.registration import register
from pedestrian_crossing_env import PedestrianCrossingEnv



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
        return self.out(x)    # calculate raw Q-values for each action

# Experience Replay Buffer
class ReplayMemory():
    def __init__(self, maxlen):
        # each transition = (state, action, next_state, reward, done)
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        # adding a new experience/transition to memroy
        self.memory.append(transition)

    def sample(self, sample_size):
        # randomly sample a batch of experiences for training policy network.
        # here is where we break correlation between consecutive states.
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)#current size of memory buffer

# trainer
class SafePedestrianDQL():
    # Hyperparameters
    learning_rate_a = 0.001 #0.0005 #0.01 #0.005 #0.001         # learning rate (alpha)
    discount_factor_g = 0.9         # discount rate (gamma)    
    network_sync_rate = 8 #3 #10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000       # size of replay memory
    mini_batch_size = 32     # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later for the policy network.

    def train(self, episodes): #
        env = gym.make('PedestrianCrossing-v0')
        num_states = env.observation_space.shape[0] # should be 6 (pedest position, lanes, time)
        num_actions = env.action_space.n #should be 3
        
        epsilon = 1.0 # 1 = 100% random actions, so starts fully exploratory.
        memory = ReplayMemory(self.replay_memory_size) # init experience replay memory.

        # create policy & target network.
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states*2, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states*2, out_actions=num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict())# make the target & policy networks the same
        # setup device & move models to GPU if they're available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_dqn = policy_dqn.to(device)
        target_dqn = target_dqn.to(device)

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)


        ## ADDING SCHEDULER ==============================
        # 'gamma' is like your lr decay after 'x' steps
        
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9) #gamma=0.1)
        
        ## ADDING SCHEDULER ==============================

        rewards_per_episode = np.zeros(episodes)# list to keep track of rewards collected per episode. Init list to 0's.
        epsilon_history = []        #list to keep track of epsilon decay
        step_count=0 # track number of steps taken. Used for syncing policy => target network.
        # The main training loop, per episode.
        for i in range(episodes):
            state, _ = env.reset() # Init to state 0
            state = self.state_to_tensor(state, device)# need to convert state to pytorch tensor.
            terminated = False # <- new flag for when any terminating condition is met for pedestrian.

            total_reward = 0

            while not terminated:
                # select action based on epsilon-greedy policy
                if random.random() < epsilon:
                    # select random action              actions: 0=wait,1=move forward,2=move backward
                    action = env.action_space.sample()# 
                else: # select best action
                    with torch.no_grad():
                        action = policy_dqn(state).argmax().item() # exploit learned policy

                # execute action
                next_state, reward, terminated, _, _ = env.step(action)
                next_state = self.state_to_tensor(next_state, device) if not terminated else None

                # save experience into memory
                memory.append((state, action, next_state, reward, terminated))  #new_state, reward, terminated)) 
                # MOVE to the next state
                state = next_state #new_state
                total_reward+= reward
                
                #DIDNT HAVE THIS BEFORE increment step count
                step_count += 1

            # check if enough experience has been collected and if at least 1 reward has been collected
            # Optimization step
            if len(memory)>self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn, device)

                # copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0

            #episode summary
            rewards_per_episode[i] = total_reward
            #-----------------------
            epsilon = max(epsilon - 1/episodes, 0.05) #
            
            #----------------------
            epsilon_history.append(epsilon)
            
            # registering THE CURRENT STEPS for schedduler!
            scheduler.step()
            
            if (i + 1) % 5000 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Learning rate at episode {i + 1}: {current_lr}")
            
            print(f"Episode {i+1}/{episodes} finished with reward {total_reward}")
        env.close()

        torch.save(policy_dqn.state_dict(), "pedestrian_crossing_dql.pt") # save policy
        
        window = 3000 # print average reward over last 'X' episodes
        if len(rewards_per_episode) < window:
            recent_avg = np.mean(rewards_per_episode)
            print(f"\nAverage reward over all {len(rewards_per_episode)} episodes: {recent_avg:.2f}")
        else:
            recent_avg = np.mean(rewards_per_episode[-window:])
            print(f"\nAverage reward over last {window} episodes: {recent_avg:.2f}")
        
        
        plt.figure(figsize=(18,5))
        
        plt.subplot(1,3,1)
        plt.plot(rewards_per_episode)
        plt.title("Total Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        plt.subplot(1,3,2)
        plt.plot(epsilon_history)
        plt.title("Epsilon Decay")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        
        plt.subplot(1,3,3)
        plt.plot(rewards_per_episode[-100:])
        plt.title("Reward (last 100 Episodes)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
            
        plt.tight_layout()
        plt.savefig('pedestrian_crossing_dql_training.png')

        
        # Histogram of rewards from the last 60 episodes
        last_n = 60
        final_rewards = rewards_per_episode[-last_n:]
        
        plt.figure(figsize=(6, 4))
        
        # Use unique values as bins to avoid empty spaces
        unique_bins = np.unique(final_rewards)
        counts, bins, patches = plt.hist(final_rewards, bins=unique_bins, edgecolor='black', align='left', rwidth=0.8)
        
        # Add text labels above each bar in the histogram
        for count, patch in zip(counts, patches):
            plt.text(patch.get_x() + patch.get_width() / 2, count, int(count),
                     ha='center', va='bottom')
        
        plt.title(f"Histogram of Final {last_n} Episode Rewards")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig('pedestrian_crossing_dql_histogram.png')
        plt.show()
        
        print(f"Reward variance (last {last_n}): {np.var(final_rewards):.2f}")

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn, device):
        # unpacking the batch of transitions
        states, actions, next_states, rewards, terminateds = zip(*mini_batch)
        # preparing the tensors
        states = torch.cat(states).to(device)
        actions = torch.tensor(actions, device=device).unsqueeze(1) # adding a column dimension
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        # Identify which transitions are non-terminal
        non_final_mask = torch.tensor([s is not None for s in next_states], device = device, dtype = torch.bool)
        non_final_next_states = torch.cat([s for s in next_states if s is not None]).to(device)
        # Q(s, a) from policy network
        current_q_values = policy_dqn(states).gather(1, actions)
        
        # Basically compute max_a Q(s', a) from target network for non-terminal states??
        
        next_q_values = torch.zeros(self.mini_batch_size, device=device) # target values
        next_q_values[non_final_mask] = target_dqn(non_final_next_states).max(1)[0].detach()
        
        # The BELLMAN Equation here: target Q = r + gamma * max_a Q(s', a)
        expected_q_values = (next_q_values.unsqueeze(1)*self.discount_factor_g) + rewards
        
        #    Compute the loss between current Q and Target Q
        loss = self.loss_fn(current_q_values, expected_q_values)

        # Optimize the model, so Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_to_tensor(self, state:int, device):
        # Here I normalize inputs for better learning stability. Might not be needed...
        
        state[0] /= 5.0 #pedestrian position (0-5) ->[0,1]
        state[-1] /= 50.0 # time elapsed (0-50) -> scaled to [0,1]
        # convert numpy array state to 2D torch tensor (1, num_features)
        return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)#squeeze 0 adds a batch dim


if __name__ == '__main__':
    # registering the environment
    register(
        id="PedestrianCrossing-v0",
        entry_point="pedestrian_crossing_env:PedestrianCrossingEnv"
        )
    pedestrian_crossing = SafePedestrianDQL()
    pedestrian_crossing.train(episodes=35000) #20000) #13000)
    #safe_pedestrian.evaluate() # <- this will include things like: safety rate, average wait time, etc.

