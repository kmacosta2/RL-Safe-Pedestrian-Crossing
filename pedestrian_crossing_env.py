import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class PedestrianCrossingEnv(gym.Env):
    """
    the custom OpenAI Gymnasium environment for pedestrian crossing.
    The pedestrian must cross four lanes of traffic while avoiding getting hit by vehicles.
    The goal is to reach the other sidewalk safely (and efficiently).
    """
    def __init__(self):
        super(PedestrianCrossingEnv, self).__init__()

        # defining the number of lanes and pedestrian positions
        self.num_lanes = 4  # so, four lanes
        self.num_positions = self.num_lanes + 2  # remember -> four lanes + two sidewalks

        # defining the state space      which a state consists of:
        # [pedestrian position, traffic lane 1, traffic lane 2, traffic lane 3, traffic lane 4, time elapsed]
        self.observation_space = spaces.Box(
            low=np.array([0] + [0]*self.num_lanes + [0]),
            high=np.array([self.num_positions - 1] + [1]*self.num_lanes + [50]),
            dtype=np.float32
        )
        # defining the action space (0: wait, 1: move forward one lane, 2: move backward one lane)
        self.action_space = spaces.Discrete(3)
        self.reset()

    def reset(self, seed=None, options=None):
        """
        resets the environment for a new episode.
        """
        super().reset(seed=seed)
        self.pedestrian_position = 0  # start on the initial sidewalk (Lane 0)
        self.time_elapsed = 0
        self.done = False
        self.traffic_lanes = self._generate_traffic()

        return self._get_state(), {}

    def step(self, action):
        """
        applies an action, updates the state, and calculates reward.
        """
        self.time_elapsed += 1
        self.traffic_lanes = self._generate_traffic()
        # perform action: forward moves the pedestrian one lane, backward moves one lane back
        if action == 1:  # move forward (one lane)
            self.pedestrian_position = min(self.pedestrian_position + 1, self.num_positions - 1)
        elif action == 2 and self.pedestrian_position > 0 and self.pedestrian_position != 0:
            self.pedestrian_position -= 1

        # determine current lane (-1 means sidewalk here)
        current_lane = self.pedestrian_position - 1 if 1 <= self.pedestrian_position <= self.num_lanes else -1

        # collision detection: pedestrian is hit if in a lane with a vehicle
        if current_lane >= 0 and self.traffic_lanes[current_lane] == 1:
            reward = -20 #-30 #-40 #-50
            self.done = True
        elif self.pedestrian_position == self.num_positions - 1:
            reward = 10             # successfully crossed the road
            self.done = True
            
        #elif action == 1: # <-- added to encourage survival
        #    reward = 1
        else:
            reward = -1  # providing a time penalty to encourage faster crossing

        if self.time_elapsed >= 50:        # check if time limit reached
            self.done = True

        return self._get_state(), reward, self.done, False, {}

    def _generate_traffic(self):
        """
        generates traffic states for each lane with a 50% chance of a vehicle .
        """
        # [random.choice([0, 1]) for _ in range(self.num_lanes)] # reduce to 0.2-0.4  random.binomial()
        #[np.random.binomial(n=1, p=0.1) for _ in range(self.num_lanes)]
        return [np.random.binomial(n=1, p=0.025) for _ in range(self.num_lanes)]

    def _get_state(self):
        """
        returns the current state as a NumPy array.
        - pedestrian position (0 to 5)
        - lane states (0 for no car, 1 for car present)
        - time elapsed
        """
        return np.array([self.pedestrian_position] + self.traffic_lanes + [self.time_elapsed], dtype=np.float32)

    def render(self):
        """
        prints the current state variables and environment details.
        """
        print(f"Pedestrian Lane: {self.pedestrian_position} ({'Sidewalk' if self.pedestrian_position == 0 or self.pedestrian_position == self.num_positions - 1 else 'Lane'})")
        print(f"Current Lane: {'N/A' if self.pedestrian_position == 0 or self.pedestrian_position == self.num_positions - 1 else self.pedestrian_position}")
        print(f"Traffic Lanes: {self.traffic_lanes}")
        print(f"Time Elapsed: {self.time_elapsed}")
        print(f"Done: {self.done}")
        print("="*44)

# Register environment to Gymnasium
#gym.register(
#    id="PedestrianCrossing-v0",
#    entry_point=PedestrianCrossingEnv,
#)

#if __name__ == "__main__":
#    env = PedestrianCrossingEnv() # testing the environment with a random agent
#    obs, _ = env.reset()
#
#    done = False
#    total_reward = 0
#    print("\nAction space (0: wait, 1: move forward, 2: move backward)\n")
#    while not done:
#        action = env.action_space.sample()
#        obs, reward, done, _, _ = env.step(action)
#        total_reward += reward
#        print(f"Action: {action}")
#        env.render()
#        print(f'\t\t\tCurrent Reward: {reward}')

#    print("Episode finished. Total Reward:", total_reward)