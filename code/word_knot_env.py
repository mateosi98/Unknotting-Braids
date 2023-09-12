from gym import Env
from gym.spaces import Box, Discrete
import numpy as np
import random
from importlib import reload
import kl
reload(kl)


class WordKnotEnv(Env):

  def __init__(self, braid_strands=3, braid_set=[np.zeros(8, dtype=np.int32)], e=0.05, d=0.99999, m=1000):
    self.length = len(braid_set[0])
    self.strands = braid_strands
    self.set = braid_set
    self.INITIAL_SET = [_ for _ in braid_set]
    self.epsilon = e
    self.decay = d
    self.max_time = m
    # There are length moves
    self.action_space = Discrete(self.length)
    # Set the observation space as [-k,...,-k],...,[k,...,k]
    low=np.ones([self.length], dtype=np.int32)*(1-self.strands)
    high=np.ones([self.length], dtype=np.int32)*(self.strands-1)
    self.observation_space = Box(low, high, dtype=int)
    # Initialize the braid as the initial tangled given braid
    if len(self.set) > 0:
      rand = random.randint(0,len(self.set)-1)
      self.state = np.array(self.set[rand])
      self.set.remove(self.set[rand])
    else:
      self.set = [_ for _ in self.INITIAL_SET]
      rand = random.randint(0,len(self.set)-1)
      self.state = np.array(self.set[rand])
      self.set.remove(self.set[rand])
    # Constant initial state
    self.INITIAL_STATE = np.copy(self.state)
    # Time starts in zero
    self.time = 0
    # KITTY LACEY NEW
    self.kl = kl.kitty_lacey(self.strands, self.length)


  def step(self, action): 
    x = self.state
    aw = self.kl.autowrithe(x)
    done = False
    # A bit of exploration (e-greedy)
    if random.uniform(0,1) < self.epsilon:
      action = self.action_space.sample()
    # action is an int in (0,4*(lenght-1))
    if action < self.length:
      x[action] = -x[action]
    else:
      pass
    self.time += 1 
    # Check if braid is untangled & calculate reward
    aw_2 = self.kl.autowrithe(x)
    if aw_2 == 1:
      reward = 0
      done = True
    elif self.time >= self.max_time - 1:
      reward = -100
      done = True
    else:
      if aw_2 > aw:
        reward = -np.log(aw_2-aw) -1
      else:
        reward = -1
    # Set placeholder for info
    info = {}
    # Return step information
    return self.state, reward, done, info

  def reset(self, same= False):
    # Reset the braid
    if same:
      self.state = np.copy(self.INITIAL_STATE)
    else:
      if len(self.set) > 0:
        rand = random.randint(0,len(self.set)-1)
        self.state = np.array(self.set[rand])
        self.set.remove(self.set[rand])
      else:
        self.set = [_ for _ in self.INITIAL_SET]
        rand = rand = random.randint(0,len(self.set)-1)
        self.state = np.array(self.set[rand])
        self.set.remove(self.set[rand])
    self.time = 0
    self.epsilon *= self.decay
    # Return the state after reset (initial state)
    return self.state

  def render(self):
    pass
