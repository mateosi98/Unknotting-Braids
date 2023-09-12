from gym import Env
from gym.spaces import Box, Discrete
import numpy as np
import random
from importlib import reload
import kl
reload(kl)


class E1KnotEnv(Env):

  def __init__(self, braid_strands=3, braid_set=[np.zeros(8, dtype=np.int32)], e=0.05, d=0.99999, m=1000):
    self.length = len(braid_set[0])
    self.strands = braid_strands
    # self.set = braid_set
    self.e1_set = [self.transform_braid_to_matrix(b) for b in braid_set]
    self.INITIAL_SET = [_ for _ in self.e1_set]
    self.epsilon = e
    self.decay = d
    self.max_time = m
    # There are length moves
    self.action_space = Discrete(self.length)
    # Set the observation space as [-k,...,-k],...,[k,...,k]
    low=np.ones([self.length*(self.strands-1)], dtype=np.int32)*(1-self.strands)
    high=np.ones([self.length*(self.strands-1)], dtype=np.int32)*(self.strands-1)
    self.observation_space = Box(low, high, dtype=int)
    # Initialize the braid as the initial tangled given braid
    if len(self.e1_set) > 0:
      rand = random.randint(0,len(self.e1_set)-1)
      self.state = np.copy(self.e1_set[rand])
      self.e1_set = [i for i in self.e1_set if not np.array_equal(i, self.e1_set[rand])]
    else:
      self.e1_set = [_ for _ in self.INITIAL_SET]
      rand = random.randint(0,len(self.e1_set)-1)
      self.state = np.copy(self.e1_set[rand])
      self.e1_set = [i for i in self.e1_set if not np.array_equal(i, self.e1_set[rand])]
    # Constant initial state
    self.INITIAL_STATE = np.copy(self.state)
    # Time starts in zero
    self.time = 0
    # KITTY LACEY NEW
    self.kl_new = kl.kitty_lacey(self.strands, self.length)
  
  def transform_braid_to_matrix(self, braid):
    braid_matrix = np.zeros(shape=(self.strands-1,self.length), dtype=int)
    crossing = 0
    for i in braid:
      if i != 0:
        row = abs(i)-1
        number = i/(row+1)
        braid_matrix[row,crossing] = number
      crossing += 1
    return braid_matrix.reshape(self.length*(self.strands-1))

  def transform_matrix_to_braid(self, e1):
    e1 = e1.reshape(self.strands-1,self.length)
    for j in range(self.strands-1):  
      e1[j] *= j
    if self.strands > 2:
      return sum(e1)
    else:
      return e1
# a = np.array([[0,0,0], [1,1,1]])
# sum(a)
  def step(self, action): 
    x = np.copy(self.state)
    done = False
    # A bit of exploration (e-greedy)
    if random.uniform(0,1) < self.epsilon:
      action = self.action_space.sample()
    # action is an int in (0,4*(lenght-1))
    if action < self.length:
      for j in range(self.strands-1):
        x[j * self.length + action] = -x[j * self.length + action]
        # x[self.length + action] = -x[self.length + action]
    else:
      pass
    self.time += 1 
    # Check if braid is untangled & calculate reward
    reward = -1
    if self.strands > 2:
      y = self.transform_matrix_to_braid(x)
    else:
      y = np.copy(x)
    if self.kl_new.is_trivial(y):
      reward = 0
      done = True
    elif self.time >= self.max_time - 1:
      done = True
    # Set placeholder for info
    info = {}
    self.state = np.copy(x)
    # Return step information
    return self.state, reward, done, info

  def reset(self, same= False):
    # Reset the braid
    if same:
      self.state = np.copy(self.INITIAL_STATE)
    elif len(self.e1_set) > 0:
      rand = random.randint(0,len(self.e1_set)-1)
      self.state = np.copy(self.e1_set[rand])
      self.e1_set = [i for i in self.e1_set if not np.array_equal(i, self.e1_set[rand])]
    else:
      self.e1_set = [_ for _ in self.INITIAL_SET]
      rand = random.randint(0,len(self.e1_set)-1)
      self.state = np.copy(self.e1_set[rand])
      self.e1_set = [i for i in self.e1_set if not np.array_equal(i, self.e1_set[rand])]
    self.time = 0
    self.epsilon *= self.decay
    # Return the state after reset (initial state)
    return self.state

  def render(self):
    pass
