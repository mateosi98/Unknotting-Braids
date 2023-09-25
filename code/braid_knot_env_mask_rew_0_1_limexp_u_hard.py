from gym import Env
from gym.spaces import Box, Discrete
import numpy as np
import random
import pandas as pd
from importlib import reload
import kl
reload(kl)


class BraidKnotEnv(Env):

  def __init__(self, braid_strands=3, braid_set=[np.zeros(8, dtype=np.int32)], u_set =[10], e=0.05, d=0.99999, m=1000):
    self.length = len(braid_set[0])
    self.strands = braid_strands
    self.set = braid_set
    self.u_set = [self.length] * len(self.set)
    self.INITIAL_SET = [_ for _ in braid_set]
    self.INITIAL_U_SET = [_ for _ in u_set]
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
    if len(self.u_set) > 0 or len(self.set) > 0:
      rand = random.randint(0,len(self.u_set)-1)
      self.u = int(self.u_set[rand])
      self.state = np.array(self.set[rand])
      self.set.remove(self.set[rand])
      self.u_set = [self.u_set[i] for i in range(len(self.u_set)) if i != rand].copy()
    else:
      self.set = [_ for _ in self.INITIAL_SET]
      self.u_set = [_ for _ in self.INITIAL_U_SET]
      rand = random.randint(0,len(self.set)-1)
      self.u = int(self.u_set[rand])
      self.state = np.array(self.set[rand])
      self.set.remove(self.set[rand])
      self.u_set = [self.u_set[i] for i in range(len(self.u_set)) if i != rand].copy()
    # Constant initial state
    self.INITIAL_STATE = np.copy(self.state)
    # Time starts in zero
    self.time = 0
    self.action = -1
    # KITTY LACEY NEW
    self.kl_new = kl.kitty_lacey(self.strands, self.length)
    # Initialize the action log
    self.action_log = pd.DataFrame({'States':[list(self.state).copy()], 'Actions':[list() for x in range(1)]})
    self.aux_action_mask = np.ones(self.action_space.n, dtype=bool)
    self.valid_action_mask = np.ones(self.action_space.n, dtype=bool)
    try:
      self.state_index = self.action_log[self.action_log['States'].apply(lambda x: x == list(self.state))].index[0]
      self.state_actions = self.action_log.loc[self.state_index]['Actions']
    except:
      self.state_index = 0 
      self.state_actions = []
    self.action_valid = True

  def action_masks(self):
    self.state_actions.append(self.action)
    self.action_log.loc[self.state_index]['Actions'] = self.state_actions 
    self.valid_action_mask = np.ones(self.action_space.n, dtype=bool)
    self.state_index = -1
    try:
      self.state_index = self.action_log[self.action_log['States'].apply(lambda x: x == list(self.state))].index[0]
      self.state_actions = self.action_log.loc[self.state_index]['Actions']
      self.valid_action_mask[self.state_actions] = False
    except:
      self.state_index = len(self.action_log)
      self.state_actions = []
      self.action_log.loc[self.state_index] = [list(self.state).copy(), self.state_actions.copy()]
    return list(self.valid_action_mask)

  def step(self, action): 
    x = self.state
    done = False
    # A bit of exploration (e-greedy) only in valid actions
    if random.uniform(0,1) < self.epsilon:
      true_indices = np.where(self.valid_action_mask)[0]
      action = np.random.choice(true_indices)
    self.action_valid = self.valid_action_mask[action]
    if action < self.length:
      x[action] = -x[action]
    else:
      pass
    self.time += 1 
    self.action = 0 + action
    # Check if braid is untangled & calculate reward
    reward = 0
    if self.kl_new.is_trivial_23(x):
        reward = 1
        done = True
    elif self.time > self.u:
        done = True
    # Set placeholder for info
    info = {}
    return self.state, reward, done, info

  def reset(self, given=[]):
    # Reset the braid
    if len(given)>0:
      self.state = np.copy(np.array(given))
    else:
      if len(self.u_set) > 0 or len(self.set) > 0:
        rand = random.randint(0,len(self.u_set)-1)
        self.u = int(self.u_set[rand])
        self.state = np.array(self.set[rand])
        self.set.remove(self.set[rand])
        self.u_set = [self.u_set[i] for i in range(len(self.u_set)) if i != rand].copy()
      else:
        self.set = [_ for _ in self.INITIAL_SET]
        self.u_set = [_ for _ in self.INITIAL_U_SET]
        rand = random.randint(0,len(self.set)-1)
        self.u = int(self.u_set[rand])
        self.state = np.array(self.set[rand])
        self.set.remove(self.set[rand])
        self.u_set = [self.u_set[i] for i in range(len(self.u_set)) if i != rand].copy()
    self.time = 0
    self.epsilon *= self.decay

    self.action_log = pd.DataFrame({'States':[list(self.state).copy()], 'Actions':[list() for x in range(1)]})
    self.aux_action_mask = np.ones(self.action_space.n, dtype=bool)
    self.valid_action_mask = np.ones(self.action_space.n, dtype=bool)
    try:
      self.state_index = self.action_log[self.action_log['States'].apply(lambda x: x == list(self.state))].index[0]
      self.state_actions = self.action_log.loc[self.state_index]['Actions']
    except:
      self.state_index = 0 
      self.state_actions = []

    # Return the state after reset (initial state)
    return self.state

  def render(self):
    pass
