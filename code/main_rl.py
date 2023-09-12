import os
import sys
import pandas as pd
import numpy as np
general_dir_name = ''
dir_name = ''
code_dir = f"/code"
sys.path.insert(0, dir_name+code_dir)

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from importlib import reload
import braid_knot_env_rew_0_1
reload(braid_knot_env_rew_0_1)
from braid_knot_env_rew_0_1 import BraidKnotEnv
import word_knot_env
reload(word_knot_env)
from word_knot_env import WordKnotEnv

s = 3
l = 20
data_dir = f"/data_rl/"
file_dir = f"pure_{s}s_{l}l"
models_dir = f"/models"
logs_dir = f"/logs"
method_dir = f'/DQN'
version = 0
version_2 = 0

try:
    train_set = pd.read_csv(general_dir_name+data_dir+file_dir).drop('Unnamed: 0',axis=1).values.tolist()
except:
    print('NO DATA FOUND')

# env = WordKnotEnv(braid_set = set, braid_strands=s, e = 0.1, m=1000)
env = BraidKnotEnv(braid_set = train_set, braid_strands=s, e = 0.1, m=1000)
check_env(env)
# check_env(env_e1)

version_dir = f"/{s}s_{l}l/{env.__module__}"
# version_dir_e1 = f"/{s}s_{l}l/{env_e1.__module__}"

if not os.path.exists(general_dir_name + models_dir):
    os.mkdir(general_dir_name + models_dir)
if not os.path.exists(general_dir_name + models_dir+ f"/{s}s_{l}l"):
    os.mkdir(general_dir_name + models_dir+ f"/{s}s_{l}l")
if not os.path.exists(general_dir_name + models_dir+ version_dir):
    os.mkdir(general_dir_name + models_dir+ version_dir)
if not os.path.exists(general_dir_name + models_dir+ version_dir+method_dir):
    os.mkdir(general_dir_name + models_dir+ version_dir+method_dir)
# if not os.path.exists(general_dir_name + models_dir+ version_dir_e1):
#     os.mkdir(general_dir_name + models_dir+ version_dir_e1)
if not os.path.exists(general_dir_name + logs_dir):
    os.mkdir(general_dir_name + logs_dir)
if not os.path.exists(general_dir_name + logs_dir + f"/{s}s_{l}l"):
    os.mkdir(general_dir_name + logs_dir + f"/{s}s_{l}l")
if not os.path.exists(general_dir_name + logs_dir + version_dir):
    os.mkdir(general_dir_name + logs_dir + version_dir)
# if not os.path.exists(general_dir_name + logs_dir + version_dir_e1):
#     os.mkdir(general_dir_name + logs_dir + version_dir_e1)

# env = Monitor(env, filename=dir_name+logs_dir+version_dir, allow_early_resets=True)

epocs = 10
TIMESTEPS = len(train_set)*epocs
model_dqn = DQN('MlpPolicy', env, policy_kwargs={'net_arch':[256, 512, 512, 512, 256]}, verbose=1, tensorboard_log=general_dir_name+logs_dir+version_dir, learning_rate=10e-6)
# model_ppo = PPO.load("/Users/mateosallesize/Documents/SRO/Braids/Unknotting/models/3s_20l/braid_knot_env_rew_0_1/40001102.zip")


for i in range(100): 
    model_dqn.learn(total_timesteps=TIMESTEPS, progress_bar=True,tb_log_name='DQN', reset_num_timesteps=False, log_interval=1)
    model_dqn.save(f"{general_dir_name+models_dir+version_dir}/{model_dqn._total_timesteps}_DQN")
    # model_ppo_e1.learn(total_timesteps=TIMESTEPS, progress_bar=True,tb_log_name='PPO', reset_num_timesteps=False, log_interval=1)
    # model_ppo_e1.save(f"{general_dir_name+models_dir+version_dir_e1}/{model_ppo_e1._total_timesteps}")
    

##########################################################################################
##########################################################################################
#### SOME TESTING, general testing in main_greedy_and_testing
##########################################################################################
##########################################################################################

# model_dqn = DQN.load("/Users/mateosallesize/Documents/SRO/Braids/Unknotting/models/3s_10l/braid_knot_env_rew_0_1/15999774_DQN.zip")
model_ppo = PPO.load("/Users/mateosallesize/Documents/SRO/Braids/Unknotting/models/3s_20l/braid_knot_env_rew_0_1/40001102.zip")
s, l = 3,20
data_dir = f"/data/"
file_dir = f"pure_{s}s_{l}l"
try:
    test_set = pd.read_csv(general_dir_name+data_dir+file_dir).drop('Unnamed: 0',axis=1).values.tolist()
except:
    print('NO DATA FOUND')
env = BraidKnotEnv(braid_set = test_set,braid_strands=s, e = 0, m=1000)
# int(model_dqn.predict([-2, -2, -2, -2, -2, -1,  2, -2, -1,  2])[0])

def test_agent(model,data=[],max_steps=100):
    info_list = []
    index = 0
    for braid in data:
        steps = 1
        done = False
        env.reset(np.array(braid))
        while not done and steps < max_steps:
            pred_action = int(model.predict(env.state)[0])
            state, reward, _, info = env.step(pred_action)
            braid = np.copy(state)
            steps += 1
            done = reward == 1
        info_list.append([index, done, steps])
        index += 1
        print(index)
    return info_list



results_ppo = test_agent(model_ppo, test_set, 1000)

df_results_ppo = pd.DataFrame(results_ppo, columns=['INDEX','DONE', 'STEPS'])
df_results_ppo[df_results_ppo.DONE==True]
df_results_ppo[df_results_ppo.STEPS==2]
print('PPO_rew_0_1')
print('Mean_steps: ',df_results_ppo[df_results_ppo.DONE==True]['STEPS'].mean())
print('STD_steps: ',df_results_ppo[df_results_ppo.DONE==True]['STEPS'].std())
print('Accuracy: ',len(df_results_ppo[df_results_ppo.DONE==True]) / len(test_set) * 100,'%')


results_dqn = test_agent(model_ppo, test_set, 1000)

df_results_dqn = pd.DataFrame(results_dqn, columns=['INDEX','DONE', 'STEPS'])
df_results_dqn[df_results_dqn.DONE==True]
df_results_dqn[df_results_dqn.STEPS>20]
print('DQN_rew_0_1')
print('Mean_steps: ',df_results_dqn[df_results_dqn.DONE==True]['STEPS'].mean())
print('STD_steps: ',df_results_dqn[df_results_dqn.DONE==True]['STEPS'].std())
print('Accuracy: ',len(df_results_dqn[df_results_dqn.DONE==True]) / len(test_set) * 100,'%')