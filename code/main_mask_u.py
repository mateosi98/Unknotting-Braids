import os
import sys
import gym
import pandas as pd
import numpy as np
general_dir_name = ''
dir_name = ''
code_dir = f"/code"
sys.path.insert(0, dir_name+code_dir)

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from importlib import reload
import braid_knot_env_mask_rew_0_1_limexp_u
reload(braid_knot_env_mask_rew_0_1_limexp_u)
from braid_knot_env_mask_rew_0_1_limexp_u import BraidKnotEnv


s = 3
l = 20
data_dir = f"/data_rl/"
file_dir = f"pure_{s}s_{l}l"
models_dir = f"/models"
logs_dir = f"/logs"
method_dir = f'/MPPO'
version = 0


try:
    data = pd.read_csv(general_dir_name+data_dir+file_dir+"_u_fake.csv").drop(['Unnamed: 0'],axis=1)
    train_set = data.iloc[:,:l].values.tolist()
    u_set = data.u_fake.values.tolist()
except:
    print('NO DATA FOUND')
len(train_set)
len(u_set)
max(u_set)

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask

env = BraidKnotEnv(braid_set = train_set, u_set = u_set, braid_strands=s, e = 0.1, m=int(l*s))
dir_env = env.__module__+''
env = ActionMasker(env, mask_fn)  # Wrap to enable masking
check_env(env)

version_dir = f"/{s}s_{l}l/{dir_env}"

if not os.path.exists(general_dir_name + models_dir):
    os.mkdir(general_dir_name + models_dir)
if not os.path.exists(general_dir_name + models_dir+ f"/{s}s_{l}l"):
    os.mkdir(general_dir_name + models_dir+ f"/{s}s_{l}l")
if not os.path.exists(general_dir_name + models_dir+ version_dir):
    os.mkdir(general_dir_name + models_dir+ version_dir)
if not os.path.exists(general_dir_name + logs_dir):
    os.mkdir(general_dir_name + logs_dir)
if not os.path.exists(general_dir_name + logs_dir + f"/{s}s_{l}l"):
    os.mkdir(general_dir_name + logs_dir + f"/{s}s_{l}l")
if not os.path.exists(general_dir_name + logs_dir + version_dir):
    os.mkdir(general_dir_name + logs_dir + version_dir)

epocs = 10
TIMESTEPS = len(train_set)*epocs
model_mppo = MaskablePPO(MaskableActorCriticPolicy, env,policy_kwargs={'net_arch':[256, 512, 512, 512, 256]},tensorboard_log=general_dir_name+logs_dir+version_dir, learning_rate=10e-6, verbose=1)
# model_dqn = DQN('MlpPolicy', env, policy_kwargs={'net_arch':[256, 512, 512, 512, 256]}, verbose=1, tensorboard_log=general_dir_name+logs_dir+version_dir, learning_rate=10e-6)
# model_ppo = PPO.load("/Unknotting/models/3s_20l/braid_knot_env_rew_0_1/40001102.zip")


for i in range(100): 
    model_mppo.learn(total_timesteps=TIMESTEPS, progress_bar=True,tb_log_name='M_PPO', reset_num_timesteps=False, log_interval=1)
    model_mppo.save(f"{general_dir_name+models_dir+version_dir}/{model_mppo._total_timesteps}")
    # model_dqn.learn(total_timesteps=TIMESTEPS, progress_bar=True,tb_log_name='DQN', reset_num_timesteps=False, log_interval=1)
    # model_dqn.save(f"{general_dir_name+models_dir+version_dir}/{model_dqn._total_timesteps}_DQN")
    # model_ppo_e1.learn(total_timesteps=TIMESTEPS, progress_bar=True,tb_log_name='PPO', reset_num_timesteps=False, log_interval=1)
    # model_ppo_e1.save(f"{general_dir_name+models_dir+version_dir_e1}/{model_ppo_e1._total_timesteps}")
    
# tensorboard --logdir /Unknotting/logs/3s_20l/braid_knot_env_mask_rew_0_1_limexp_u/M_PPO_0

########################################################################################################
# TESING
########################################################################################################


# model_dqn = DQN.load("/Unknotting/models/3s_10l/braid_knot_env_rew_0_1/15999774_DQN.zip")
model_mppo = MaskablePPO.load("/Unknotting/models/3s_10l/braid_knot_env_mask_rew_0_1_limexp_u/68124640.zip")
# model_ppo = PPO.load("/Unknotting/models/3s_10l/braid_knot_env_rew_0_1/32013410.zip")
s, l = 3,10
data_dir = f"/data/"
file_dir = f"pure_{s}s_{l}l"
try:
    results_df = pd.read_csv(general_dir_name+data_dir+file_dir+'_results.csv').drop(['Unnamed: 0'],axis=1)
    results_df = results_df.loc[:10000]
    test_set = results_df.iloc[:, :l].values.tolist()
except:
    print('NO DATA FOUND')
env = BraidKnotEnv(braid_set = [10*[0]],braid_strands=s, e = 0, m=1000)
len(test_set)


def test_agents(models =[],data=[],max_steps=100):
    info_list = [list() for _ in range(len(models))]
    info_list_index = 0
    for model in models:
        index = 0
        for braid in data:
            steps = 0
            done = False
            env.reset(np.array(braid))
            while not done and steps < max_steps:
                pred_action = int(model.predict(env.state)[0])
                state, reward, _, info = env.step(pred_action)
                braid = np.copy(state)
                steps += 1
                done = reward == 1
            if not done:
                steps = False
            info_list[info_list_index].append([index, model.__class__.__name__, done, steps])
            index += 1
            print(info_list_index+1,'_',index)
        info_list_index += 1
    return info_list


results = test_agents(models = [model_mppo],data = test_set)
# df_results = pd.DataFrame(range(10001), columns=['INDEX'])

for i in range(len(results)):
    aux_df = pd.DataFrame(results[i],  columns=['INDEX','MODEL', 'DONE', 'STEPS'])
    results_df[results[i][0][1]] = aux_df['STEPS'].values.tolist()
results_df.greedy =results_df.greedy.astype(int) 

print('Unknoting_mean: ', results_df.unknoting_number.describe())
print('GREEDY_mean: ', results_df.greedy.describe())
print('DQN_mean: ', results_df.DQN.describe())
print('PPO_mean: ', results_df.PPO.describe())
print('MaskablePPO_mean: ', results_df.MaskablePPO.describe())

results_df.to_csv(general_dir_name+data_dir+file_dir+'_results.csv')

m=10
for i in range(m - 1, -1, -1):
    print(i-1)

#######


model_dqn = DQN.load("/Unknotting/models/3s_10l/braid_knot_env_rew_0_1/15999774_DQN.zip")
model_mppo = MaskablePPO.load("/Unknotting/models/3s_10l/sb3_contrib.common.wrappers.action_masker/22008930.zip")
model_ppo = PPO.load("/Unknotting/models/3s_10l/braid_knot_env_rew_0_1/32013410.zip")
s, l = 3,10
data_dir = f"/data/"
file_dir = f"pure_{s}s_{l}l"
try:
    results_detail_df = pd.read_csv(general_dir_name+data_dir+file_dir+'_results.csv').drop('Unnamed: 0',axis=1)
    results_detail_df = results_df.loc[:10000]
    test_set = results_detail_df.iloc[:, :l].values.tolist()
except:
    print('NO DATA FOUND')
env = BraidKnotEnv(braid_set = [10*[0]],braid_strands=s, e = 0, m=1000)
# int(model_dqn.predict([, -2, -2, -2, -2, -1,  2, -2, -1,  2])[0])
len(test_set)

def test_agents(models=[],data=[],max_steps=100):
    info_list = []
    index = 0
    for braid in data:
        n_steps = 0
        steps = []
        done = False
        env.reset(np.array(braid))
        # print(type(steps), type(n_steps))
        for model in models:
            while not done and n_steps < max_steps:
                pred_action = int(model.predict(env.state)[0])
                state, reward, _, info = env.step(pred_action)
                steps.append(pred_action)
                braid = np.copy(state)
                n_steps += 1
                done = reward == 1
            info_list.append([index,model.__class__.__name__ , done, n_steps, steps])
        index += 1
        print(index)
    return info_list



results_detail = test_agents(models = [model_dqn,model_ppo, model_mppo],data = test_set)
# df_results_detail = pd.DataFrame(results_ppo, columns=['INDEX','DONE', 'STEPS'])
# results_detail_df['ppo'] = df_results_ppo.loc[:10000].STEPS.values.tolist()

