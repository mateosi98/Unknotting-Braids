import os
import sys
import pandas as pd
import numpy as np
from random import shuffle
import kitty_lacey
general_dir_name = ''
dir_name = ''
code_dir = f"/code"
sys.path.insert(0, dir_name+code_dir)



from importlib import reload
reload(kitty_lacey)


s = 3
l = 20
data_dir = f"/data_rl/"
file_dir = f"pure_{s}s_{l}l"


def greedy_unknotting(braid, steps, stack,action_log=[], length = l, strands = s, max_steps = l):
    # print(braid, steps, stack)
    if len(stack) > 0 and braid in stack[:len(stack)-1]: # cycled back
        action_log.append('c')
        stack.append(braid)
        return False, max_steps +3 , stack,action_log, False
    elif kitty_lacey.is_trivial_June_2023(braid,strands): # braid is trvial
        # print('2')
        return braid, steps, stack,action_log, True
    elif (not kitty_lacey.is_trivial_June_2023(braid,strands)) and steps <= length**(strands+1):
        steps += 1
        stack.append(braid)
        complexities = []
        comp = kitty_lacey.complexity(braid,strands)
        for crossing in range(len(braid)):
            aux_braid = braid.copy()
            aux_braid[crossing] *= -1
            complexity = kitty_lacey.complexity(aux_braid,strands)
            complexities.append(complexity)
            if complexity == 1: # this leaf is trivial
                action_log.append(crossing)
                return aux_braid, steps, stack,action_log, True
        if comp < min(complexities): # we have reached a dead-end
            action_log.append('d')
            return braid, max_steps +2, stack,action_log, False
        best_crossing = np.argmin(complexities)
        aux_braid = braid.copy()
        aux_braid[best_crossing] *= -1
        action_log.append(best_crossing)
        return greedy_unknotting(aux_braid, steps, stack,  action_log, length, strands)
    else: # reached max steps
        return False, max_steps +1, True,action_log, False

answers = []
kitty_lacey.set_params(s, l)
try:
    results_df = pd.read_csv(general_dir_name+data_dir+file_dir+'').drop('Unnamed: 0',axis=1)
    results_df = results_df.loc[:]
    test_set = results_df.iloc[:, :l].values.tolist()
except:
    print('NO DATA FOUND')
len(test_set)

def test_greedy(data=[],fake=False):
    info_list, index = [], 0
    for braid in data:
        # greedy_unknotting(b,0,len(b), 3)[0]
        braid, steps, stack,action_log, triv = greedy_unknotting(braid,0,[],[],len(braid), s)
        if fake and steps >= l//2:
            steps = l//2
        info_list.append([index,triv,steps])
        index += 1
        print(index)
    return info_list

## NORMAL GREEDY RESULTS
results_greedy = test_greedy(test_set)
df_results_greedy = pd.DataFrame(results_greedy, columns=['INDEX','DONE', 'STEPS'])
results_df['greedy'] = df_results_greedy.loc[:10000].STEPS.values.tolist()
df_results_greedy[df_results_greedy.DONE==True].loc[:10000].STEPS.mean()
len(df_results_greedy[df_results_greedy.DONE==True]) /len(df_results_greedy)
results_df.to_csv(general_dir_name+data_dir+file_dir+'_results.csv')

## U_FAKE GREEDY RESULTS
# results_greedy = test_greedy(test_set,True)
# df_results_greedy = pd.DataFrame(results_greedy, columns=['INDEX','DONE', 'STEPS'])
# results_df['u_fake'] = df_results_greedy.STEPS.values.tolist()
# df_results_greedy[df_results_greedy.DONE==True].STEPS.mean()
# len(df_results_greedy[df_results_greedy.DONE==True]) /len(df_results_greedy)
# results_df.to_csv(general_dir_name+data_dir+file_dir+'_u_fake.csv')


df_results_greedy[df_results_greedy.DONE==True]

print('Mean_steps: ',df_results_greedy[df_results_greedy.DONE==True]['STEPS'].mean())

print('STD_steps: ',df_results_greedy[df_results_greedy.DONE==True]['STEPS'].std())

print("Accuracy: ",len(df_results_greedy[df_results_greedy.DONE==True]) / len(test_set)*100,"%")



##########################################################################################
##########################################################################################
#### TESTING
##########################################################################################
##########################################################################################

import pandas as pd
import numpy as np
from stable_baselines3 import PPO, DQN
from sb3_contrib.ppo_mask import MaskablePPO

from importlib import reload
import braid_knot_env_mask_rew_0_1
reload(braid_knot_env_mask_rew_0_1)
from braid_knot_env_mask_rew_0_1 import BraidKnotEnv

model_dqn = DQN.load("/Unknotting/models/3s_10l/braid_knot_env_rew_0_1/15999774_DQN.zip")
model_mppo = MaskablePPO.load("/Unknotting/models/3s_10l/sb3_contrib.common.wrappers.action_masker/22008930.zip")
model_mppo_u = MaskablePPO.load("/Unknotting/models/3s_10l/braid_knot_env_mask_rew_0_1_limexp_u/32440288.zip")
model_mppo_k2 = MaskablePPO.load("/Unknotting/models/3s_10l/braid_knot_env_mask_rew_0_1_limexp_u/k2/32440288.zip")
model_ppo = PPO.load("/Unknotting/models/3s_10l/braid_knot_env_rew_0_1/32013410.zip")
s, l = 3,10
data_dir = f"/data/"
file_dir = f"pure_{s}s_{l}l"

##########################################################################################
#### Detail
##########################################################################################

results_df = pd.read_csv(general_dir_name+data_dir+file_dir+'_results.csv').drop('Unnamed: 0',axis=1)
results_detail_df = results_df.loc[:9]
test_set = results_detail_df.iloc[:, :l].values.tolist()


env = BraidKnotEnv(braid_set = [10*[0]],braid_strands=s, e = 0, m=1000)
len(test_set)

def test_agents_detail(models=[],data=[],max_steps=100):
    info_list = []
    index = 0
    for braid in data:
        # print(type(steps), type(n_steps))
        for model in models:
            done = False
            aux_braid = braid.copy()
            env.reset(np.array(aux_braid))
            n_steps = 0    
            steps = []
            while not done and n_steps < max_steps:
                pred_action = int(model.predict(env.state)[0])
                state, reward, _, info = env.step(pred_action)
                steps.append(pred_action)
                aux_braid = np.copy(state)
                n_steps += 1
                done = reward == 1
            info_list.append([index,braid,model.__class__.__name__ , done, n_steps, steps])
        index += 1
        print(index)
    return info_list



results_detail = test_agents_detail(models = [model_dqn,model_ppo, model_mppo],data = test_set)
df_results_detail_rl = pd.DataFrame(results_detail, columns=['INDEX','BRAID','MODEL','DONE','N_STEPS','STEPS'])


def test_greedy_detail(data=[], max_steps=100):
    info_list, index = [], 0
    for braid in data:
        aux_braid = braid.copy()
        # greedy_unknotting(b,0,len(b), 3)[0]
        aux_braid, n_steps, stack,steps, triv = greedy_unknotting(aux_braid,0,[],[],len(braid), s)
        info_list.append([index,braid,'GREEDY',triv,n_steps,steps])
        index += 1
        print(index)
    return info_list

results_greedy_detail = test_greedy_detail(test_set,100)

df_results_detail_greedy = pd.DataFrame(results_greedy_detail, columns=['INDEX','BRAID','MODEL','DONE','N_STEPS','STEPS'])
df_results_detail = pd.concat([df_results_detail_rl, df_results_detail_greedy], ignore_index=True)
df_results_detail
df_results_detail.to_csv(general_dir_name+data_dir+file_dir+'_results_detailed.csv')

##########################################################################################
#### Environments (with MPPO)
##########################################################################################

model_mppo = MaskablePPO.load("/Unknotting/models/3s_10l/braid_knot_env_mask/32440288.zip")
model_mppo_0_1 = MaskablePPO.load("/Unknotting/models/3s_10l/braid_knot_env_mask_rew_0_1/30012524.zip")
model_mppo_0_1_le = MaskablePPO.load("/Unknotting/models/3s_10l/braid_knot_env_mask_rew_0_1_limexp/32013400.zip")
model_mppo_0_1_le_u = MaskablePPO.load("/Unknotting/models/3s_10l/braid_knot_env_mask_rew_0_1_limexp_u/32440288.zip")
model_mppo_0_1_le_k2 = MaskablePPO.load("/Unknotting/models/3s_10l/braid_knot_env_mask_rew_0_1_limexp_u/k2/32440288.zip")



s, l = 3,10
data_dir = f"/data/"
file_dir = f"pure_{s}s_{l}l"
try:
    results_df = pd.read_csv(general_dir_name+data_dir+file_dir+'').drop(['Unnamed: 0'],axis=1)
    results_df = results_df.loc[:10000]
    test_set = results_df.iloc[:, :l].values.tolist()
except:
    print('NO DATA FOUND')
env = BraidKnotEnv(braid_set = [l*[0]],braid_strands=s, e = 0, m=1000)
len(test_set)

def test_envs(models ={},data=[],max_steps=100):
    info_list = [list() for _ in range(len(models))]
    info_list_index = 0
    for model_name, model in models.items():
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
                steps = max_steps+1
            # info_list[info_list_index].append([index, model.__class__.__name__, done, steps])
            info_list[info_list_index].append([index, model_name, done, steps])
            index += 1
            print(info_list_index+1,'_',index)
        info_list_index += 1
    return info_list

models = {"braid_knot_env":model_mppo,"braid_knot_env_0_1":model_mppo_0_1,"braid_knot_env_0_1_limexp":model_mppo_0_1_le,"braid_knot_env_0_1_limexp_u":model_mppo_0_1_le_u,"braid_knot_env_0_1_limexp_k2":model_mppo_0_1_le_k2}
results = test_envs(models = models,data = test_set)
len(results)
results[1][0][1]

for i in range(len(results)):
    aux_df = pd.DataFrame(results[i],  columns=['INDEX','MODEL', 'DONE', 'STEPS'])
    results_df[results[i][0][1]] = aux_df['STEPS'].values.tolist()
  
for model_name in models.keys():
    print(model_name, results_df[results_df[str(model_name)] <=100][str(model_name)].describe())
    

results_df.to_csv(general_dir_name+data_dir+file_dir+'_results_env.csv')



##########################################################################################
#### General
##########################################################################################

s, l = 3,10
data_dir = f"/data/"
file_dir = f"pure_{s}s_{l}l"
try:
    results_df = pd.read_csv(general_dir_name+data_dir+file_dir+'_results.csv').drop(['Unnamed: 0'],axis=1)
    results_df = results_df.loc[:10000]
    test_set = results_df.iloc[:, :l].values.tolist()
except:
    print('NO DATA FOUND')
env = BraidKnotEnv(braid_set = [l*[0]],braid_strands=s, e = 0, m=1000)
model_mppo_u.predict(env.state)[0]
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
                steps = max_steps+1
            # info_list[info_list_index].append([index, model.__class__.__name__, done, steps])
            info_list[info_list_index].append([index, 'MaskablePPO_K2', done, steps])
            index += 1
            print(info_list_index+1,'_',index)
        info_list_index += 1
    return info_list


results = test_agents(models = [model_mppo_k2],data = test_set)
len(results)
aux_df = pd.DataFrame(results[0],  columns=['INDEX','MODEL', 'DONE', 'STEPS'])
results_df[results[0][0][1]] = aux_df['STEPS'].values.tolist()

for i in range(len(results)):
    aux_df = pd.DataFrame(results[i],  columns=['INDEX','MODEL', 'DONE', 'STEPS'])
    results_df[results[i][0][1]] = aux_df['STEPS'].values.tolist()
  
print('Unknoting_mean: ', results_df.unknoting_number.describe())
print('GREEDY_mean: ', results_df[results_df['greedy'] <=100].greedy.describe())
len(results_df[results_df['greedy'] <=100])/10001
# print('DQN_mean: ', results_df.DQN.describe())
print('PPO_mean: ', results_df.PPO[results_df['PPO'] <=100].describe())
len(results_df[results_df['PPO'] <=100])/10001
print('MaskablePPO_mean: ', results_df[results_df['MaskablePPO'] <=100].MaskablePPO.describe())
len(results_df[results_df['MaskablePPO'] <=100])/10001
print('MaskablePPO_U_mean: ', results_df[results_df['MaskablePPO_U'] <=100].MaskablePPO_U.describe())
len(results_df[results_df['MaskablePPO_U'] <=100])/10001
results_df[results_df['MaskablePPO_U'] >100]
print('MaskablePPO_K2_mean: ', results_df[results_df['MaskablePPO_K2'] <=100].MaskablePPO_K2.describe())
len(results_df[results_df['MaskablePPO_K2'] <=100])/10001
results_df[results_df['MaskablePPO_K2'] >100]

results_df.to_csv(general_dir_name+data_dir+file_dir+'_results.csv')


##########################################################################################
#### Greedy + MPPO
##########################################################################################

from importlib import reload
import braid_knot_env_mask_rew_0_1_limexp_u_hard
reload(braid_knot_env_mask_rew_0_1_limexp_u_hard)
from braid_knot_env_mask_rew_0_1_limexp_u_hard import BraidKnotEnv

model_mppo = MaskablePPO.load("/Unknotting/models/3s_20l/braid_knot_env_mask_rew_0_1_limexp_u_hard/66354215.zip")

s, l = 3,20
data_dir = f"/data/"
file_dir = f"pure_{s}s_{l}l"
try:
    results_df = pd.read_csv(general_dir_name+data_dir+file_dir+'').drop(['Unnamed: 0'],axis=1)
    results_df = results_df.loc[:10000]
    test_set = results_df.iloc[:, :l].values.tolist()
except:
    print('NO DATA FOUND')
env = BraidKnotEnv(braid_set = [l*[0]],braid_strands=s, e = 0, m=l)
len(test_set)

# When Greedy returns triv = False, we use a series of predictions of MPPO.
# the variable done tells if MPPO was able to solve it.

def test_greedy_plus_mppo(data=[], model= None, max_steps = l):
    info_list, index = [], 0
    for braid in data:
        # greedy_unknotting(b,0,len(b), 3)[0]
        result_braid, steps, stack,action_log, triv = greedy_unknotting(braid,0,[],[],len(braid), s)
        done = False
        if not triv:
            env.reset(np.array(braid))
            steps = 0
            while not done and steps < max_steps:
                pred_action = int(model.predict(env.state)[0])
                state, reward, _, info = env.step(pred_action)
                steps += 1
                done = reward == 1
                triv = reward == 1
            if not done:
                steps = max_steps
                triv = False
        info_list.append([index,triv,done,steps])
        index += 1
        print(index)
    return info_list

# Comparison between Greedy and Greedy + MPPO
# There is a +2.26% of unknotted braids using Greedy + MPPO.

lst_greedy = test_greedy(data = test_set)
df_greedy = pd.DataFrame(lst_greedy, columns= ['index','triv','steps'])
df_greedy[df_greedy["triv"] == True].describe()

lst = test_greedy_plus_mppo(data = test_set, model = model_mppo)
df = pd.DataFrame(lst, columns= ['index','triv','done','steps'])
df[df["triv"] == True].describe()
df[df["done"] == True].describe()
df[df["triv"] == False][df["done"] == False].describe()

