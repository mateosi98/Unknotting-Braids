import os
import sys
import pandas as pd
import numpy as np
from random import shuffle
import kitty_lacey
general_dir_name = '/Users/mateosallesize/Documents/SRO/Braids/Unknotting'
dir_name = '/Users/mateosallesize/Documents/SRO/Braids/Unknotting'
code_dir = f"/code"
sys.path.insert(0, dir_name+code_dir)



from importlib import reload
reload(kitty_lacey)


s = 3
l = 10
data_dir = f"/data_hard/"
file_dir = f"pure_{s}s_{l}l"



def greedy_unknotting(braid, steps, stack,action_log=[], length = l, strands = s, max_steps = 100):
    # print(braid, steps, stack)
    if len(stack) > 0 and braid in stack[:len(stack)-1]: # cycled back
        action_log.append('c')
        stack.append(braid)
        return False, max_steps +3 , stack, action_log, False, False
    elif kitty_lacey.is_trivial_June_2023(braid,strands): # braid is trvial
        # print('2')
        return braid, steps, stack,action_log, True, False
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
                return aux_braid, steps, stack,action_log, True, False
        if comp < min(complexities): # we have reached a dead-end
            action_log.append('h')
            return braid, max_steps + 2, stack, action_log, False, True
        best_crossing = np.argmin(complexities)
        aux_braid = braid.copy()
        aux_braid[best_crossing] *= -1
        action_log.append(best_crossing)
        return greedy_unknotting(aux_braid, steps, stack,  action_log, length, strands)
    else: # reached max steps
        return False, max_steps +1, True, action_log, False, False



answers = []
data_dir = f"/data_hard/"
file_dir = f"pure_{s}s_{l}l"
kitty_lacey.set_params(s, l)
try:
    results_df = pd.read_csv(general_dir_name+data_dir+file_dir+'').drop('Unnamed: 0',axis=1)
    results_df = results_df.loc[:10000]
    test_set = results_df.iloc[:, :l].values.tolist()
except:
    print('NO DATA FOUND')
len(test_set)

count = 0
for b in test_set:
    triv = kitty_lacey.is_trivial_June_2023(b, 3)
    if triv:
        count +=1
count

##########################################################################################
##########################################################################################
#### TESTING the increase in hard braid rate
##########################################################################################
##########################################################################################

lengths = [6,8,10,12,14,16]
# lengths = [10]
s = 3
answers = []
for l in lengths:
    trivial_set, aw_set, index = [], [], 0
    data_dir = f"/data_hard/"
    file_dir = f"pure_{s}s_{l}l"
    kitty_lacey.set_params(s, l)
    try:
        prev_set = pd.read_csv(dir_name+data_dir+file_dir).drop('Unnamed: 0',axis=1).values.tolist()
    except:
        prev_set = []
        print('NO DATA FOUND')
    print(l,'-',len(prev_set))
    for b in prev_set:
        braid, steps, stack, action_log, triv, hard = greedy_unknotting(b,0,[], [],len(b), s)
        if hard: 
            aw_set.append(braid+[index])
        index += 1
    answers.append(len(aw_set) / len(prev_set))



answers


##########################################################################################
##########################################################################################
#### Examples of HARD BRAIDS
##########################################################################################
##########################################################################################

l = 10
s = 3
answers = []
trivial_set, aw_set, index = [], [], 0
data_dir = f"/data_hard/"
file_dir = f"pure_{s}s_{l}l"
kitty_lacey.set_params(s, l)
try:
    prev_set = pd.read_csv(dir_name+data_dir+file_dir).drop('Unnamed: 0',axis=1).values.tolist()
except:
    prev_set = []
    print('NO DATA FOUND')
print(l,'-',len(prev_set))
for b in prev_set:
    braid, steps, stack, action_log, triv, hard = greedy_unknotting(b,0,[], [],len(b), s)
    if hard:
        trivial_set.append(braid+stack+ action_log+[index])
        break
    index += 1

trivial_set

len(prev_set)
len(aw_set)
index_stuck = [item for item in trivial_set if type(item[0]) == list]
n_stuck = len(index_stuck)
n_stuck
index_stuck[0]








# set[57]
# set_stuck = [set[index] for index in index_stuck]
# set_stuck[0]

# n_max = sum([1 for i in trivial_set if i== True])
# print(n_stack, n_max)
# len(set)
# trivial_set[0:5]


# tensorboard --logdir logs/2s_10l/braid_knot_env/PPO_0
# logs/2s_10l/braid_knot_env/PPO_0
# tensorboard --logdir=logs/2s_10l/braid_knot_env/PPO_0,logs/2s_10l/e1_knot_env/PPO_0






# a = [-1, -2, -2, -1, -2, -2]
# greedy_unknotting(a,0,[],len(a), 3)


# # braid, steps, stack = greedy_unknotting(set[185],0,[],len(set[185]), 3)
# def something(b):   
#     braid, steps, stack = greedy_unknotting(b,0,[],len(b), 3)
#     for i in stack:
#         com, b_com = [], []
#         print(f"KL_comp :{kitty_lacey.complexity(i,s=3)}")
#         # print(i)
#         for c in range(len(i)):
#             a = i.copy()    
#             inv = a[c] * -1
#             a[c] = inv
#             # a[c] = a[c] * -1
#             b_com.append(a)
#             com.append(kitty_lacey.complexity(a,s=3))
#         # print(b_com[np.argmin(com)])
#         print(com)
#         # print(np.argmin(com))
#         # b = i.copy()
#         # b[np.argmin(com)] *= -1
#         # print(kitty_lacey.complexity(b,s=3))

# i, b_com, com = [-2, 1, 1, 2, 1, 1, 2, 2], [], []
# for c in range(len(i)):
#     a = i.copy()    
#     inv = a[c] * -1
#     a[c] = inv
#     # a[c] = a[c] * -1
#     b_com.append(a)
#     com.append(kitty_lacey.complexity(a,s=3))

# something(aw_set[0][:len(aw_set[0])-1])

# s = 3
# l = 20

# answers = []
# data_dir = f"/data/"
# file_dir = f"pure_{s}s_{l}l"
# kitty_lacey.set_params(s, l)
# try:
#     results_df = pd.read_csv(general_dir_name+data_dir+file_dir+'_results.csv').drop('Unnamed: 0',axis=1)
#     results_df = results_df.loc[:10000]
#     test_set = results_df.iloc[:, :l].values.tolist()
# except:
#     print('NO DATA FOUND')
# len(test_set[0])

# def test_greedy(data=[], max_steps=100):
#     info_list, index = [], 0
#     for braid in data:
#         # greedy_unknotting(b,0,len(b), 3)[0]
#         braid, steps, stack,action_log, triv = greedy_unknotting(braid,0,[],[],len(braid), s)
#         info_list.append([index,triv,steps])
#         index += 1
#         print(index)
#     return info_list

# results_greedy = test_greedy(test_set,100)
# df_results_greedy = pd.DataFrame(results_greedy, columns=['INDEX','DONE', 'STEPS'])
# results_df['greedy'] = df_results_greedy.loc[:10000].STEPS.values.tolist()
# df_results_greedy[df_results_greedy.DONE==True].loc[:10000].STEPS.mean()
# len(df_results_greedy[df_results_greedy.DONE==True]) /len(df_results_greedy)
# results_df.to_csv(general_dir_name+data_dir+file_dir+'_results.csv')

# df_results_greedy[df_results_greedy.DONE==True]

# print('Mean_steps: ',df_results_greedy[df_results_greedy.DONE==True]['STEPS'].mean())

# print('STD_steps: ',df_results_greedy[df_results_greedy.DONE==True]['STEPS'].std())

# print("Accuracy: ",len(df_results_greedy[df_results_greedy.DONE==True]) / len(test_set)*100,"%")