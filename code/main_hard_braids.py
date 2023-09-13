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






