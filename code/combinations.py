import itertools
import sys
import pandas as pd
from importlib import reload
import kitty_lacey
reload(kitty_lacey)

general_dir_name = ''
dir_name = ''
code_dir = f"/code"
sys.path.insert(0, dir_name+code_dir)

s = 3
l = 10
kitty_lacey.set_params(s, l)

def combination_search(braid, length, strands, depth, action_log):
    if depth >= length:
        return False, depth, action_log
    else:
        if kitty_lacey.is_trivial_June_2023(braid, strands):
            return True, depth, action_log
        else:
            depth += 1
            combs = itertools.combinations(range(length), depth)
            for comb in combs:
                aux_braid = braid.copy()
                for cros in comb:
                    aux_braid[cros] *= -1
                if kitty_lacey.is_trivial_June_2023(aux_braid, strands):
                    action_log.append(comb)
                    return True, depth, action_log
            return combination_search(braid, length, strands, depth, action_log)

b = [1, -1, -1, -1, 2, 1, 1, 2, 1, 1]
combination_search(b, l, s, 0,[])

answers = []
data_dir = f"/data_rl/"
file_dir = f"pure_{s}s_{l}l"
kitty_lacey.set_params(s, l)
try:
    results_df = pd.read_csv(general_dir_name+data_dir+file_dir).drop('Unnamed: 0',axis=1)
    # results_df = results_df.loc[:1000]
    test_set = results_df.iloc[:, :l].values.tolist()
except:
    print('NO DATA FOUND')
results_df
general_dir_name+data_dir+file_dir+'_results.csv'
len(test_set)

# test_set = [test_set[6448], test_set[7964], test_set[14588]]

def test_combinations(data=[]):
    info_list, index, depth = [], 0, 0
    for braid in data:
        o_braid = braid.copy()
        triv, n_steps, steps = combination_search(braid, l, s, depth,[])
        info_list.append([index,o_braid, 'UN', triv,n_steps, steps])
        index += 1
        print(index)
    return info_list


results_combinations = test_combinations(test_set)

# df_results_detail = pd.read_csv(general_dir_name+data_dir+file_dir+'_results_detailed.csv').drop('Unnamed: 0',axis=1)
df_results_combinations = pd.DataFrame(results_combinations, columns=['INDEX','BRAID','MODEL','DONE', 'N_STEPS', 'STEPS'])
results_detail = pd.DataFrame(results_combinations, columns=['INDEX','BRAID','MODEL','DONE', 'N_STEPS', 'STEPS']).loc[:9]
# results_detail = pd.concat([df_results_detail,df_results_combinations.loc[:9]], ignore_index=True)
results_df['unknoting_number'] = df_results_combinations.N_STEPS.values.tolist()

# results_detail.to_csv(general_dir_name+data_dir+file_dir+'_results_detailed.csv')
results_df.to_csv(general_dir_name+data_dir+file_dir+'_u.csv')

df_results_combinations[df_results_combinations.DONE==False]

df_results_combinations[df_results_combinations.DONE==True]

df_results_combinations[df_results_combinations.STEPS>2]

print('Mean_steps: ',df_results_combinations[df_results_combinations.DONE==True]['STEPS'].mean())

print('STD_steps: ',df_results_combinations[df_results_combinations.DONE==True]['STEPS'].std())

len(df_results_combinations[df_results_combinations.DONE==True]) / len(test_set)

df_results_combinations['BRAID'] = test_set
df_results_combinations = df_results_combinations.drop(['INDEX', 'DONE'], axis = 1)

df_results_combinations[df_results_combinations.STEPS==5]

df_results_combinations.to_csv('exhaustive_unknoting_3s_10l.csv', index=False)



answers = []
data_dir = f"/data/"
file_dir = f"pure_{s}s_{l}l"
kitty_lacey.set_params(s, l)
try:
    results_df = pd.read_csv(general_dir_name+data_dir+file_dir+'_results.csv').drop('Unnamed: 0',axis=1)
    results_df = results_df.loc[:9]
    test_set = results_df.iloc[:, :l].values.tolist()
except:
    print('NO DATA FOUND')
results_df
general_dir_name+data_dir+file_dir+'_results.csv'
len(test_set)



def test_combinations(data=[]):
    info_list, index, depth = [], 0, 0
    for braid in data:
        triv, n_steps, steps = combination_search(braid, l, s, depth,[])
        info_list.append([index,triv,n_steps, steps])
        index += 1
    return info_list


results_combinations_detailed = test_combinations(test_set)

df_results_combinations = pd.DataFrame(results_combinations, columns=['INDEX','DONE', 'STEPS'])

results_df.to_csv(general_dir_name+data_dir+file_dir+'_results.csv')