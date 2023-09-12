
import pandas as pd
import random
import itertools
import sys

dir_name = '/Users/mateosallesize/Documents/SRO/Braids/Unknotting'
sys.path.insert(0, dir_name)

dir_name = "/Users/mateosallesize/Documents/SRO/Braids/Unknotting/"

def set_params(s = 3,l = 10): 
    global strands_in_braid
    global braid_len
    global braid_half_len
    strands_in_braid = s
    braid_len = l # note braid_len needs to be even for the birthday paradox code below to work correctly
    braid_half_len = l // 2

def reduce_word_in_kei_group(w):
  reduced = True
  while(reduced):
      reduced = False
      for i in range(len(w)-1):
          if w[i] == w[i+1]:
              w.pop(i)
              w.pop(i)
              reduced = True
              break

def apply_sigma_negative(i, words):
    (words[i], words[i+1]) = (words[i] + words[i+1] + words[i], words[i])

def apply_sigma_positive(i, words):
    (words[i], words[i+1]) = (words[i+1], words[i+1] + words[i] + words[i+1])

def braid_to_automorphisms_slow(braid, words):
    for s in braid:
        if s > 0:
            apply_sigma_positive(s-1, words)
        elif s < 0:
            apply_sigma_negative(-s-1, words)
    for i in range(len(words)):
        reduce_word_in_kei_group(words[i])

def braid_to_automorphisms_fast(braid, words):
    if len(braid) <= 1:
        braid_to_automorphisms_slow(braid, words)
    else:
        b1 = braid[:(len(braid)//2)]
        b2 = braid[(len(braid)//2):]
        w1 = [[i] for i in range(1, strands_in_braid+1)]
        #w1 = words
        w2 = [[i] for i in range(1, strands_in_braid+1)]
        braid_to_automorphisms_fast(b1, w1)
        braid_to_automorphisms_fast(b2, w2)
        for i in range(len(words)):
            words[i] = []
            for a in w2[i]:
                #print(w1, w2) 
                words[i].extend(w1[a-1])
    for i in range(len(words)):
        reduce_word_in_kei_group(words[i])

def braid_to_automorphisms(braid, words):
  for s in braid:
      if s > 0:
          apply_sigma_positive(s-1, words)
      elif s < 0:
          apply_sigma_negative(-s-1, words)
  for i in range(len(words)):
      reduce_word_in_kei_group(words[i])

def inverse_braid(braid):
    return [-a for a in braid[::-1]]

def is_trivial(braid, s=3):
    set_params(s, len(braid))
    braid1 = braid[:braid_half_len]
    braid2 = inverse_braid(braid[braid_half_len:])
    wf1 = [[i] for i in range(1, s+1)]
    braid_to_automorphisms_fast(braid1, wf1)
    wf2 = [[i] for i in range(1, s+1)]
    braid_to_automorphisms_fast(braid2, wf2)
    return wf1 == wf2

def is_braid_trivial(braid, braid_strands):
  words = [[i] for i in range(1, braid_strands+1)]
  braid_to_automorphisms(braid, words)
  is_trivial = True
  for i in range(braid_strands):
      if words[i] == [i+1]:
          continue
      is_trivial = False
  return is_trivial

def convert_set_to_list(set):
  list_of_tuples = sorted(set)
  list_of_lists = []
  for item in list_of_tuples:
    list_of_lists.append(list(item))
  return list_of_lists

def is_identity(braid, braid_strands):
  permutation = list(range(braid_strands))
  for b in braid:
    b = abs(b)
    permutation[b-1], permutation[b] = permutation[b], permutation[b-1]
  #print(braid, permutation)
  return permutation == list(range(braid_strands))

def create_braids(braid_strands, braid_len, size):
  set_of_braids = set()
  with open("trivial_braid_examples.txt", "w") as output_file:
    n = braid_strands - 1
    letters = list(range(-n, 0)) + list(range(1, n+1))
    how_many_braids_generated = 0
    while(how_many_braids_generated < size):
        braid = []
        for i in range(braid_len):
            braid.append(random.choice(letters))
        if tuple(braid) in set_of_braids: continue
        if braid_strands == 3:
          if (not is_cep_ces_aut(braid, braid_strands)) and is_identity(braid, braid_strands):
              how_many_braids_generated += 1
              print(how_many_braids_generated)
              print(braid, file=output_file)
              set_of_braids.add(tuple(braid))
        else:
          if (not is_trivial(braid, braid_strands)) and is_identity(braid, braid_strands):
              how_many_braids_generated += 1
              print(how_many_braids_generated)
              print(braid, file=output_file)
              set_of_braids.add(tuple(braid))
  return convert_set_to_list(set_of_braids)

def produce(size = 100, strands = 3, length = 10):
  set_params(s=strands,l=length)
  braids = create_braids(strands_in_braid, braid_len, size)
  file_name = "pure_"+str(strands_in_braid)+"s_"+str(braid_len)+"l"
  try:
    df_b = pd.read_csv(dir_name+file_name).drop('Unnamed: 0',axis=1)
    past_braids = df_b.values.tolist()
    print('Loaded ' + str(len(past_braids)))
  except:
      past_braids = []
  for i in past_braids:
      braids.append(i)
  braids.sort()
  braids = list(braids for braids,_ in itertools.groupby(braids))
  df_a = pd.DataFrame(braids)
  df_a.to_csv(dir_name+file_name)
  print('Created '+ file_name + ' with '+ str(df_a.shape[0]))


def braid_to_e1(braid, number_of_rows):
  e1 = []
  for row_number in range(1, number_of_rows+1):
    row = []
    for crossing in braid:
      if abs(crossing) == row_number:
        if crossing > 0:
          row.append(1)
        else:
          row.append(-1)
      else:
        row.append(0)
    e1.append(row)
  return e1

def e1_to_e2(braid):
  new_braid = [ [0 for _ in range(len(braid[0]))] for _ in range(len(braid)+1)]
  for i in range(len(braid)):
      for j in range(len(braid[0])):
          if braid[i][j] == 1:
              new_braid[i][j] = 1
              new_braid[i+1][j] = -1
          if braid[i][j] == -1:
              new_braid[i][j] = -1
              new_braid[i+1][j] = 1
  # now braid is the standard encoding e1, and new_braid is e2
  return new_braid

def modify_e3(e3):
  for i in range(len(e3[0])):
    # in column i, find 0
    j = [e3[0][i], e3[1][i], e3[2][i]].index(0)
    j = (j-1+3) % 3
    e3[j][i] = 0
  return e3

def braid_to_e2(braid, n):
  return e1_to_e2(braid_to_e1(braid, n))

def e1_to_e3(braid):
  new_braid = [ [0 for _ in range(len(braid[0]))] for _ in range(len(braid)+1)]
  for i in range(len(braid)):
      for j in range(len(braid[0])):
          if braid[i][j] == 1:
              new_braid[i][j] = 1
              new_braid[i+1][j] = -1
          if braid[i][j] == -1:
              new_braid[i][j] = -1
              new_braid[i+1][j] = 1
  # now braid is the standard encoding e1, and new_braid is e2
  permutation = list(range(len(new_braid)))
  for j in range(len(braid[0])):
      # act with the permutation on the i-th column in new_braid
      column = [new_braid[i][j] for i in range(len(new_braid))]
      #column = [column[permutation[i]] for i in range(len(new_braid))]
      column = [column[permutation.index(i)] for i in range(len(new_braid))]
      for i in range(len(new_braid)):
          new_braid[i][j] = column[i]
      # update the permutation using braid
      column = [braid[i][j] for i in range(len(braid))]
      if 1 in column:
          i = column.index(1)
      else:
          i = column.index(-1)
      permutation[i], permutation[i+1] = permutation[i+1], permutation[i]
  #print(flatten(new_braid, label), file=output_file)
  return new_braid

def braid_to_e3(braid, n):
  return e1_to_e3(braid_to_e1(braid, n))

def alternating_sum(l):
  return sum(l[::2]) - sum(l[1::2])

def flatten(matrix, label):
    to_print = matrix + [[label]]
    return ', '.join([str(a) for a in list(itertools.chain(*to_print))])

def is_cep_ces_aut(braid, strands):
    e1 = braid_to_e1(braid, strands_in_braid)
    s = sum(sum(row) for row in e1)
    if s != 0:
        return True
    e3 = e1_to_e3(e1)
    alt_sums = [alternating_sum(strand) for strand in e3]
    if not all([s == 0 for s in alt_sums]): 
        return True
    if is_trivial(braid):
        return False
    else:
        return True


############################################################################################################################################

############################################################################################################################################

############################################################################################################################################

del set
produce(10000, 3, 14)

