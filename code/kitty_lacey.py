import random
import itertools
import math

# a braid is stretched from left to right
# strand positions are numbered from the bottom to the top
# a positive [negative] crossing is a clockwise [anti-clockwise] half-twist

strands_in_braid = 2
braid_len = 0 # note braid_len needs to be even for the birthday paradox code below to work correctly
braid_half_len = braid_len // 2

def set_params(s,b): 
    global strands_in_braid
    global braid_len
    global braid_half_len
    strands_in_braid = s
    braid_len = b # note braid_len needs to be even for the birthday paradox code below to work correctly
    braid_half_len = b // 2

# A braid e.g. [-1, 2, 1] is represented as a product of sigma_i,
# with only the index i being recorded in the list;
# a negative index stands for the inverse of a sigma;
# sigma_i takes strand i+1 over strand i.

# an automorphism induced by sigma_i acting on two words x and y
# (which are labels of the two affected strands, that is, strands i and i+1)
# changes them to y and yxy, respectively

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
    #this function does not return the word, but changes the word as a side effect

def apply_sigma_negative(i, words):
    (words[i], words[i+1]) = (words[i] + words[i+1] + words[i], words[i])
    #this function does not return words, but changes words as a side effect

def apply_sigma_positive(i, words):
    (words[i], words[i+1]) = (words[i+1], words[i+1] + words[i] + words[i+1])
    #this function does not return words, but changes words as a side effect

def braid_to_automorphisms_slow(braid, words):
    for c in braid:
        if c > 0:
            apply_sigma_positive(c-1, words)
        elif c < 0:
            apply_sigma_negative(-c-1, words)
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
    # to choose between different implementations of this function
    braid_to_automorphisms_fast(braid, words)

def inverse_braid(braid):
    return [-a for a in braid[::-1]]

def braids_are_equal(b1, b2):
    w1 = [[i] for i in range(1, strands_in_braid+1)]
    w2 = [[i] for i in range(1, strands_in_braid+1)]
    braid_to_automorphisms(b1, w1)
    braid_to_automorphisms(b2, w2)
    return w1 == w2

def generate_classes_of_equal_braids():
    n = strands_in_braid - 1
    list_of_possible_crossings = list(range(-n, 0)) + list(range(1, n+1))
    all_half_braids = list(itertools.product(list_of_possible_crossings, repeat=braid_half_len))
    list_of_classes_of_equal_braids = []
    while len(all_half_braids) > 0:
        braid_to_consider = all_half_braids.pop()
        # check if the braid is in one of the classes
        class_found = False
        for i in range(len(list_of_classes_of_equal_braids)):
            if braids_are_equal(braid_to_consider, list_of_classes_of_equal_braids[i][0]):
                class_found = True
                list_of_classes_of_equal_braids[i].append(braid_to_consider)
                break
        if not class_found:
            list_of_classes_of_equal_braids.append([braid_to_consider])
    return list_of_classes_of_equal_braids

def generate_classes_of_equal_braids_with_zeros():
    n = strands_in_braid - 1
    list_of_possible_crossings = list(range(-n, 0)) + [0] + list(range(1, n+1))
    all_half_braids = list(itertools.product(list_of_possible_crossings, repeat=braid_half_len))
    list_of_classes_of_equal_braids = []
    while len(all_half_braids) > 0:
        braid_to_consider = all_half_braids.pop()
        # check if the braid is in one of the classes
        class_found = False
        for i in range(len(list_of_classes_of_equal_braids)):
            if braids_are_equal(braid_to_consider, list_of_classes_of_equal_braids[i][0]):
                class_found = True
                list_of_classes_of_equal_braids[i].append(braid_to_consider)
                break
        if not class_found:
            list_of_classes_of_equal_braids.append([braid_to_consider])
    return list_of_classes_of_equal_braids

'''
# testing the 'fast' function
n = strands_in_braid - 1
list_of_possible_crossings = list(range(-n, 0)) + list(range(1, n+1))
for braid in list(product(list_of_possible_crossings, repeat=5)):
    ws = [[i] for i in range(1, strands_in_braid+1)]
    braid_to_automorphisms_slow(braid, ws)
    wf = [[i] for i in range(1, strands_in_braid+1)]
    braid_to_automorphisms_fast(braid, wf)
    if not ws == wf:
        print(ws, wf)
'''

def generate_all_trivial_braids(my_strands_in_braid, my_braid_len):
    global strands_in_braid, braid_len, braid_half_len
    strands_in_braid = my_strands_in_braid
    braid_len = my_braid_len
    braid_half_len = braid_len // 2

    list_of_classes_of_equal_braids = generate_classes_of_equal_braids()
    trivial_braids = []
    for c in list_of_classes_of_equal_braids:
        for b1 in c:
            for b2 in c:
                trivial_braids.append(list(b1) + inverse_braid(b2))
    return trivial_braids

#trivial_braids = generate_all_trivial_braids()
#print(trivial_braids)

################################################################################################################################################################

def is_trivial_June_2023(braid, s):
    set_params(s, len(braid))
    braid1 = braid[:braid_half_len]
    braid2 = inverse_braid(braid[braid_half_len:])
    wf1 = [[i] for i in range(1, strands_in_braid+1)]
    braid_to_automorphisms_fast(braid1, wf1)
    wf2 = [[i] for i in range(1, strands_in_braid+1)]
    braid_to_automorphisms_fast(braid2, wf2)
    return wf1 == wf2

def is_trivial(braid, s):
    set_params(s, len(braid))
    wf = [[i] for i in range(1, strands_in_braid+1)]
    braid_no_intersections = [[i] for i in range(1, strands_in_braid+1)]
    braid_to_automorphisms_fast(braid, wf)
    return wf == braid_no_intersections

def complexity(braid, s):
    set_params(s, len(braid))
    wf = [[i] for i in range(1, strands_in_braid+1)]
    braid_to_automorphisms_fast(braid, wf)
    return max([len(i) for i in wf])

################################################################################################################################################################

def create_balanced_dataset(number_of_braids_of_each_kind):
    # finish this function to create a dataset for AL
    n = strands_in_braid - 1
    list_of_possible_crossings = list(range(-n, 0)) + list(range(1, n+1))
    trivial_braids = []
    non_trivial_braids = []
    while len(trivial_braids) < number_of_braids_of_each_kind or len(non_trivial_braids) < number_of_braids_of_each_kind:
        # generate a random braid
        braid = [random.choice(list_of_possible_crossings) for _ in range(braid_len)]
        # try to add the new braid to the correctly chosen list of braids
        if is_trivial(braid):
            if not braid in trivial_braids and len(trivial_braids) < number_of_braids_of_each_kind:
                trivial_braids.append(braid)
        else:
            if not braid in non_trivial_braids and len(non_trivial_braids) < number_of_braids_of_each_kind:
                non_trivial_braids.append(braid)
            
    with open("trivial-and-not-braids-12-3.txt", "w") as output_file:
        for braid in trivial_braids:
            print(braid, 1, file=output_file)
        for braid in non_trivial_braids:
            print(braid, 0, file=output_file)

# create_balanced_dataset(1000)

'''
# code counting trivial braids
for braid_half_len in range(1, 10):
    #list_of_classes_of_equal_braids = generate_classes_of_equal_braids()
    list_of_classes_of_equal_braids = generate_classes_of_equal_braids_with_zeros()
    print(2 * braid_half_len, sum([len(c)*len(c) for c in list_of_classes_of_equal_braids]))
'''

'''
# this fragment produces all braids of a given size and states, for each one, if it is trivial
n = strands_in_braid - 1
list_of_possible_crossings = list(range(-n, 0)) + list(range(1, n+1))
for braid in list(product(list_of_possible_crossings, repeat=6)):
    wf = [[i] for i in range(1, strands_in_braid+1)]
    braid_no_intersections = [[i] for i in range(1, strands_in_braid+1)]
    braid_to_automorphisms_fast(braid, wf)
    
    if wf == braid_no_intersections: 
        print(braid, wf, braid_no_intersections)
    
    #print(braid, wf == braid_no_intersections)
'''

'''
# this is how the fast function for checking if trivial is used
braid = [1, 2, 1, -2, -1, 2, -1, 1, -2, -2]
print(is_trivial(braid))
'''



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

def is_cep_ces_aut(braid, strands = strands_in_braid):
    set_params(strands, len(braid))
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