import csv
import itertools
import os
from random import shuffle
from random import choice
#import kitty_lacey_for_calling

def braid_to_e1(braid, number_of_rows):
  e1 = []
  for row_number in range(1, number_of_rows+1):
    row = []
    for crossing in braid:
      if abs(crossing) == row_number:
        if crossing > 0:
          row.append(1)
        else:
          row.append(1)
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
              new_braid[i+1][j] = 1
          if braid[i][j] == -1:
              new_braid[i][j] = 1
              new_braid[i+1][j] = 1
  # now braid is the standard encoding e1, and new_braid is e2
  return new_braid

def braid_to_e2(braid, n):
  return e1_to_e2(braid_to_e1(braid, n))

def e1_to_e3(braid):
  new_braid = [ [0 for _ in range(len(braid[0]))] for _ in range(len(braid)+1)]
  for i in range(len(braid)):
      for j in range(len(braid[0])):
          if braid[i][j] == 1:
              new_braid[i][j] = 1
              new_braid[i+1][j] = 1
          if braid[i][j] == -1:
              new_braid[i][j] = 1
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

def is_identity(braid):
  permutation = list(range(strands_in_braid))
  for b in braid:
    b = abs(b)
    permutation[b-1], permutation[b] = permutation[b], permutation[b-1]
  #print(braid, permutation)
  return permutation == list(range(strands_in_braid))


def braid_to_e3(braid, n):
  return e1_to_e3(braid_to_e1(braid, n))

def alternating_sum(l):
  return sum(l[::2]) - sum(l[1::2])

def flatten(matrix, label):
    to_print = matrix + [[label]]
    return ', '.join([str(a) for a in list(itertools.chain(*to_print))])

number_of_braids_of_each_kind = 1000
strands_in_braid = 3
braid_len = 12
n = strands_in_braid - 1
trivial_braids = []
non_trivial_braids = []

while len(trivial_braids) < number_of_braids_of_each_kind or len(non_trivial_braids) < number_of_braids_of_each_kind:
  list_of_possible_crossings = list(range(1, n+1))
  braid = [choice(list_of_possible_crossings) for _ in range(braid_len)]
  if is_identity(braid):
    if not braid in trivial_braids and len(trivial_braids) < number_of_braids_of_each_kind:
      trivial_braids.append(braid)
  else:
    if not braid in non_trivial_braids and len(non_trivial_braids) < number_of_braids_of_each_kind:
      non_trivial_braids.append(braid)
#print('trivial', trivial_braids)
#print('non-trivial', non_trivial_braids)
output_file_prefix = "C:\\Box\\Braids\\trivial-and-not-braids\\permutations\\p-12-3-2000"
# create file with e1
output_file = output_file_prefix + '-e1.txt'
with open(output_file, "w") as output_file:
  for braid in trivial_braids:
    label = 1
    e1 = braid_to_e1(braid, n)
    print(flatten(e1, label), file=output_file)
  for braid in non_trivial_braids:
    label = 0
    e1 = braid_to_e1(braid, n)
    print(flatten(e1, label), file=output_file)
# create file with e2
output_file = output_file_prefix + '-e2.txt'
with open(output_file, "w") as output_file:
  for braid in trivial_braids:
    label = 1
    e2 = braid_to_e2(braid, n)
    print(flatten(e2, label), file=output_file)
  for braid in non_trivial_braids:
    label = 0
    e2 = braid_to_e2(braid, n)
    print(flatten(e2, label), file=output_file)
# create file with e3
output_file = output_file_prefix + '-e3.txt'
with open(output_file, "w") as output_file:
  for braid in trivial_braids:
    label = 1
    e3 = braid_to_e3(braid, n)
    print(flatten(e3, label), file=output_file)
  for braid in non_trivial_braids:
    label = 0
    e3 = braid_to_e3(braid, n)
    print(flatten(e3, label), file=output_file)