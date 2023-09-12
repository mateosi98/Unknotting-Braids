import itertools
from random import shuffle
from random import choice
from importlib import reload
import kl
reload(kl)

class kitty_lacey_new:
    def __init__(self, s, l):
        self.kl_instance = kl.kitty_lacey(s,l)
        self.strands_in_braid = s
        self.braid_len = l

    def braid_to_e1(self,braid,number_of_rows):
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

    def e1_to_e2(self,braid):
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

    def braid_to_e2(self,braid,n):
        return self.e1_to_e2(self.braid_to_e1(braid, n))

    def e1_to_e3(self,braid):
        new_braid = [[0 for _ in range(len(braid[0]))] for _ in range(len(braid)+1)]
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

    def braid_to_e3(self,braid, n):
        return self.e1_to_e3(self.braid_to_e1(braid, n))

    def alternating_sum(self,l):
        return sum(l[::2]) - sum(l[1::2])

    def flatten(self, matrix, label):
        to_print = matrix + [[label]]
        return ', '.join([str(a) for a in list(itertools.chain(*to_print))])

    def is_trivial_new(self,braid):
        e1 = self.braid_to_e1(braid, 3)
        s = sum(sum(row) for row in e1)
        if self.kl_instance.is_trivial(braid):
            print('3')
        if s != 0:
            print('1')
            return False
        e3 = self.e1_to_e3(e1)
        alt_sums = [self.alternating_sum(strand) for strand in e3]
        if not all([s == 0 for s in alt_sums]): 
            print('2')
            print(e3)
            print(alt_sums)
            return False
        if self.kl_instance.is_trivial(braid):
            print('3')
            return True
        else:
            print('4')
            return False


# a = kitty_lacey_new(3,10)
# a.is_trivial_new([-2,  2, -2,  2,  1, -1,  2,  1, -1, -2])