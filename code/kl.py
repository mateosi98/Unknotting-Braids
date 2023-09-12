import random
from itertools import product

class kitty_lacey:
    def __init__(self,s,l):
        self.strands_in_braid = s
        self.braid_len = l
        self.braid_half_len = self.braid_len // 2

    def reduce_word_in_kei_group(self,w):
        reduced = True
        while(reduced):
            reduced = False
            for i in range(len(w)-1):
                if w[i] == w[i+1]:
                    w.pop(i)
                    w.pop(i)
                    reduced = True
                    break

    def apply_sigma_negative(self,i, words):
        (words[i], words[i+1]) = (words[i] + words[i+1] + words[i], words[i])

    def apply_sigma_positive(self,i, words):
        (words[i], words[i+1]) = (words[i+1], words[i+1] + words[i] + words[i+1])

    def braid_to_automorphisms_slow(self,braid, words):
        for s in braid:
            if s > 0:
                self.apply_sigma_positive(s-1, words)
            elif s < 0:
                self.apply_sigma_negative(-s-1, words)
        for i in range(len(words)):
            self.reduce_word_in_kei_group(words[i])

    def braid_to_automorphisms_fast(self,braid, words):
        if len(braid) <= 1:
            self.braid_to_automorphisms_slow(braid, words)
        else:
            b1 = braid[:(len(braid)//2)]
            b2 = braid[(len(braid)//2):]
            w1 = [[i] for i in range(1, self.strands_in_braid+1)]
            w2 = [[i] for i in range(1, self.strands_in_braid+1)]
            self.braid_to_automorphisms_fast(b1, w1)
            self.braid_to_automorphisms_fast(b2, w2)
            for i in range(len(words)):
                words[i] = []
                for a in w2[i]:
                    words[i].extend(w1[a-1])
        for i in range(len(words)):
            self.reduce_word_in_kei_group(words[i])
        
    def braid_to_automorphisms(self,braid, words):
        self.braid_to_automorphisms_fast(braid, words)

    def inverse_braid(self,braid):
        return [-a for a in braid[::-1]]

    def braids_are_equal(self,b1, b2):
        w1 = [[i] for i in range(1, self.strands_in_braid+1)]
        w2 = [[i] for i in range(1, self.strands_in_braid+1)]
        self.braid_to_automorphisms(b1, w1)
        self.braid_to_automorphisms(b2, w2)
        return w1 == w2

    def generate_classes_of_equal_braids(self):
        n = self.strands_in_braid - 1
        list_of_possible_crossings = list(range(-n, 0)) + list(range(1, n+1))
        all_half_braids = list(product(list_of_possible_crossings, repeat=self.braid_half_len))
        list_of_classes_of_equal_braids = []
        while len(all_half_braids) > 0:
            braid_to_consider = all_half_braids.pop()
            class_found = False
            for i in range(len(list_of_classes_of_equal_braids)):
                if self.braids_are_equal(braid_to_consider, list_of_classes_of_equal_braids[i][0]):
                    class_found = True
                    list_of_classes_of_equal_braids[i].append(braid_to_consider)
                    break
            if not class_found:
                list_of_classes_of_equal_braids.append([braid_to_consider])
        return list_of_classes_of_equal_braids

    def generate_classes_of_equal_braids_with_zeros(self):
        n = self.strands_in_braid - 1
        list_of_possible_crossings = list(range(-n, 0)) + [0] + list(range(1, n+1))
        all_half_braids = list(product(list_of_possible_crossings, repeat=self.braid_half_len))
        list_of_classes_of_equal_braids = []
        while len(all_half_braids) > 0:
            braid_to_consider = all_half_braids.pop()
            class_found = False
            for i in range(len(list_of_classes_of_equal_braids)):
                if self.braids_are_equal(braid_to_consider, list_of_classes_of_equal_braids[i][0]):
                    class_found = True
                    list_of_classes_of_equal_braids[i].append(braid_to_consider)
                    break
            if not class_found:
                list_of_classes_of_equal_braids.append([braid_to_consider])
        return list_of_classes_of_equal_braids

    def generate_all_trivial_braids(self):#,my_strands_in_braid, my_braid_len):

        list_of_classes_of_equal_braids = self.generate_classes_of_equal_braids()
        trivial_braids = []
        for c in list_of_classes_of_equal_braids:
            for b1 in c:
                for b2 in c:
                    trivial_braids.append(list(b1) + self.inverse_braid(b2))
        return trivial_braids

    def is_trivial_23(self,braid, s=3):
        # set_params(s, len(braid))
        braid1 = braid[:self.braid_half_len]
        braid2 = self.inverse_braid(braid[self.braid_half_len:])
        wf1 = [[i] for i in range(1, s+1)]
        self.braid_to_automorphisms_fast(braid1, wf1)
        wf2 = [[i] for i in range(1, s+1)]
        self.braid_to_automorphisms_fast(braid2, wf2)
        return wf1 == wf2


    def is_trivial(self,braid):
        wf = [[i] for i in range(1, self.strands_in_braid+1)]
        braid_no_intersections = [[i] for i in range(1, self.strands_in_braid+1)]
        self.braid_to_automorphisms_fast(braid, wf)
        return wf == braid_no_intersections
    
    def autowrithe(self,braid):
        wf = [[i] for i in range(1, self.strands_in_braid+1)]
        # braid_no_intersections = [[i] for i in range(1, self.strands_in_braid+1)]
        self.braid_to_automorphisms_fast(braid, wf)
        return max([len(item) for item in wf])

    def create_balanced_dataset(self,number_of_braids_of_each_kind):
        n = self.strands_in_braid - 1
        list_of_possible_crossings = list(range(-n, 0)) + list(range(1, n+1))
        trivial_braids = []
        non_trivial_braids = []
        while len(trivial_braids) < number_of_braids_of_each_kind or len(non_trivial_braids) < number_of_braids_of_each_kind:
            braid = [random.choice(list_of_possible_crossings) for _ in range(self.braid_len)]
            if self.is_trivial(braid):
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
