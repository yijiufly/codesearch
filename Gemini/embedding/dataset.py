import glob
import random
from collections import defaultdict

import tensorflow as tf
import numpy as np
import pickle as p
from numpy.random import choice, permutation
from itertools import combinations
import util
import os
import sys
import pdb
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir) + "/coogleconfig")

from random import shuffle

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from raw_graphs import *
from gemini_feature_extraction_ST import disassemble
print(raw_graphs)
flags = tf.app.flags
FLAGS = flags.FLAGS


# Added to test standalone testing of this filef
# To be commented if testing with NN Code
# flags.DEFINE_string('data_file', 'train.pickle', "Stores the train sample after preprocessing")
# flags.DEFINE_string('test_file', 'test.pickle', "Stores the test sample after preprocessing")

class BatchGenerator():
    def __init__(self, filter_size=0):
        np.random.seed()
        self.filter_size = filter_size
        self.sample = defaultdict(list)
        self.train_sample, self.test_sample = self.create_sample_space()
        self.negative_sample = None
    # for testing
    # need to commented out while training
    # g, g1, g2 = self.get_train_acfg()
    # print("g: ", g)
    # print("g1: ", g1)
    # print("g2: ", g2)

    # create sample space from all the available '.ida' files
    def create_sample_space(self):
        # get all binary file list

        for file_path in glob.iglob(r"/home/yijiufly/Downloads/codesearch/Gemini/trainingdataset/*.ida"):
            if os.path.exists(file_path):
                print(file_path)
                acfgs = p.load(open(file_path, 'rb'), encoding='latin1')
            else:
                # disassemble file
                print(file_path)
                acfgs = disassemble(file_path)
                p.dump(acfgs, open(file_path + '.ida', 'wb'), protocol=2)

            filter_cnt = 0

            for acfg in acfgs.raw_graph_list:
                if len(acfg.fv_list) < self.filter_size:
                    filter_cnt += 1
                    continue
                fvec_list = []
                func_name = acfg.funcname

                # This loop is to delete first two elements of feature vectors
                # because they are list and we need numeric valure for our matrix
                # if there is method to convert those list to values this loop can be commented out
                for fv in acfg.fv_list:
                    # deleting first 2 element of each feature vector
                    # del fv[:2]
                    fvec_list.append(fv)

                # converting to matrix form
                acfg_mat = np.array(fvec_list)

                # setting up neighbor matrix from edge list
                num_nodes = len(fvec_list)
                acfg_nbr = np.zeros((num_nodes, num_nodes))

                for edge in acfg.edge_list:
                    acfg_nbr.itemset((edge[0], edge[1]), 1)
                    acfg_nbr.itemset((edge[1], edge[0]), 1)

                self.sample[func_name].append((func_name, acfg_mat, acfg_nbr))
            print(filter_cnt, len(acfgs.raw_graph_list))

        # TODO: filter out potential testing functions
        # divide the training and testing data
        train_sample, test_sample = self.divide_sample_space()

        return train_sample, test_sample

    # get train acfg
    def get_train_acfg(self):
        return self.get_acfg_pairs(self.train_sample)

    # get test acfg
    def get_test_acfg(self):
        return self.get_acfg_pairs(self.test_sample)

    # get randomly selected acgf pair sampled from sample list
    def get_acfg_pairs(self, sample):
        if self.negative_sample is None:
            k1, k2 = np.random.choice(list(sample.keys()), 2, False)
            idx1, idx2 = np.random.choice(len(sample[k1]), 2)
            g, g1 = sample[k1][idx1], sample[k1][idx2]
            g2 = random.choice(sample[k2])
            return g, g1, g2, g
        else:
            k = np.random.choice(list(self.negative_sample.keys()))
            idx1, idx2 = np.random.choice(len(sample[k]), 2)
            g, g1 = sample[k][idx1], sample[k][idx2]
            k3 = np.random.choice(list(self.negative_sample[k]))
            g2 = random.choice(sample[k3])
            return g, g1, g2, g

    # Divide sample space into training and testing sample
    def divide_sample_space(self):
        sample_size = sum([len(v) for v in self.sample.values()])
        train_size = int(sample_size * .5)
        keys = list(self.sample.keys())
        shuffle(keys)
        train_sample = defaultdict(list)
        test_sample = defaultdict(list)
        it = iter(keys)
        total_len = 0
        while True:
            k = next(it)
            train_sample[k] = self.sample[k]
            total_len += len(self.sample[k])
            if total_len >= train_size:
                break
        for k in it:
            test_sample[k] = self.sample[k]

        return train_sample, test_sample


if __name__ == '__main__':
    sample_gen = BatchGenerator()
