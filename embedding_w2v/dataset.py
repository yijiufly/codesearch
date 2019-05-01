import glob
import random
from collections import defaultdict
import tensorflow as tf
import numpy as np
import pickle as p
from numpy.random import choice, permutation
from itertools import combinations
from extractor import disassemble, inst_emb_gen
from gensim.models import Word2Vec
import util
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir) + "/coogleconfig")

from random import shuffle

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from raw_graphs import *

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
        try:  # check if model file exists
            if os.path.exists("model.bin"):
                print("Loading model,..")
            else:
                os.stat("model.bin")
        except:  # create dict if not exists
            print("No models found...")
            print("Generating models...")
            self.dict_sample = self.create_dict_word2vec()
            print(self.dict_sample)
            self.model = Word2Vec(self.dict_sample, window=2, min_count=1, workers=8)
            # words = list(self.model.wv.vocab)
            # print(self.model.wv[" "])
            # print(words)
            self.model.wv.save_word2vec_format("model.bin")

        self.train_sample, self.test_sample = self.create_sample_space()

    # create dictionary by using all data
    def create_dict_word2vec(self):
        dicts = []
        for path in glob.iglob(r"/home/yijiufly/Downloads/codesearch/embedding_w2v/test/out_analysis/x64/*.so"):
            print(path)
            dict_word = inst_emb_gen(path)
            dicts.extend(dict_word)
        return dicts

    # create sample space from all the available '.ida' files
    def create_sample_space(self):
        for file_path in glob.iglob(r"/home/yijiufly/Downloads/codesearch/embedding_w2v/test/out_analysis/x64/*.so"):
            # disassemble file
            acfgs = disassemble(file_path)

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
                    del fv[:2]
                    fvec_list.append(fv)

                # converting to matrix form
                acfg_mat = np.array(fvec_list)

                # setting up neighbor matrix from edge list
                num_nodes = len(fvec_list)
                acfg_nbr = np.zeros((num_nodes, num_nodes))

                # undirected graph
                for edge in acfg.edge_list:
                    acfg_nbr.itemset((edge[0], edge[1]), 1)
                    # acfg_nbr.itemset((edge[1], edge[0]), 1)
                    # comment out to get bi-directional info

                self.sample[func_name].append((func_name, acfg_mat, acfg_nbr))

                # print(filter_cnt, len(acfgs.raw_graph_list))
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
        k1, k2 = np.random.choice(list(sample.keys()), 2, False)
        idx1, idx2 = np.random.choice(len(sample[k1]), 2)
        g, g1 = sample[k1][idx1], sample[k1][idx2]
        g2 = random.choice(sample[k2])
        return g, g1, g2

    # Divide sample space into training and testing sample
    def divide_sample_space(self):
        sample_size = sum([len(v) for v in self.sample.values()])
        train_size = int(sample_size * .7)
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
