"""
Test embedding implementation using TF/KERAS

By: yu

"""
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#to use tfdbg
#wrap session object with debugger wrapper

from tensorflow.python import debug as tf_debug
from random import shuffle

import tensorflow as tf
import numpy as np
import os
import operator
import time

#local library
from siamese_emb import Siamese
from dataset import BatchGenerator
'''
import pickle as p

from embedding import Embedding
'''
flags.DEFINE_integer('vector_size', 100, "Vector size of acfg")
flags.DEFINE_integer('emb_size', 64, "Embedding size for acfg")
flags.DEFINE_float('learning_rate', 0.001, "Learning Rate for Optimizer")
flags.DEFINE_string('data_file', 'train.pickle', "Stores the train sample after preprocessing")
flags.DEFINE_string('test_file', 'test.pickle', "Stores the test sample after preprocessing")
flags.DEFINE_integer('T', 5, "Number of time to be interated while embedding generation")
'''
# GPU utilization percentage

#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
WANNACRYPATH    = "/home/yzheng04/Downloads/wannacry/"

MALWARE1        = "0fb900a14c943976cfc9029a3a710333f5434bea9e3cccacd0778bd25ccfa445.ida"
MALWARE2        = "5f18adb787561f2f2d82dcd334dc77a9cac1982b33c52bd4abf977c1d94b5dd8.ida"

def test_siamese():
    test1 = Embedding()

    # for i in range( 0 ):
    #     # newidafile = "openssl.x86_64_O" + str(i) + ".ida"
    #     # newidafile = "0.001_" + str(i) + ".ida"
    #     newidafile = "sample.exe.ida"
    #     embedding = Embedding.embed_a_binary(test1, newidafile)[1]
    #     p.dump(embedding, open("emb_sample" ,'wb'))

    ### embedding single file
    # newidafile = "0.001_0.ida"
    newidafile = WANNACRYPATH + MALWARE2
    embedding = Embedding.embed_a_binary(test1, newidafile)[1]
    p.dump(embedding, open("emb_malware_2" ,'wb'))

if __name__ == "__main__":
    test_siamese()
