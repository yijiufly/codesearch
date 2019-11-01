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


import tensorflow as tf
import pickle as p
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from embedding import Embedding

flags = tf.app.flags
FLAGS = flags.FLAGS

# flags.DEFINE_integer('vector_size', 6, "Vector size of acfg")
# flags.DEFINE_integer('emb_size', 64, "Embedding size for acfg")
# flags.DEFINE_float('learning_rate', 0.001, "Learning Rate for Optimizer")
# flags.DEFINE_string('data_file', 'train.pickle', "Stores the train sample after preprocessing")
# flags.DEFINE_string('test_file', 'test.pickle', "Stores the test sample after preprocessing")
# flags.DEFINE_integer('T', 5, "Number of time to be interated while embedding generation")

# GPU utilization percentage

# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


WANNACRYPATH    = "/home/ericlee/projects/Gemini/libc/"

SAMPLE1        = "musl_64_libc.so.ida"
# SAMPLE2        = ["musl_64_libc.so.ida"]
SAMPLE2        = ["musl_32_libc.so.ida",
                  "g_32_libc.so.ida", "g_64_libc.so.ida",
                  "bionic_32_libc.so.ida", "bionic_64_libc.so.ida",
                  "mips_32_libc.so.ida", "mips_64_libc.so.ida"]


test_function = "strncmp"


def test_siamese():
    test = Embedding()
    for binary in SAMPLE2:
        idafile_1 = WANNACRYPATH + SAMPLE1
        idafile_2 = WANNACRYPATH + binary
        func_names_1, embedding_1 = test.embed_a_binary(idafile_1)
        func_names_2, embedding_2 = test.embed_a_binary(idafile_2)

        if test_function in func_names_1:
            result_list = {}
            idx1 = func_names_1.index(test_function)
            for idx2, func2 in enumerate(func_names_2):
                result_list[test_function+' vs '+func2] = cosine_similarity([embedding_1[idx1]],
                                                                            [embedding_2[idx2]]).item()
            result_list = sorted(result_list.items(), key=lambda item: item[1], reverse=True)
            print(SAMPLE1, binary, '\n')
            for name, value in result_list:
                if name == test_function+' vs '+test_function:
                    print(name, ':', value, ':', result_list.index((name,value)) + 1, '\n')
            for name, value in result_list[:20]:
                print(name, ':', value)
            print("")
        else:
            print("no such function")


if __name__ == "__main__":
    test_siamese()
