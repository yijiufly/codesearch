#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Siamese graph embedding implementaition using tensorflow

By:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle as p
import time

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.linalg import block_diag
from sklearn.metrics import auc
# from embedding import Embedding
from dataset import BatchGenerator
# local library%
from siamese_emb import Siamese
from collections import defaultdict
# to use tfdbg
# wrap session object with debugger wrapper
import pdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('vector_size',128, "Vector size of acfg")
flags.DEFINE_integer('emb_size', 64, "Embedding size for acfg")
flags.DEFINE_float('learning_rate', 0.0001, "Learning Rate for Optimizer")
flags.DEFINE_string('data_file', 'train.pickle', "Stores the train sample after preprocessing")
flags.DEFINE_string('test_file', 'test.pickle', "Stores the test sample after preprocessing")
flags.DEFINE_integer('T', 5, "Number of time to be interated while embedding generation")

FILTER_SIZE = 0


def get_some_embedding(it, cnt=35):
    acfg_mat = []
    acfg_nbr_mat = []
    mul_mat = []
    func_name_list = []

    while len(func_name_list) < cnt:
        try:
            data = next(it)
        # data = it
        except StopIteration:
            break
        func_name = data[0]
        acfg = data[1]
        if len(acfg) < FILTER_SIZE:
            continue
        acfg_nbr = data[2]
        func_name_list.append(func_name)
        acfg_mat.append(acfg)
        acfg_nbr_mat.append(acfg_nbr)
        mul_mat.append(np.ones(len(acfg_nbr)))
    if len(mul_mat) != 0:
        acfg_mat = np.vstack(acfg_mat)
        acfg_nbr_mat = block_diag(*acfg_nbr_mat)
        mul_mat = block_diag(*mul_mat)
    return acfg_mat, acfg_nbr_mat, mul_mat, func_name_list


class Training:
    def __init__(self):
        self.g_test_similarity = self.test_similarity_internal()

    def test_similarity_internal(self):
        self.funca = tf.placeholder(tf.float32, (None, None))
        self.funcb = tf.placeholder(tf.float32, (None, None))
        mul = tf.matmul(self.funca, self.funcb, transpose_b=True)
        na = tf.norm(self.funca, axis=1, keepdims=True)
        nb = tf.norm(self.funcb, axis=1, keepdims=True)
        return mul / tf.matmul(na, nb, transpose_b=True)

    def test_similarity(self, sess, funca, funcb):
        # funca: embeddings of list a
        # funcb : embeddings of list b
        # ret: predicted value
        return sess.run(self.g_test_similarity, feed_dict={self.funca: funca, self.funcb: funcb})


def train_siamese(num_of_iterations):
    # Training part
    print("starting graph def")
    with tf.Graph().as_default():
        # init class
        siamese = Siamese()
        data_gen = BatchGenerator(FILTER_SIZE)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        print("siamese model  object initialized")

        init_op = tf.global_variables_initializer()

        print("started session")
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))

        saver = tf.train.Saver()
        with sess.as_default() as sess:

            # can use other optimizers
            optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
            # optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            train_op = optimizer.minimize(siamese.loss)

            print("defined training operations")
            print("initializing global variables")

            sess.run(init_op)
            # Trainning parameters
            TRAIN_ITER = num_of_iterations  # number of iterations in each training
            # model saved path
            SAVEPATH = "./model/model.ckpt"

            ## start of counting training time
            whole_train_start = time.time()

            print("model training start:")
            ROUND = 0
            MAX_ROUND = 1
            MAX_AUC = 0
            ###############################################################
            ####                    training                           ####
            ###############################################################
            while ROUND < MAX_ROUND:
                time_train_start = time.time()
                # Temporary loss value
                ROUND += 1
                temp_loss = 10
                for i in range(1, TRAIN_ITER):  ## default 1k, set to 100 for test
                    g, g1, g2, g3 = data_gen.get_train_acfg()
                    r0, r1 = sess.run([train_op, siamese.loss],
                                  feed_dict={siamese.x1: g[1], siamese.x2: g1[1], siamese.y: 1,
                                             siamese.n1: g[2], siamese.n2: g1[2]})

                    r2, r3 = sess.run([train_op, siamese.loss],
                                  feed_dict={siamese.x1: g3[1], siamese.x2: g2[1], siamese.y: -1,
                                             siamese.n1: g3[2], siamese.n2: g2[2]})

                    # currently saving for best loss modify to save for best AUC
                    if r3 < temp_loss:
                        # Save the variables to disk.
                        saver.save(sess, SAVEPATH)
                        print("Model saved", i, r3)
                        temp_loss = r3

                ## Restore variables from disk for least loss.
                ## To be changed for best AUC
                # end of counting training time
                time_train_end = time.time()
                # get total training time
                print("traing duration: ", time_train_end - time_train_start)


                saver.restore(sess, SAVEPATH)

                ###############################################################
                ####                    validation part                    ####
                ###############################################################
                print("generating embedding for test samples")
                emb_list = []
                name_list = []
                test_list = []
                for v in data_gen.test_sample.values():
                    test_list.extend(v)
                it = iter(test_list)
                emb_func = [siamese.get_embedding()]

                while True:
                    acfg_mat, acfg_nbr_mat, mul_mat, func_name_list = get_some_embedding(it)
                    if len(mul_mat) == 0:
                        break
                    emb = sess.run(emb_func, feed_dict={siamese.x: np.concatenate([acfg_mat, np.transpose(mul_mat)], 1),
                                                    siamese.n: acfg_nbr_mat})
                    emb_list.extend(emb[0])
                    name_list.extend(func_name_list)

                print("evaluating prediction values")

                training = Training()
                resultMat = training.test_similarity(sess, emb_list, emb_list)

                rank_index_list = []

                to_sort_list = tf.placeholder(tf.float32, (None, None))
                sort_func = tf.contrib.framework.argsort(to_sort_list, direction='DESCENDING')

                time_eval_start = time.time()
                for i in range(0, len(resultMat), 5000):
                    ret = sess.run(sort_func, feed_dict={to_sort_list: resultMat[i:i + 5000]})
                    rank_index_list.extend(ret)
                time_eval_end = time.time()
                print("sort duration: ", time_eval_end - time_eval_start)
                del resultMat

                func_counts = len(rank_index_list)
                total_tp = []
                total_fp = []
                for func in range(func_counts):
                    real_name = name_list[func]
                    tp = [0]
                    fp = [0]
                    for rank, idx in enumerate(rank_index_list[func]):
                        if name_list[idx] == real_name:
                            tp.append(1)
                            fp.append(fp[-1])
                        else:
                            tp.append(max(tp[-1], 0))
                            fp.append(fp[-1] + 1)
                    total_tp.append(tp[1:])
                    total_fp.append(fp[1:])
                num_positive = sum(len(v) * len(v) for k, v in data_gen.test_sample.items())
                num_negative = func_counts * func_counts - num_positive
                total_tp = np.sum(total_tp, axis=0, dtype=np.float) / func_counts
                total_fp = np.sum(total_fp, axis=0, dtype=np.float) / num_negative
                time_eval_end = time.time()
                print("eval duration: ", time_eval_end - time_eval_start)
                v_auc = auc(total_fp, total_tp)
                print("Round %d, auc = %f" % (ROUND, v_auc))

                p.dump(total_tp, open("tp_round_" + str(ROUND), 'wb'), protocol=2)
                p.dump(total_fp, open("fp_round_" + str(ROUND), 'wb'), protocol=2)
                ###############################################################
                ####             generate negative samples                 ####
                ###############################################################
                # if MAX_AUC != 0 and abs(v_auc - MAX_AUC) < 0.000001:
                #     break
                MAX_AUC = max(v_auc, MAX_AUC)
                print("generating negative sample pairs for train samples")
                emb_list = []
                name_list = []
                test_list = []
                for v in data_gen.train_sample.values():
                    test_list.extend(v)
                it = iter(test_list)
                emb_func = [siamese.get_embedding()]

                while True:
                    acfg_mat, acfg_nbr_mat, mul_mat, func_name_list = get_some_embedding(it)
                    if len(mul_mat) == 0:
                        break
                    emb = sess.run(emb_func, feed_dict={siamese.x: np.concatenate([acfg_mat, np.transpose(mul_mat)], 1),
                                                    siamese.n: acfg_nbr_mat})
                    emb_list.extend(emb[0])
                    name_list.extend(func_name_list)

                training = Training()
                resultMat = training.test_similarity(sess, emb_list, emb_list)

                rank_index_list = []

                to_sort_list = tf.placeholder(tf.float32, (None, None))
                sort_func = tf.contrib.framework.argsort(to_sort_list, direction='DESCENDING')

                time_eval_start = time.time()
                for i in range(0, len(resultMat), 5000):
                    ret = sess.run(sort_func, feed_dict={to_sort_list: resultMat[i:i + 5000]})
                    rank_index_list.extend(ret)
                time_eval_end = time.time()
                print("sort duration: ", time_eval_end - time_eval_start)
                del resultMat

                func_counts = len(rank_index_list)
                negative_samples = defaultdict(set)
                for func in range(func_counts):
                    real_name = name_list[func]
                    for rank, idx in enumerate(rank_index_list[func]):
                        if name_list[idx] == real_name:
                            break
                        else:
                            negative_samples[real_name].add(name_list[idx])
                data_gen.negative_sample = negative_samples
                print("number of negative keys: ", len(negative_samples.keys()))
            whole_train_end = time.time()
            print("total train duration: ", whole_train_end - whole_train_start)



def plot_eval_siamese(total_fp, total_tp):
    plt.figure(1)
    plt.title('ROC')
    plt.plot(total_fp, total_tp, '-', label='ROC')
    plt.legend(loc='lower right')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == "__main__":
    train_siamese(2000)
    #total_fp = p.load(open('fp_round_1','rb'))
    #total_tp = p.load(open('tp_round_1','rb'))
    #plot_eval_siamese(total_fp, total_tp)
