import os

from model import UniSkip, Encoder
from data_loader import DataLoader
from vocab import load_dictionary
from config import *
from torch import nn

from torch.autograd import Variable
import torch
import numpy as np
import eval_utils as utils
import word2vec
import pickle


# read test data

with open('./saved_models/w2i.pkl', 'rb') as f:
    w2i = pickle.load(f)
vocab_size = len(w2i)
print('vocab_size:', vocab_size)

usable_encoder = utils.UsableEncoder()
usable_w2v = utils.UsableWord2Vec(vocab_size)

functions1 = {}
functions2 = {}

with open('./data/test/sequences_ginstall_706_O3.txt') as f1:
    text = f1.readlines()
    for num, line in enumerate(text):
        if num % 2 == 0:
            seq = text[num+1].strip().split(';')
            seq = [s for s in seq if s]
            if len(seq) > 20000:
                seq = seq[:20000]
            if len(seq) > 10:
                functions1[text[num].strip()] = seq

with open('./data/test/sequences_ginstall_830_O3.txt') as f2:
    text = f2.readlines()
    for num, line in enumerate(text):
        if num % 2 == 0:
            seq = text[num+1].strip().split(';')
            seq = [s for s in seq if s]
            if len(seq) > 20000:
                seq = seq[:20000]
            if len(seq) > 10:
                functions2[text[num].strip()] = seq

acc_skip_thought = 0
acc_w2v = 0

length_skip_thought = 0
length_w2v = 0

for f1, text1 in functions1.items():
    skip_thought_result1 = usable_encoder.encode(text1)
    skip_thought_result1 = skip_thought_result1.sum(axis=0)
    skip_thought_result1_normed = skip_thought_result1 / len(text1)

    w2v_result1 = usable_w2v.encode(text1, w2i)
    w2v_result1 = w2v_result1.sum(axis=0)
    w2v_result1_normed = w2v_result1 / len(text1)    

    distance_lst_skip_thought = []
    target_distance_skip_thought = -1

    distance_lst_w2v = []
    target_distance_w2v = -1

    for f2, text2 in functions2.items():

            skip_thought_result2 = usable_encoder.encode(text2)
            skip_thought_result2 = skip_thought_result2.sum(axis=0)
            skip_thought_result2_normed = skip_thought_result2 / len(text2)

            w2v_result2 = usable_w2v.encode(text2, w2i)
            w2v_result2 = w2v_result2.sum(axis=0)
            w2v_result2_normed = w2v_result2 / len(text2)

            distance_skip_thought = np.linalg.norm(skip_thought_result1_normed - skip_thought_result2_normed)
            distance_w2v = np.linalg.norm(w2v_result1_normed - w2v_result2_normed)
            if f1 == f2:
                target_distance_skip_thought = distance_skip_thought
                target_distance_w2v = distance_w2v
            else:
                distance_lst_skip_thought.append(distance_skip_thought)
                distance_lst_w2v.append(distance_w2v)

    if target_distance_skip_thought != -1: 
        if target_distance_skip_thought < min(distance_lst_skip_thought):
            acc_skip_thought += 1
            length_skip_thought += 1
        else:
            length_skip_thought += 1

    if target_distance_w2v != -1: 
        if target_distance_w2v < min(distance_lst_w2v):
            acc_w2v += 1
            length_w2v += 1
        else:
            length_w2v += 1
    print("skip_thought: {}\tw2v: {}".format(acc_skip_thought/length_skip_thought, acc_w2v/length_w2v))
print("==============================")
print("skip_thought: {}\tw2v: {}".format(acc_skip_thought/length_skip_thought, acc_w2v/length_w2v))

