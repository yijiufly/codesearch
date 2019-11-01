import os
from binaryninja import *
import matplotlib.pyplot as plt
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
import re
import pickle

def parse_instruction(ins, symbol_map, string_map):
    ins = re.sub('\s\s+', ', ', ins)
    parts = ins.split(', ')
    # operand = []
    # if len(parts) > 1:
    #     operand = parts[1:]
    # for i in range(len(operand)):
    #     symbols = re.split('([0-9A-Za-z]+)', operand[i])
    #     for j in range(len(symbols)):
    #         if symbols[j][:2] == '0x' and len(symbols[j]) == 8:
    #             if int(symbols[j], 16) in symbol_map:
    #                 symbols[j] = symbol_map[int(symbols[j], 16)]
    #             elif int(symbols[j], 16) in string_map:
    #                 symbols[j] = string_map[int(symbols[j], 16)]
    #     operand[i] = ''.join(symbols)
    # opcode = parts[0]
    return ','.join(parts)



# read test data
with open('./saved_models/w2i.pkl', 'rb') as f:
    w2i = pickle.load(f)
vocab_size = len(w2i)
print('vocab_size:', vocab_size)

usable_encoder = utils.UsableEncoder()
usable_w2v = utils.UsableWord2Vec(vocab_size)

bin_file_1 = "./data/test/ginstall_706_O3"
bin_file_2 = "./data/test/ginstall_830_O3"

ground_truth_file = "./data/test/addrMapping"

basicblock_1 = {}
basicblock_2 = {}

symbol_map1 = {}
string_map1 = {}

symbol_map2 = {}
string_map2 = {}

bv1 = BinaryViewType.get_view_of_file(bin_file_1)
bv2 = BinaryViewType.get_view_of_file(bin_file_2)

for sym in bv1.get_symbols():
    symbol_map1[sym.address] = sym.full_name
for string in bv1.get_strings():
    string_map1[string.start] = string.value

for sym in bv2.get_symbols():
    symbol_map2[sym.address] = sym.full_name
for string in bv2.get_strings():
    string_map2[string.start] = string.value

for func in bv1.functions:
    for block in func.mlil:
        addr = block[0].address
        instructions = []
        for ins in block:
            text = parse_instruction(bv1.get_disassembly(ins.address), symbol_map1, string_map1)
            instructions.append(text)
        
        basicblock_1[addr] = instructions

for func in bv2.functions:
    for block in func.mlil:
        addr = block[0].address
        instructions = []
        for ins in block:
            text = parse_instruction(bv2.get_disassembly(ins.address), symbol_map2, string_map2)
            instructions.append(text)
        basicblock_2[addr] = instructions


ground_truth_bb_pairs = []

## read the ground truth
with open(ground_truth_file) as f:
    for line in f.readlines():
        pair = re.findall('\[(.*?)\]', line)
        # print("pair: {} {}".format(pair[0], pair[1]))
        assert len(pair) == 2
        original_addr_list = pair[0].split(', ')
        mod_addr_list = pair[1].split(', ')
        if len(original_addr_list) > 4 or len(mod_addr_list) > 4 or abs(len(original_addr_list) - len(mod_addr_list)) > 2:
            continue
        else:
            # print(original_addr_list, mod_addr_list)
            original_addr_list = [int(item) for item in original_addr_list if int(item) in basicblock_1]
            mod_addr_list = [int(item) for item in mod_addr_list if int(item) in basicblock_2]
            for bb1 in original_addr_list:
                for bb2 in mod_addr_list:
                    ground_truth_bb_pairs.append((bb1, bb2))

ground_truth_bb_pairs = np.array(ground_truth_bb_pairs)


st_embeddings_1 = []
st_embeddings_2 = []
w2v_embeddings_1 = []
w2v_embeddings_2 = []

bin1_mapping = {}
bin2_mapping = {}


for addr1, text1 in basicblock_1.items():
    if addr1 in ground_truth_bb_pairs[:,0]:
        st_embeddings_1.append(usable_encoder.encode(text1).sum(axis=0) / len(text1))
        w2v_embeddings_1.append(usable_w2v.encode(text1, w2i).sum(axis=0) / len(text1))
        bin1_mapping[addr1] = len(w2v_embeddings_1) - 1

for addr2, text2 in basicblock_2.items():
    if addr2 in ground_truth_bb_pairs[:,1]:
        st_embeddings_2.append(usable_encoder.encode(text2).sum(axis=0) / len(text2))
        w2v_embeddings_2.append(usable_w2v.encode(text2, w2i).sum(axis=0) / len(text2))
        bin2_mapping[addr2] = len(w2v_embeddings_2) - 1     

acc_st = 0
acc_w2v = 0

num_positive = len(ground_truth_bb_pairs)
num_negative = len(ground_truth_bb_pairs) * len(st_embeddings_2) - num_positive

# for rank in range(st_embeddings_2):
print("start")

st_target_rank_lst = []
w2v_target_rank_lst = []


for pair in ground_truth_bb_pairs:
    idx = bin1_mapping[pair[0]]
    idy = bin2_mapping[pair[1]]

    st_source = st_embeddings_1[idx]
    w2v_source = w2v_embeddings_1[idx]
    
    st_target = st_embeddings_2[idy]
    w2v_target = w2v_embeddings_2[idy]

    st_target_distance = np.linalg.norm(st_source-st_target)
    w2v_target_distance = np.linalg.norm(w2v_source-w2v_target)

    st_distance_lst = []
    w2v_distance_lst = []
    for i in range(len(st_embeddings_2)):
        st_distance_lst.append(np.linalg.norm(st_source-st_embeddings_2[i]))
        w2v_distance_lst.append(np.linalg.norm(w2v_source-w2v_embeddings_2[i]))
    
    st_target_rank_lst.append(sorted(st_distance_lst).index(st_target_distance))
    w2v_target_rank_lst.append(sorted(w2v_distance_lst).index(w2v_target_distance))

# calculate & draw ROP curve

st_total_tp = []
st_total_fp = []

w2v_total_tp = []
w2v_total_fp = []


for rank in range(1, len(st_embeddings_2)):
    acc_st = 0
    acc_w2v = 0
    fp_st = 0
    fp_w2v = 0
    for i in range(len(st_target_rank_lst)):
        if st_target_rank_lst[i] < rank:
            acc_st += 1
            fp_st += rank -1
        else:
            fp_st += rank
        if w2v_target_rank_lst[i] < rank:
            acc_w2v += 1
            fp_w2v += rank -1
        else:
            fp_w2v += rank

    st_total_tp.append(acc_st / len(ground_truth_bb_pairs))
    w2v_total_tp.append(acc_w2v / len(ground_truth_bb_pairs))

    st_total_fp.append(fp_st / num_negative)
    w2v_total_fp.append(fp_w2v / num_negative)


plt.figure(1)
plt.title('ROC')
plt.plot(st_total_fp, st_total_tp, '-', label='ST')
plt.plot(w2v_total_fp, w2v_total_tp, '-', label='W2V')
plt.legend(loc='lower right')
plt.xlim([-0.05, 1.1])
plt.ylim([-0.05, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()    



