import glob
import pickle
import queue
import time
import re
import os
import numpy as np
import eval_utils as utils

import binaryninja as binja

from obj import Obj

CALL_INST = {binja.LowLevelILOperation.LLIL_CALL, binja.LowLevelILOperation.LLIL_CALL_PARAM,
             binja.LowLevelILOperation.LLIL_CALL_OUTPUT_SSA, binja.LowLevelILOperation.LLIL_CALL_SSA,
             binja.LowLevelILOperation.LLIL_CALL_STACK_ADJUST, binja.LowLevelILOperation.LLIL_CALL_STACK_SSA}
LOGIC_INST = {binja.LowLevelILOperation.LLIL_AND, binja.LowLevelILOperation.LLIL_TEST_BIT,
              binja.LowLevelILOperation.LLIL_OR, binja.LowLevelILOperation.LLIL_XOR,
              binja.LowLevelILOperation.LLIL_NOT, binja.LowLevelILOperation.LLIL_ROR,
              binja.LowLevelILOperation.LLIL_ROL, binja.LowLevelILOperation.LLIL_ASR,
              binja.LowLevelILOperation.LLIL_LSL, binja.LowLevelILOperation.LLIL_LSR}
ARITH_INST = {binja.LowLevelILOperation.LLIL_ADD, binja.LowLevelILOperation.LLIL_ADD_OVERFLOW,
              binja.LowLevelILOperation.LLIL_ADD_OVERFLOW, binja.LowLevelILOperation.LLIL_SUB,
              binja.LowLevelILOperation.LLIL_FSUB, binja.LowLevelILOperation.LLIL_DIVS,
              binja.LowLevelILOperation.LLIL_DIVS_DP, binja.LowLevelILOperation.LLIL_DIVU,
              binja.LowLevelILOperation.LLIL_DIVU_DP, binja.LowLevelILOperation.LLIL_FDIV,
              binja.LowLevelILOperation.LLIL_MUL, binja.LowLevelILOperation.LLIL_MULS_DP,
              binja.LowLevelILOperation.LLIL_MULU_DP, binja.LowLevelILOperation.LLIL_FMUL,
              binja.LowLevelILOperation.LLIL_ADC, binja.LowLevelILOperation.LLIL_SBB,
              binja.LowLevelILOperation.LLIL_BOOL_TO_INT, binja.LowLevelILOperation.LLIL_FLOAT_TO_INT,
              binja.LowLevelILOperation.LLIL_ROUND_TO_INT, binja.LowLevelILOperation.LLIL_INT_TO_FLOAT}
TRANSFER_INST = {binja.LowLevelILOperation.LLIL_IF, binja.LowLevelILOperation.LLIL_GOTO}


class BasicBlockMap(dict):
    def __missing__(self, key):
        v = len(self)
        self[key] = v
        return v


def parse_instruction(ins):

    ins = re.sub('\s\s+', ', ', ins)
    parts = ins.split(', ')
    return ','.join(parts)


def calc_st_embeddings(usable_encoder: utils.UsableEncoder, w2i,  bv: binja.BinaryViewType, block: binja.BasicBlock):
    text = []
    idx = block.start
    for inst in block:
        text.append(parse_instruction(bv.get_disassembly(idx)))
        idx += inst[1]
    if text:
        embd = usable_encoder.encode(text, w2i).sum(axis=0) / len(text)
    else:
        embd = np.zeros(128)
    return embd
# def calc_w2v_embeddings(func: binja.Function, block: binja.BasicBlock):


def calc_statistics(func: binja.Function, block: binja.BasicBlock):
    num_as, num_calls, num_insts, num_lis, num_tis = 0, 0, 0, 0, 0
    idx = block.start
    for inst in block:
        llil = func.get_lifted_il_at(idx)
        idx += inst[1]
        num_insts += 1
        if not hasattr(llil, 'operation'):
            continue
        if llil.operation in CALL_INST:
            num_calls += 1
        elif llil.operation in ARITH_INST:
            num_as += 1
        elif llil.operation in LOGIC_INST:
            num_lis += 1
        elif llil.operation in TRANSFER_INST:
            num_tis += 1
    return [0, 0, calc_descendents(block), num_as, num_calls, num_insts, num_lis, num_tis]


def calc_descendents(block: binja.BasicBlock):
    q = queue.Queue()
    q.put(block)
    visited = set()
    visited.add(block.start)
    cnt = 0

    while not q.empty():
        b = q.get()
        for edge in b.outgoing_edges:
            target = edge.target
            if target.start not in visited:
                cnt += 1
                q.put(target)
                visited.add(target.start)
    return cnt


def build_neighbors(func: binja.Function, bb_map: BasicBlockMap):
    edge_list = []
    for block in func:
        src_id = bb_map[block.start]
        for edge in block.outgoing_edges:
            dst_id = bb_map[edge.target.start]
            edge_list.append((src_id, dst_id))
    return edge_list


def disassemble(path):
    bv = binja.BinaryViewType.get_view_of_file(path)
    binja.log_to_stdout(True)
    with open('./saved_models/w2i.pkl', 'rb') as f:
        w2i = pickle.load(f)
    vocab_size = len(w2i)
    usable_encoder = utils.UsableWord2Vec(vocab_size)
    s = time.time()

    raw_graph_list = []
    for func in bv.functions:
        bb_map = BasicBlockMap()

        edge_list = build_neighbors(func, bb_map)
        fvec_list = [0] * len(bb_map)

        for block in func:
            # fv_list = calc_statistics(func, block)
            fv_list = calc_st_embeddings(usable_encoder, w2i, bv, block)
            fvec_list[bb_map[block.start]] = fv_list

        acfg = Obj()
        acfg.fv_list = fvec_list
        acfg.funcname = func.name
        acfg.edge_list = edge_list
        raw_graph_list.append(acfg)
    acfgs = Obj()
    acfgs.raw_graph_list = raw_graph_list

    elapse = time.time() - s
    print('-------', elapse)
    return acfgs

#
for ida_path in glob.iglob(r'/home/ericlee/projects/Gemini/trainingdataset/*'):
    # if not os.path.splitext(ida_path)[-1]:
    start = time.time()
    acfgs = disassemble(ida_path)
    elapse = time.time() - start
    print(ida_path, elapse)
    # break
    pickle.dump(acfgs, open(ida_path + '.ida', 'wb'))
