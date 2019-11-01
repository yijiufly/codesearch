import glob
import pickle
import queue
import time
import re
import os
import numpy as np
import eval_utils as utils
import multiprocessing
import binaryninja as binja
from collections import defaultdict
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


def calc_st_embeddings(usable_encoder: utils.UsableEncoder, bv: binja.BinaryViewType, block: binja.BasicBlock):
    text = []
    idx = block.start
    for inst in block:
        text.append(parse_instruction(bv.get_disassembly(idx)))
        idx += inst[1]
    if text:
        embd = usable_encoder.encode(text).sum(axis=0) / len(text)
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

def get_strings_per_function(bv):
    addr = []
    string_dict = defaultdict(list)
    for key in bv.sections.keys():
        if 'rodata' in key:
            start = bv.sections[key].start
            end = bv.sections[key].end
            for string in bv.get_strings(start, end):
                addr.append(string.start)
    if addr == []:
        for string in bv.get_strings():
            addr.append(string.start)

    for func in bv.functions:
        string_list = []
        for block in func.low_level_il.basic_blocks:
            for instr in block:
                for oper in instr.operands:
                    if isinstance(oper, binja.LowLevelILInstruction):
                        if oper.operation == binja.LowLevelILOperation.LLIL_CONST_PTR or oper.operation == binja.LowLevelILOperation.LLIL_CONST:
                            if int(str(oper), 16) in addr:
                                string_list.append(bv.get_string_at(int(str(oper), 16)).value)
        if len(string_list) > 0:
            string_list_strip = list(set(string_list))
            string_dict[func.name] = string_list_strip

    return string_dict



def disassemble(path):
    bv = binja.BinaryViewType.get_view_of_file(path)
    binja.log_to_stdout(True)

    usable_encoder = utils.UsableEncoder()
    s = time.time()

    raw_graph_list = []
    for func in bv.functions:
        bb_map = BasicBlockMap()

        edge_list = build_neighbors(func, bb_map)
        fvec_list = [0] * len(bb_map)

        for block in func:
            # fv_list = calc_statistics(func, block)
            fv_list = calc_st_embeddings(usable_encoder, bv, block)
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
    string_dict = get_strings_per_function(bv)
    return acfgs, string_dict

def _worker(q, target, args):
    ret = target(args)
    q.put(ret)


def run_process_with_timeout(target, args, timeout=3600):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(q, target, args))
    p.start()
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not p.is_alive():
                break
            try:
                return q.get(timeout=5)
            except:
                continue
    finally:
        p.terminate()
        p.join()
        q.close()

if __name__ == '__main__':
    #for dir in os.listdir('/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx'):
    #    ida_path = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx/'+dir+'/'+dir
    #for ida_path in glob.iglob(r'/home/yijiufly/Downloads/codesearch/data/nginx/openssl-1.0.1d/objfiles/*.o'):
    for ida_path in glob.iglob(r'/home/yijiufly/Downloads/codesearch/data/openssl/openssl-1.0.1d/objfiles*/*.o'):
        # if not os.path.splitext(ida_path)[-1]:
        print(ida_path)
        start = time.time()
        acfgs, string_dict = run_process_with_timeout(disassemble, ida_path)
        elapse = time.time() - start
        print(ida_path, elapse)
        pickle.dump(acfgs, open(ida_path + '.ida', 'wb'),protocol=2)
        pickle.dump(string_dict, open(ida_path + '.str', 'wb'),protocol=2)
