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


class NewBasicBlock():
    def __init__(self, start):
        self.start = start
        self.instructions = []
        self.outgoing_edges = []
        self.call_sites = []
        self.has_return = False

    def add_instructions(self, ins):
        self.instructions.append(ins)

    def add_outgoing_edges(self, edge):
        self.outgoing_edges.append(edge)

    def add_call_sites(self, call): # call:(instr_idx, call_target)
        self.call_sites.append(call)


class NewFunction():
    def __init__(self, name):
        self.name = name
        self.basic_blocks = []
        self.caller = []
        self.has_call = False

    def add_basic_block(self, bb):
        self.basic_blocks.append(bb)

    def add_caller(self, caller_name):
        self.caller.append(caller_name)



def parse_instruction(ins):

    ins = re.sub('\s\s+', ', ', ins)
    parts = ins.split(', ')
    return ','.join(parts)


def calc_st_embeddings(usable_encoder: utils.UsableEncoder, instructions: list):
    text = []
    for inst in instructions:
        text.append(inst)
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


def build_neighbors(func: NewFunction, bb_map: BasicBlockMap):
    edge_list = []
    for block in func.basic_blocks:
        src_id = bb_map[block.start]
        for edge in block.outgoing_edges:
            #print(src_id, block.start, edge)
            dst_id = bb_map[edge]
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


def disassemble(global_func_dict, usable_encoder):
    s = time.time()
    functions = list(global_func_dict.values())
    raw_graph_list = []
    for func in functions:
        bb_map = BasicBlockMap()
        
        if func.has_call:
            tmp_func = NewFunction('tmp')
            for block in func.basic_blocks:
                idx = 0
                block.call_sites = list(filter(lambda x: x[1] in global_func_dict, block.call_sites))
                for (call_idx, callee_name) in block.call_sites:
                    instructions = block.instructions[idx:call_idx + 1]
                    fv_list = calc_st_embeddings(usable_encoder, instructions)
                    tmp = NewBasicBlock(str(block.start + idx))
                    idx = call_idx + 1
                    tmp.fv_list = fv_list
                    for callee_block in global_func_dict[callee_name].basic_blocks:
                        callee = NewBasicBlock(callee_name + str(block.start) + str(callee_block.start))
                        callee.fv_list = callee_block.fv_list
                        callee.outgoing_edges = [callee_name + str(block.start) + str(i) for i in callee_block.outgoing_edges]
                        if callee.has_return:
                            if idx >= len(block.instructions):
                                for edge in block.outgoing_edges:
                                    callee.add_outgoing_edges(str(edge))
                            else:
                                callee.outgoing_edges.append(str(block.start + call_idx + 1))
                        tmp_func.add_basic_block(callee)
                    tmp.outgoing_edges.append(callee_name + str(block.start) + str(global_func_dict[callee_name].basic_blocks[0].start))
                    tmp_func.add_basic_block(tmp)
                    
                if idx < len(block.instructions):
                    instructions = block.instructions[idx:]
                    fv_list = calc_st_embeddings(usable_encoder, instructions)
                    tmp = NewBasicBlock(str(block.start + idx))
                    tmp.fv_list = fv_list
                    for edge in block.outgoing_edges:
                        tmp.add_outgoing_edges(str(edge))
                    tmp_func.add_basic_block(tmp)

            edge_list = build_neighbors(tmp_func, bb_map)
            fvec_list = [0] * len(bb_map)
            for block in tmp_func.basic_blocks:
                fvec_list[bb_map[block.start]] = block.fv_list
        else:
            edge_list = build_neighbors(func, bb_map)
            fvec_list = [0] * len(bb_map)

            for block in func.basic_blocks:
            # fv_list = calc_statistics(func, block)
                fvec_list[bb_map[block.start]] = block.fv_list

        for i in range(len(fvec_list)):
            if fvec_list[i] is 0:
                fvec_list[i] = np.zeros(128)
        acfg = Obj()
        acfg.fv_list = fvec_list
        acfg.funcname = func.name
        acfg.edge_list = edge_list
        raw_graph_list.append(acfg)
    acfgs = Obj()
    acfgs.raw_graph_list = raw_graph_list

    elapse = time.time() - s
    print('-------', elapse)
    #string_dict = get_strings_per_function(bv)
    return acfgs#, string_dict


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

def get_name(addr, view):
    symbol = view.get_symbol_at(addr)

    if symbol is not None:
        name = symbol.name.replace('!', '_').replace('@', '_')
    else:
        name = 'sub_{:x}'.format(abs(addr))

    return name

def preprocessing(path, global_func_dict, usable_encoder):
    if type(path) == type(''):
        s = time.time()
        bv = binja.BinaryViewType.get_view_of_file(path)
        elapse = time.time() - s
        print('disassemble--', elapse)
    else:
        bv = path
        binja.log_to_stdout(True)

    s = time.time()
    for func in bv.functions:
        new_func = NewFunction(func.name)

        
        for block in func.low_level_il.basic_blocks:
            current_block = NewBasicBlock(block.source_block.start)
            current_addr = block.source_block.start
            current_idx_source = 0
            for instr in block.source_block:
                current_block.add_instructions(parse_instruction(bv.get_disassembly(current_addr)))
                
                # if encounter a calling instruction
                if instr[0][0].text == 'call':
                    # add callee address to caller block's outgoing edges
                    try:
                        addr = int(instr[0][2].text, 16)
                    except:
                        current_addr += instr[1]
                        current_idx_source += 1
                        continue
                    calledge = get_name(addr, bv)
                    current_block.add_call_sites((current_idx_source, calledge))
                    new_func.has_call = True

                current_addr += instr[1]
                current_idx_source += 1

            for instr in block:
                if instr.operation == binja.LowLevelILOperation.LLIL_RET:
                    current_block.has_return = True
            for edge in block.source_block.outgoing_edges:
                current_block.add_outgoing_edges(edge.target.start)
            
            fv_list = calc_st_embeddings(usable_encoder, current_block.instructions)
            current_block.fv_list = fv_list
            new_func.add_basic_block(current_block)

        global_func_dict[func.name] = new_func

if __name__ == '__main__':
    global_func_dict = defaultdict(NewFunction)
    start = time.time()
    usable_encoder = utils.UsableEncoder()
    libdir = '/home/yijiufly/Downloads/codesearch/data/openssl/openssl-1.0.1d/'
    # for ida_path in glob.iglob(libdir + 'objfiles*/*.o'):
    #     # if not os.path.splitext(ida_path)[-1]:
    #     print(ida_path)
    #     #acfgs, string_dict = run_process_with_timeout(disassemble, ida_path)
    #     preprocessing(ida_path, global_func_dict, usable_encoder)

    # pickle.dump(global_func_dict, open(libdir + 'tmp_global_func.p', 'wb'),protocol=2)
    global_func_dict = pickle.load(open(libdir + 'tmp_global_func.p', 'rb'))
    acfgs = disassemble(global_func_dict, usable_encoder)
    elapse = time.time() - start
    pickle.dump(acfgs, open(libdir + 'inline.ida', 'wb'),protocol=2)
    #pickle.dump(string_dict, open(ida_path + '.str', 'wb'),protocol=2)
