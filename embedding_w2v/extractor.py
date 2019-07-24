import glob
import queue
import time
import binaryninja as binja
import pickle
from gensim.models.keyedvectors import KeyedVectors
from obj.base import Object as Obj
import os
# import sys
# sys.path.insert(0,'/home/yijiufly/Downloads/binaryninja/python')
# print(sys.path)

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


#
# def get_cfg(bv: binja.binaryview.BinaryView, func: binja.function.Function):
#     cfg = networkx.DiGraph()
#     control_blocks = {}
#     func_end = max(b.end for b in func)
#     for block in func:
#         base = block.start
#         end = block.end
#         *_, last_inst = block
#         last_inst_addr = end - last_inst[1]
#         control_blocks[last_inst_addr] = (base, last_inst_addr)
#     build_network(bv, cfg, control_blocks, func_end)
#
#     for node_id in cfg:
#         start, end = cfg.node[node_id]['label']
#         calc_statistics(func, start)
#
#
# def build_network(bv: binja.binaryview.BinaryView, cfg: networkx.DiGraph, control_blocks: dict, func_end):
#     keys = sorted(control_blocks.keys())
#     visited = {}
#     for addr in keys:
#         start, end = control_blocks[addr]
#         src_node = (start, end)
#         if src_node not in visited:
#             src_id = len(cfg)
#             visited[src_node] = src_id
#             cfg.add_node(src_id)
#             cfg.node[src_id]['label'] = src_node
#         else:
#             src_id = visited[src_node]
#
#         if start == func.start:
#             cfg.node[src_id]['c'] = 'start'
#         if end == func_end:
#             cfg.node[src_id]['c'] = 'end'
#         refs = bv.get_code_refs(start)
#         for ref in refs:
#             if ref.address not in control_blocks:
#                 continue
#             dst_node = control_blocks[ref.address]
#             if dst_node not in visited:
#                 visited[dst_node] = len(cfg)
#             dst_id = visited[dst_node]
#             cfg.add_edge(dst_id, src_id)
#             cfg.node[dst_id]['label'] = dst_node


def calc_statistics(func, block):
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


# # create dictionary by using all data
# def create_dict_word2vec():
#     dicts = []
#     for path in glob.iglob(
#             r"/home/yzheng04/Workspace/Deepbitstech/Allinone/embedding/test/out_analysis/x64/*.so"):
#         # print(path)
#         dict_word = inst_emb_gen(path)
#         dicts.extend(dict_word)
#     return dicts


def calc_w2v_features(func, block, symbol_map, string_map, model):
    # load pre-trained word2vec model

    inst = get_instructions(block, symbol_map, string_map, dict_gen=1)
    inst_emb = model[inst]
    sum_inst = sum(inst_emb)
    avg_inst = sum_inst / len(inst_emb)
    data = [0, 0]
    data.extend(avg_inst)
    return data


def calc_descendents(block):
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


def build_neighbors(func, bb_map):
    edge_list = []
    for block in func:
        src_id = bb_map[block.start]
        for edge in block.outgoing_edges:
            dst_id = bb_map[edge.target.start]
            edge_list.append((src_id, dst_id))
    return edge_list


def disassemble(path):
    bv = binja.BinaryViewType.get_view_of_file(path)
    # bv.add_analysis_option('linearsweep')
    # binja.log_to_stdout(True)
    # bv.update_analysis_and_wait()
    # print("flag 1")

    s = time.time()
    model = KeyedVectors.load_word2vec_format("model.bin")

    symbol_map = {}
    for sym in bv.get_symbols():
        symbol_map[sym.address] = sym.full_name
    string_map = {}
    for string in bv.get_strings():
        string_map[string.start] = string.value

    raw_graph_list = []
    for func in bv.functions:
        bb_map = BasicBlockMap()

        edge_list = build_neighbors(func, bb_map)
        fvec_list = [0] * len(bb_map)

        for block in func:
            # if block.length == 0:
            #     continue
            # fv_list = calc_statistics(func, block)
            try:
                #fv_list = calc_statistics(func, block)  # use acfg
                fv_list = calc_w2v_features(func, block, symbol_map, string_map, model)  # use w2v block emb
            except:
                fv_list = [0] * 102
            fvec_list[bb_map[block.start]] = fv_list
            # print("=======================")
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


def inst_emb_gen(path, dict_gen=1):
    bv = binja.BinaryViewType.get_view_of_file(path)

    symbol_map = {}
    for sym in bv.get_symbols():
        symbol_map[sym.address] = sym.full_name
    string_map = {}
    for string in bv.get_strings():
        string_map[string.start] = string.value

    acfgs = Obj
    acfgs.raw_graph_list = []

    inst_list = []
    for func in bv.functions:
        bb_map = BasicBlockMap()
        block_insts = []

        acfg = Obj()
        acfg.funcname = func.name
        fv_list = []
        acfg.fv_list = fv_list
        edge_list = []
        acfg.edge_list = edge_list

        for block in func:
            src_id = bb_map[block.start]
            for edge in block.outgoing_edges:
                dst_id = bb_map[edge.target.start]
                edge_list.append((src_id, dst_id))

            insts = get_instructions(block, symbol_map, string_map, dict_gen)
            block_insts.append(insts)
        inst_list.extend(block_insts)
    # print(inst_list)
    return inst_list


def get_instructions(block, symbol_map, string_map, dict_gen=1):
    retval = []
    for tokens, inst_len in block:
        for i, token in enumerate(tokens):
            if token.type == binja.InstructionTextTokenType.CodeRelativeAddressToken:
                tokens[i] = '<tag>'
            elif token.text.startswith('0x'):
                try:
                    val = int(token.text[2:], 16)
                except:
                    val = 0
                if val in symbol_map:
                    tokens[i] = 'foo'
                elif val in string_map:
                    tokens[i] = '<str>'
                else:
                    tokens[i] = '0'
            else:
                v = str(tokens[i])
                if v.isspace():
                    v = ' '
                tokens[i] = v
        retval.append(''.join(tokens).upper().replace(' ', '~'))
    A = '\n'.join(retval).replace(', ', ',')

    if dict_gen:
        # print(retval)
        return retval
    else:
        return A

if __name__ == '__main__':
    for dir in os.listdir('/home/yijiufly/Downloads/codesearch/data/openssl'):
        #for ida_path in glob.iglob(r'/home/yijiufly/Downloads/codesearch/data/openssl/'+dir+'/*.so'):
        ida_path = '/home/yijiufly/Downloads/codesearch/data/openssl/'+dir+'/libssl.so'
        if os.path.exists(ida_path + '.ida'):
            continue
        #print(ida_path)
        start = time.time()
        acfgs = disassemble(ida_path)
        elapse = time.time() - start
        print(ida_path, elapse)
        pickle.dump(acfgs, open(ida_path + '.ida', 'wb'),protocol=2)
