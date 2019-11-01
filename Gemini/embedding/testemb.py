import pickle as p
import sys
import pdb
ida1 = '/home/yijiufly/Downloads/codesearch/data/nginx/openssl-1.0.1d/objfiles/e_aes.o'
bv = binja.BinaryViewType.get_view_of_file(ida1)
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
