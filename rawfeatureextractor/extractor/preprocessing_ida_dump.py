import os
import ntpath
import idaapi
import idc
import func
import pickle


def get_mapped_addr(name):
    codename = ntpath.basename(name)
    if codename.endswith(".bin"):
        splits = codename.split("-")
        print splits
        id = int(splits[1])
        name = splits[2]
        eip = int(splits[4], 16)
        start = int(splits[5], 16)
        end = int(splits[6][:-4], 16)
        return id, name, eip, start, end


def set_segment_name(ea, name):
    seg = idaapi.getseg(ea)
    if not seg:
        return False
    return idaapi.set_segm_name(seg, name)


def load_dump(path):
    codemap = get_mapped_addr(path)
    dumppath = path
    linput = idaapi.open_linput(dumppath, False)
    suc = idaapi.load_binary_file(codemap[1], linput, idaapi.NEF_SEGS, 0, 0, codemap[3], 0)
    set_segment_name(codemap[3], 'CODEDUMP' + str(codemap[0]))
    idaapi.auto_make_code(codemap[2])
    idaapi.auto_make_code(codemap[3])
    idaapi.refresh_idaview_anyway()
    idaapi.close_linput(linput)


if __name__ == '__main__':
    """Load dump files into IDA and generate acfgs
    ARGV[1]: directory to dump files
    ARGV[2]: the output directory
    """
    dump_path = idc.ARGV[1]
    for root, dirs, files in os.walk(dump_path):
        for f in files:
            if f.endswith('.bin'):
                load_dump(os.path.join(root, f))

    path = idc.ARGV[2]
    analysis_flags = idc.get_inf_attr(idc.INF_AF)
    analysis_flags &= ~idc.AF_IMMOFF
    idc.set_inf_attr(idc.INF_AF, analysis_flags)
    idaapi.auto_wait()
    cfgs = func.get_func_cfgs_c(idc.get_first_seg())
    binary_name = idc.get_root_filename() + '.ida'
    fullpath = os.path.join(path, binary_name)
    pickle.dump(cfgs, open(fullpath, 'w'))
    idc.qexit(0)
