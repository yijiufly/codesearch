import traceback
try:
    from idc import *
    from func import *
    from raw_graphs import *

    import os
    import hashlib
    import sys
    # idapython plugin used for extract acfg from binary
    # then save acfgs to '.ida' file
    if __name__ == '__main__':
        # path = '.'
        path = idc.ARGV[1] if len(idc.ARGV) >= 1 else '.'
        #path = '/home/yijiufly/Downloads/codesearch/out_analysis/test'
        # used for docker
        # path = "/output"

        analysis_flags = idc.get_inf_attr(idc.INF_AF)
        analysis_flags &= ~idc.AF_IMMOFF
        # turn off "automatically make offset" heuristic
        idc.set_inf_attr(idc.INF_AF, analysis_flags)
        idaapi.auto_wait()
        cfgs = get_func_cfgs_c(idc.get_first_seg())
        binary_name = idc.get_root_filename() + '.ida'
        fullpath = os.path.join(path, binary_name)
        with open(fullpath, 'wb') as f:
            pickle.dump(cfgs, f)
        idc.qexit(0)
except Exception:
    with open('generateidaexception', 'w') as f:
        f.write(traceback.format_exc())
