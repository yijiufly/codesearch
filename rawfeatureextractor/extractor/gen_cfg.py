from func import *
from raw_graphs import *
from idc import *
import os
import hashlib
import sys

# idapython plugin used for extract acfg from binary
# then save acfgs to '.ida' file
if __name__ == '__main__':
	#path = '/home/yijiufly/Downloads/codesearch/out_analysis/test'
	path = idc.ARGV[1]
	
	# used for docker
	# path = "/output"
	analysis_flags = idc.get_inf_attr(idc.INF_AF)
	analysis_flags &= ~idc.AF_IMMOFF
	# turn off "automatically make offset" heuristic
	idc.set_inf_attr(idc.INF_AF, analysis_flags)
	idaapi.auto_wait()

	# cfgs = get_func_cfgs_c(FirstSeg())
	binary_name = idc.get_root_filename()
	# fullpath = os.path.join(path, binary_name)
	# pickle.dump(cfgs, open(fullpath,'w'))

	gen_cfg_gdl(idc.get_first_seg(), path)
	
	idc.qexit(0)
