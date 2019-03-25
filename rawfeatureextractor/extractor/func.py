#
# Reference Lister
#
# List all functions and all references to them in the current section.
#
# Implemented with the idautils module
#
from idautils import *
from idaapi import *
from idc import *
import ida_segment
import ida_frame
import networkx as nx
import cfg_constructor as cfg
import cPickle as pickle
import pdb
from raw_graphs import *
#from discovRe_feature.discovRe import *
from discovRe import *
import json
#import wingdbstub
#wingdbstub.Ensure()
def gt_funcNames(ea):
	funcs = []
	plt_func, plt_data = processpltSegs()
	for funcea in Functions(get_segm_start(ea)):
			funcname = get_unified_funcname(funcea)
			if funcname in plt_func:
				print funcname
				continue
			funcs.append(funcname)
	return funcs

def get_funcs(ea):
	funcs = {}
		# Get current ea
		# Loop from start to end in the current segment
	plt_func, plt_data = processpltSegs()
	for funcea in Functions(get_segm_start(ea)):
		funcname = get_unified_funcname(funcea)
		if funcname in plt_func:
			continue
		func = get_func(funcea)
		blocks = FlowChart(func)
		funcs[funcname] = []
		for bl in blocks:
				start = bl.start_ea
				end = bl.end_ea
				funcs[funcname].append((start, end))
	return funcs

# used for the callgraph generation.
def get_func_namesWithoutE(ea):
	funcs = {}
	plt_func, plt_data = processpltSegs()
	for funcea in Functions(get_segm_start(ea)):
			funcname = get_unified_funcname(funcea)
			if 'close' in funcname:
				print funcea
			if funcname in plt_func:
				print funcname
				continue
			funcs[funcname] = funcea
	return funcs

# used for the callgraph generation.
def get_func_names(ea):
	funcs = {}
	for funcea in Functions(get_segm_start(ea)):
			funcname = get_unified_funcname(funcea)
			funcs[funcname] = funcea
	return funcs

def get_func_bases(ea):
		funcs = {}
		plt_func, plt_data = processpltSegs()
		for funcea in Functions(get_segm_start(ea)):
				funcname = get_unified_funcname(funcea)
				if funcname in plt_func:
					continue
				funcs[funcea] = funcname
		return funcs

def get_func_range(ea):
		funcs = {}
		for funcea in Functions(get_segm_start(ea)):
				funcname = get_unified_funcname(funcea)
		func = get_func(funcea)
		funcs[funcname] = (func.start_ea, func.end_ea)
		return funcs

def get_unified_funcname(ea):
	funcname = get_func_name(ea)
	if len(funcname) > 0:
		if '.' == funcname[0]:
			funcname = funcname[1:]
	return funcname

def get_func_sequences(ea):
	funcs_bodylist = {}
	funcs = get_funcs(ea)
	for funcname in funcs:
		if funcname not in funcs_bodylist:
			funcs_bodylist[funcname] = []
		for start, end in funcs[funcname]:
			inst_addr = start
			while inst_addr <= end:
				opcode = print_insn_mnem(inst_addr)
				funcs_bodylist[funcname].append(opcode)
				inst_addr = next_head(inst_addr)
	return funcs_bodylist

# Extract acfgs from binary
# This function is used by preprocessing_ida.py
# Input is address of the start of the first segment
# Output is a collection of acfgs
def get_func_cfgs_c(ea):
	binary_name = idc.get_root_filename()
	raw_gs = raw_graphs(binary_name)
	externs_eas, ea_externs = processpltSegs()
	i = 0
	for funcea in Functions(get_segm_start(ea)):
		funcname = get_unified_funcname(funcea)
		#if funcname == "dtls1_process_heartbeat":
		func = get_func(funcea)
		i += 1
		icfg = cfg.getCfg(func, externs_eas, ea_externs)
		func_f = get_discoverRe_feature(func, icfg[0])
		raw_g = raw_graph(funcname, icfg, func_f)
		raw_gs.append(raw_g)
		raw_gs.call_graph.add_edge_to(get_cg_node_refs(funcea), funcname)
	raw_gs.call_graph.cal_feature()
	return raw_gs

def gen_cfg_gdl(ea, path):
	binary_name = idc.get_root_filename()
	raw_gs = raw_graphs(binary_name)
	i = 0
	for funcea in Functions(get_segm_start(ea)):
		funcname = get_unified_funcname(funcea)
		func = get_func(funcea)
		# path = "/home/renee/Downloads/genius_bugsearch/raw_feature_extractor/temp/"+str(i)+funcname+".gdl"
		# DIR_PATH = os.path.dirname(os.path.realpath(__file__))
		# TEMP_PATH = os.path.join(DIR_PATH, 'temp')
		# print TEMP_PATH
		i += 1
		fullpath = path+"/"+funcname+".gdl"	
		title = "Graph of "+funcname
		cfg.gen_cfg_graph(func, fullpath, title)
		#if funcname == "ssl23_connect":
		#	break

# Extract function call reference relation
def get_cg_node_refs(funcea):
	callers = set()
	for ref_ea in CodeRefsTo(funcea, 0):
		funcname = get_unified_funcname(ref_ea)
		callers.add(funcname)
	return callers

def get_func_cfgs_ctest(ea):
	binary_name = idc.get_root_filename()
	raw_cfgs = raw_graphs(binary_name)
	externs_eas, ea_externs = processpltSegs()
	i = 0
	diffs = {}
	for funcea in Functions(get_segm_start(ea)):
		funcname = get_unified_funcname(funcea)
		func = get_func(funcea)
		print i
		i += 1
		icfg, old_cfg = cfg.getCfg(func, externs_eas, ea_externs)
		diffs[funcname] = (icfg, old_cfg)
		#raw_g = raw_graph(funcname, icfg)
		#raw_cfgs.append(raw_g)
			
	return diffs

def get_func_cfgs(ea):
	func_cfglist = {}
	i = 0
	for funcea in Functions(get_segm_start(ea)):
		funcname = get_unified_funcname(funcea)
		func = get_func(funcea)
		print i
		i += 1
		try:
			icfg = cfg.getCfg(func)
			func_cfglist[funcname] = icfg
		except:
			pass
			
	return func_cfglist

def get_func_cfg_sequences(func_cfglist):
	func_cfg_seqlist = {}
	for funcname in func_cfglist:
		func_cfg_seqlist[funcname] = {}
		cfg = func_cfglist[funcname][0]
		for start, end in cfg:
			codesq = get_sequences(start, end)
			func_cfg_seqlist[funcname][(start,end)] = codesq

	return func_cfg_seqlist


def get_sequences(start, end):
	seq = []
	inst_addr = start
	while inst_addr <= end:
		opcode = print_insn_mnem(inst_addr)
		seq.append(opcode)
		inst_addr = next_head(inst_addr)
	return seq

def get_stack_arg(func_addr):
	print func_addr
	args = []
	stack = ida_frame.get_frame(func_addr)
	if not stack:
			return []
	firstM = get_first_member(stack)
	lastM = get_last_member(stack)
	i = firstM
	while i <=lastM:
		mName = get_member_name(stack,i)
		mSize = get_member_size(stack,i)
		if mSize:
				i = i + mSize
		else:
				i = i+4
		if mName not in args and mName and ' s' not in mName and ' r' not in mName:
			args.append(mName)
	return args

		#pickle.dump(funcs, open('C:/Documents and Settings/Administrator/Desktop/funcs','w'))

def processExternalSegs():
	funcdata = {}
	datafunc = {}
	for n in xrange(ida_segment.get_segm_qty()):
		seg = ida_segment.getnseg(n)
		ea = seg.start_ea
		segtype = idc.get_segm_attr(ea, idc.SEGATTR_TYPE)
		if segtype in [idc.SEG_XTRN]:
			start = idc.get_segm_start(ea)
			end = idc.get_segm_end(ea)
			cur = start
			while cur <= end:
				name = get_unified_funcname(cur)
				funcdata[name] = hex(cur)
				cur = next_head(cur)
	return funcdata

# produce function and data mapping
# TODO: support arm and powerpc
def processpltSegs():
	funcdata = {}
	datafunc = {}
	for n in xrange(ida_segment.get_segm_qty()):
		seg = ida_segment.getnseg(n)
		ea = seg.start_ea
		segname = get_segm_name(ea)
		if segname in ['.plt', 'extern', '.MIPS.stubs']:
			start = seg.start_ea
			end = seg.end_ea
			cur = start
			while cur < end:
				name = get_unified_funcname(cur)
				funcdata[name] = hex(cur)
				datafunc[cur]= name
				cur = next_head(cur)
	return funcdata, datafunc

		
def processDataSegs():
	funcdata = {}
	datafunc = {}
	for n in xrange(ida_segment.get_segm_qty()):
		seg = ida_segment.getnseg(n)
		ea = seg.start_ea
		segtype = idc.get_segm_attr(ea, idc.SEGATTR_TYPE)
		if segtype in [idc.SEG_DATA, idc.SEG_BSS]:
			start = idc.get_segm_start(ea)
			end = idc.get_segm_end(ea)
			cur = start
			while cur <= end:
				refs = [v for v in DataRefsTo(cur)]
				for fea in refs:
					name = get_unified_funcname(fea)
					if len(name)== 0:
						continue
					if name not in funcdata:
						funcdata[name] = [cur]
					else:
						funcdata[name].append(cur)
					if cur not in datafunc:
						datafunc[cur] = [name]
					else:
						datafunc[cur].append(name)
				cur = next_head(cur)
	return funcdata, datafunc

def obtainDataRefs(callgraph):
	datarefs = {}
	funcdata, datafunc = processDataSegs()
	for node in callgraph:
		if node in funcdata:
			datas = funcdata[node]
			for dd in datas:
				refs = datafunc[dd]
				refs = list(set(refs))
				if node in datarefs:
					print refs
					datarefs[node] += refs
					datarefs[node] = list(set(datarefs[node]))
				else:
					datarefs[node] = refs
	return datarefs


