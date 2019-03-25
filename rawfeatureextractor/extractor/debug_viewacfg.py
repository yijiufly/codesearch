import argparse
import os
import pickle as p
from raw_graphs import *
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
import os

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def load_acfg(filePath):
	acfgs = p.load(open(filePath, "r"))
	return acfgs

def transform_acfgs(filePath, dumpPath):
	acfgs = load_acfg(filePath)
	i = 0
	for acfg in acfgs.raw_graph_list:
		write_dot(acfg.g, os.path.join(dumpPath, str(i)))
		i = i + 1

def parse_command():
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", required=True, type=str, help="The path where store .ida file")
	parser.add_argument("--outpath", required=False, type=str, default=".", help="The directory where dump the output file, default is current folder")
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_command()
	path = args.path
	outPath = args.outpath
	transform_acfgs(path, outPath)
