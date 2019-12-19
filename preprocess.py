import os
import argparse
import hashlib
import subprocess
import shlex
import sys
import shutil
import glob
import mongowrapper.MongoWrapper as mdb
import ConfigParser
import pickle as p
import time

def parse_command():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        "--path", type=str, help=",The complete path of folder from which binary file to be preprocessed", required=True)
    parser.add_argument(
        "--name", type=str, help="The name of binary file to be preprocessed", required=True)
    # parser.add_argument("--out", type=str,
    #                     help="The output path of IDA file", required=True)
    args = parser.parse_args()
    return args

def build_libDB(path, name):
    configParser = ConfigParser.RawConfigParser()
    configParser.read('config')
    mongodb = mdb(configParser.get("Mongodb", "DBNAME"),  configParser.get("Mongodb", "TABLENAME"))
    for folder in os.listdir(path):
        string_path = os.path.join(path, folder, name + '.str')
        assert os.path.exists(string_path)
        string_table = p.load(open(string_path, 'r'))
        for func in string_table.keys():
            my_dict = {"name": func, "strings": string_table[func], "version": folder, "library": name}
            #print my_dict
            mongodb.save(my_dict)
        print 'save ' + str(len(string_table.keys())) + ' strings'

def run_binaryninja(path, name, is_lib):
    sys.path.insert(0, './Gemini')
    from gemini_feature_extraction_ST import disassemble
    import binaryninja as bn
    if is_lib:
        # libraries are dealed with in batch (all versions at the same time), and generate callgraph from a set of .o files
        from gencallgraph_lib import gencallgraph
        libdir = path
        for folder in os.listdir(libdir):
            start = time.time()
            # if os.path.exists(libdir + '/' + folder + '/edges.p'):
            #     continue
            calledges = []
            for objpath in glob.iglob(libdir + '/' + folder + '/objfiles/*.o'):
                print(objpath)
                view = bn.BinaryViewType.get_view_of_file(objpath)
                # generate callgraph
                gencallgraph(view, calledges)

                # generate ida, str files
                acfgs, string_dict = disassemble(view)
                p.dump(acfgs, open(objpath + '.ida', 'wb'), protocol=2)
                p.dump(string_dict, open(objpath + '.str', 'wb'), protocol=2)
            p.dump(calledges, open(libdir + '/' + folder + '/edges.p', 'w'))
            elapse = time.time() - start
            print(libdir, folder, elapse)
    else:
        # deal with one binary each time
        start = time.time()
        from gencallgraph_bin import generate_callgraph_thread
        objpath = os.path.join(path, name)
        print(objpath)
        view = bn.BinaryViewType.get_view_of_file(objpath)
        graph = generate_callgraph_thread(view)
        outf = open(path+'_bn.dot','w')
        outf.write(graph)
        outf.close()

        # generate ida, str files
        acfgs, string_dict = disassemble(view)
        p.dump(acfgs, open(objpath + '.ida', 'wb'), protocol=2)
        p.dump(string_dict, open(objpath + '.str', 'wb'), protocol=2)
        elapse = time.time() - start
        print(objpath, elapse)

def gen_embeddings(path, name, is_lib):
    pass

def build_faissDB():
    pass

# pass binary dir and binary name as argument
if __name__ == '__main__':
    args = parse_command()
    # get as an argument
    bin_path = args.path
    bin_name = args.name
    build_libDB(bin_path, bin_name)
