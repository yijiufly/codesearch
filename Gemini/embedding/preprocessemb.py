import os
import argparse
import hashlib
import subprocess
import shlex
import sys
import shutil
import pickle as p
from embedding import Embedding
import traceback


def parse_command():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        "--path", type=str, help=",The complete path of folder from which ida file to be preprocessed", required=True)
    args = parser.parse_args()
    return args


def loadFiles(PATH, ext=None):  # use .ida or .emb for ida file and embedding file
    filenames = [f for f in os.listdir(PATH) if f.endswith(ext)]
    return filenames


# pass binary dir and binary name as argument
if __name__ == '__main__':
    # args = parse_command()
    # get as an argument
    try:
        ida_path = sys.argv[1]  # args.path
        lib_name = sys.argv[2]
        isLib = sys.argv[3]
        if isLib == 'True':
            ida = loadFiles(ida_path + '/objfiles', ".ida")
        else:
            ida = loadFiles(ida_path, ".ida")
        # intialtize embedding
        emb = Embedding()
        funcname_all = []
        embedding_all = []
        func_name_list_with_size_all = []
        for i, value in enumerate(ida):
            if isLib == 'True':
                idapath = ida_path + '/objfiles/' + value
            else:
                idapath = ida_path + '/' + value
            # generage embedding for all .emb files
            print(idapath)
            funcname, embedding, func_name_list_with_size = Embedding.embed_a_binary(
                emb, idapath)
            funcname_all.extend(funcname)
            embedding_all.extend(embedding)
            func_name_list_with_size_all.extend(func_name_list_with_size)
        embfile = ida_path + "/" + lib_name + ".so_newmodel.emb"
        p.dump(embedding_all, open(embfile, 'wb'), protocol=2)
        p.dump(funcname_all, open(ida_path + "/" + lib_name + ".so_newmodel.nam", "wb"), protocol=2)
        p.dump(func_name_list_with_size_all, open(ida_path + "/" + lib_name + ".so_newmodel_withsize.nam", "wb"), protocol=2)

        if isLib == 'True':
            # combine all the str files into one dictionary
            strings = loadFiles(ida_path + '/objfiles', ".str")
            global_string_dict = dict()
            for str_file in strings:
                str_dict = p.load(open(ida_path + '/objfiles/' + str_file, 'r'))
                global_string_dict.update(str_dict)

            p.dump(global_string_dict, open(ida_path + "/" + lib_name + ".str", "wb"), protocol=2)

    except Exception:
        print(traceback.format_exc())
