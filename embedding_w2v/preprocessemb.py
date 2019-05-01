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
    filenames = []
    if ext == ".ida":
        filenames = [f for f in os.listdir(PATH) if f.endswith(ext)]
    if ext == ".emb":
        filenames = [f for f in os.listdir(PATH) if f.endswith(ext)]
    return filenames


# pass binary dir and binary name as argument
if __name__ == '__main__':
    # args = parse_command()
    # get as an argument
    try:
        ida_path = sys.argv[1]  # args.path

        pemb = loadFiles(ida_path, ".emb")
        if len(pemb) > 0:
            print(ida_path + '/' + pemb[0])
            os._exit(0)
        ida = loadFiles(ida_path, ".ida")
        # intialtize embedding
        emb = Embedding()
        OUTPATH = "/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/funcemb_testing/"
        for i, value in enumerate(ida):
            idapath = ida_path + '/' + value
            # generage embedding for all .emb files
            print(idapath)
            funcname, embedding = Embedding.embed_a_binary(
                emb, idapath)
            embfile = idapath + ".emb"
            #p.dump(embedding, open(embfile, 'wb'))
            p.dump(funcname, open(idapath + ".nam", "wb"))
            #p.dump(fullfuncname, open(idapath + ".fullnam", "wb"))
            name = embfile.split('/')[-1]
            label = embfile.split('/')[-2]
            #label = label[8:] + '_' + name
            for i in range(len(embedding)):
                #OUTFILE = OUTPATH + label[8:] + "_"+ name[:-8] + "{" + nams[i] + "}.emb"
                OUTFILE = OUTPATH + label + "{" + funcname[i] + "}.emb"
                #print OUTFILE
                file = open(OUTFILE,'wb')
                p.dump(embedding[i], file, protocol=2)
                file.close()

        # check wheter ida processing done or not
        # if embfile is None:
        #     print("no emb output")
        # else:
        #     print(embfile)
    except Exception:
        print(traceback.format_exc())
