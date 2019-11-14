import numpy as np
from labels import Labels
import cPickle as p
import sys
import operator
import pdb
from testbinary import TestBinary
from library import build1gram, build2gram, grouping
import sys
import os
import time
import traceback
import multiprocessing
import argparse
import mongowrapper.MongoWrapper as mdb
import ConfigParser
def loadFiles(PATH, ext=None):  # use .ida or .emb for ida file and embedding file
    filenames = []
    filenames = [f for f in os.listdir(PATH) if f.endswith(ext)]
    return filenames

def loadFile(PATH):
    if os.path.exists(PATH):
        return PATH
    else:
        raise Exception

def testABinary(binPath, binName):
    start_time = time.time()
    try:
        dotfile = loadFile(os.path.join(binPath, binName + '_bn.dot'))
        namfile = loadFile(os.path.join(binPath, binName + '.ida_newmodel_withsize.nam'))
        embfile = loadFile(os.path.join(binPath, binName + '.ida_newmodel.emb'))
        strfile = loadFile(os.path.join(binPath, binName + '.str'))

        testbin = TestBinary(binName, binPath, dotfile, embfile, namfile, strfile, 1)
        testbin.search()
        print("--- %s seconds ---" % (time.time() - start_time))
    except Exception:
        print 'Missing generate embeddings, callgraphs, string files'
        print traceback.format_exc()



def parallelQuery(folders):
    folders.sort()
    l = len(folders)
    cores = 2#multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    num = l/cores
    start_time = time.time()
    for i in range(cores):
        pool.apply_async(testSomeBinary, (i*num, (i+1)*num, folders, ))
    if num*cores < l:
        pool.apply_async(testSomeBinary, (num*cores, l, folders, ))

    pool.close()
    pool.join()
    print "parallelQuery done"
    print("--- totoal time: %s seconds ---" % (time.time() - start_time))

def testSomeBinary(i, j, folders, dir, binName):
    #print "PID:", os.getpid()
    #print i,j
    for folder in folders[i:j]:
        print folder
        start_time = time.time()
        try:
            dotfile = loadFile(os.path.join(dir, folder, binName + '_bn.dot'))
            namfile = loadFile(os.path.join(dir, folder, binName + '.ida_newmodel_withsize.nam'))
            embfile = loadFile(os.path.join(dir, folder, binName + '.ida_newmodel.emb'))
            strfile = loadFile(os.path.join(dir, folder, binName + '.str'))
        except Exception:
            print 'Missing generate embeddings, callgraphs, string files'
            print traceback.format_exc()
            continue

        binFolder = os.path.join(dir, folder)

        testbin = TestBinary(binName, binFolder, dotfile, embfile, namfile, strfile, 1)
        testbin.search()
        
        print("--- %s seconds ---" % (time.time() - start_time))


def indexing():
    ####build 2-gram lsh
    path_lib = '/home/yijiufly/Downloads/codesearch/data/zlib/zlib-O2'
    for folder in os.listdir(path_lib):
        build2gram(path_lib, folder, 'libz')
    # path_lib = '/home/yijiufly/Downloads/codesearch/data/openssl'
    # for folder in os.listdir(path_lib):
    #     build2gram(path_lib, folder,'libcrypto')
    #     build2gram(path_lib, folder,'libssl')

    ####build function lsh
    path_lib = '/home/yijiufly/Downloads/codesearch/data/zlib/zlib-O2'
    for folder in os.listdir(path_lib):
        build1gram(path_lib, folder, 'libz')
    # path_lib = '/home/yijiufly/Downloads/codesearch/data/openssl'
    # for folder in os.listdir(path_lib):
    #     build1gram(path_lib, folder,'libcrypto')
    #     build1gram(path_lib, folder,'libssl')


def addingStrings(binPath, binName):
    config = ConfigParser.RawConfigParser()
    config.read('config')
    stringfiles = loadFiles(binPath, ext='.str')
    mongodb = mdb(config.get("Mongodb", "DBNAME"),  config.get("Mongodb", "TABLENAME"))
    for s in stringfiles:
        string_dict = p.load(open(s, 'rb'))
        for key in string_dict.keys():
            my_dict = {"name": key, "strings": string_dict[key], "version": binName}
            print my_dict
            mongodb.save(my_dict)

def parse_command():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument("--path", type=str, help="The complete path of folder from which binary file to be preprocessed", required=True)
	parser.add_argument("--name", type=str, help="The name of binary file to be processed", required=True)
        parser.add_argument("--mode", type=str, help="The type of action to take", required=True)
	args = parser.parse_args()
	return args

#######################################################################
#Main
#######################################################################
if __name__ == '__main__':
	#get arguments
    args = parse_command()
    binPath = args.path
    binName = args.name
    mode = args.mode
    #dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx'
    
    if mode == "Searching":
        if not os.path.exists(os.path.join(binPath, binName)):
            exit(1)
        testABinary(binPath, binName)

    elif mode == "SearchingMul":
        if not os.path.exists(binPath):
            exit(1)
        folders = os.listdir(binPath)
        folders.sort()
        #parallelQuery(folders)
        testSomeBinary(0, len(folders), folders, binPath, binName)
    elif mode == "Indexing":
        #indexing()
        grouping()
    elif mode == "AddingStrings":
        addingStrings(binPath, binName)
    else:
        print "Invalid command " + mode
