from db import db
import numpy as np
import pickle as p
import os
from library import Library
import pymongo
from redis import Redis
import traceback
from multiprocessing import Pool
import time
from binary import TestBinary
import pdb
import multiprocessing

def _get_binary_hash(file_path):
    fd = open(file_path, 'rb')
    md5 = hashlib.sha256()
    data = fd.read()
    md5.update(data)
    return md5.hexdigest()

def loadFiles(PATH, ext=None):  # use .ida or .emb for ida file and embedding file
    filenames = []
    filenames = [f for f in os.listdir(PATH) if f.endswith(ext)]
    return filenames

def loadQueryData(path):
    f = open(path, "rb")
    # load data
    newNameList = [line.rstrip() for line in f]
    f.close()
    # generate signature
    file11 = []
    for i in range(0, len(newNameList)):
        testsample = newNameList[i]
        PATH1 = "data/versiondetect/test2/func_emb_nginx/" + testsample
        f1 = open(PATH1, 'rb')
        file1 = p.load(f1).tolist()
        file11.append(file1)
        f1.close()
    data = np.array(file11)
    return data, newNameList


def loadOneQueryBinary(funcnamepath, embpath):
    names = p.load(open(funcnamepath, 'r'))
    file11 = []
    embnames = []
    for i in range(len(names)):
        embname = funcnamepath.split('/')[-1][:-8] + "{" + names[i] + "}.emb"
        PATH = embpath + '/' + embname
        f1 = open(PATH, 'rb')
        file1 = p.load(f1).tolist()
        file11.append(file1)
        embnames.append(embname)
        f1.close()
    data = np.array(file11)
    return data, embnames


def loadDataFromFolder(path):
    names = os.listdir(path)
    file11 = []
    for i in names:
        filepath = os.path.join(path, i)
        file = p.load(open(filepath, 'rb')).tolist()
        file11.append(file)

    data = np.array(file11)
    return data, names


def getHashMap(configname, storage_object, dim=64):
    db_instance = db()
    db_instance.loadHashMap(configname, storage_object, dim)

    return db_instance

def cleanAllBuckets(db_instance):
    db_instance.engine.clean_all_buckets()

def addToHashMap(db_instance, data, funcList):
    db_instance.indexing(data, funcList)

    return db_instance


def doSearch(db_instance, query):
    return db_instance.querying(query)


def output_format(queryList, testkNN):
    f = open(queryList, "r")
    list = [line.rstrip() for line in f]
    querydata = []
    for idx, testcases in enumerate(testkNN):
        for testcase in testcases:
            id = list.index(testcase[1])
            # print str(idx + len(list)), str(id), str(1-testcase[2])
            querydata.append([idx + len(list), id, 1 - testcase[2]])
    return querydata


def queryForOneBinary3Gram(qdata, outpath):
    redis_object = Redis(host='localhost', port=6379, db=3)
    hashMap = getHashMap("test3-3gram", redis_object, dim=192)

    print("\nStart query for test data")
    testkNN = doSearch(hashMap, qdata)
    p.dump(testkNN, open(outpath, "w"))

def queryForOneBinary2Gram(qdata, outpath):
    redis_object3 = Redis(host='localhost', port=6379, db=0)
    hashMap3 = getHashMap(["filted1-2gram-binary0", "filted1-2gram-binary1", "filted1-2gram-binary2"], redis_object3, dim=128)
    #hashMap3 = getHashMap(["filted1-2gram-binary"], redis_object3, dim=128)
    print("\nStart query for test data")
    testkNN = doSearch(hashMap3, qdata)
    p.dump(testkNN, open(outpath, "w"))
    return testkNN

def queryForOneBinary1Gram(testbin, outpath):
    redis_object3 = Redis(host='localhost', port=6379, db=1)
    #hashMap3 = getHashMap(["filted1-2gram-binary0", "filted1-2gram-binary1", "filted1-2gram-binary2"], redis_object3, dim=128)
    hashMap3 = getHashMap(["filted1-1gram-binary0", "filted1-1gram-binary1", "filted1-1gram-binary2"], redis_object3, dim=64)
    nodes = set()
    for key in testbin.funcNameFilted.keys():
        if testbin.funcNameFilted[key] != -1:
            nodes.add(key)
    query = []
    for node in nodes:
        emb = testbin.funcName2emb[node]
        query.append([emb, node, 1])
    testkNN = hashMap3.querying(query)
    p.dump(testkNN, open(outpath, "w"))
    return testkNN

def build1gramDB(configname):
    redis_object = Redis(host='localhost', port=6379, db=0)
    hashMap = getHashMap(configname, redis_object, dim=64)
    #redis_object = Redis(host='localhost', port=6379, db=0)
    hashMap = getHashMap(configname, redis_object)
    # openssl_data, names = loadDataFromFolder(
    #     'data/versiondetect/test3/funcemb_openssl/')
    # hashMap = addToHashMap(hashMap, openssl_data, names)
    zlib_data, namelist = loadDataFromFolder(
        'data/versiondetect/test3/funcemb_zlibO2')
    hashMap = addToHashMap(hashMap, zlib_data, namelist)
    hashMap = addToHashMap(hashMap, [], [])


def build3gram2(folder, lib):
    #dotfile = loadFiles(os.path.join(path_lib, folder), ext='.dot')[0]
    filter_size = 1
    dotfile = lib+'.so_bn.dot'
    dot_path = os.path.join(path_lib, folder, dotfile)
    #libraryName = folder.split('-')[1] + '_' + dotfile.rsplit('.',1)[0]
    libraryName = folder + '_' + dotfile.rsplit('.',1)[0]
    emb_path = os.path.join(path_lib, folder, lib+'.so_newmodel.emb')
    nam_path = os.path.join(path_lib, folder, lib+'.so_newmodel_withsize.nam')
    nam_path_full = os.path.join(path_lib, folder, lib+'.so_newmodel.nam')
    libitem = Library(libraryName, dot_path, emb_path, nam_path,filter_size)
    libitem.buildNGram(nam_path_full)
    print(len(libitem.twoGramList))
    for [threegram, name, distance] in libitem.twoGramList:
        addToHashMap(hashMap3, [threegram], [[lib, folder, name, distance]])

    print('load ' + libitem.libraryName)

def build1gram(folder, lib):
    #dotfile = loadFiles(os.path.join(path_lib, folder), ext='.dot')[0]
    filter_size = 1
    dotfile = lib+'.so_bn.dot'
    dot_path = os.path.join(path_lib, folder, dotfile)
    #libraryName = folder.split('-')[1] + '_' + dotfile.rsplit('.',1)[0]
    libraryName = folder + '_' + dotfile.rsplit('.',1)[0]
    emb_path = os.path.join(path_lib, folder, lib+'.so_newmodel.emb')
    nam_path = os.path.join(path_lib, folder, lib+'.so_newmodel_withsize.nam')
    nam_path_full = os.path.join(path_lib, folder, lib+'.so_newmodel.nam')
    libitem = Library(libraryName, dot_path, emb_path, nam_path,filter_size)
    libitem.loadOneBinary(nam_path_full, libitem.embFile)
    for func in libitem.funcNameFilted.keys():
        if libitem.funcNameFilted[func] != -1:
            addToHashMap(hashMap3, [libitem.funcName2emb[func]], [[lib, folder, func]])

    print('load ' + libitem.libraryName)

if __name__ == '__main__':
    ####build 2-gram lsh
    # redis_object3 = Redis(host='localhost', port=6379, db=0)
    # global hashMap3
    # hashMap3 = getHashMap(["filted1-2gram-binary0", "filted1-2gram-binary1", "filted1-2gram-binary2"], redis_object3, dim=128)
    #
    # global path_lib
    # path_lib = '/home/yijiufly/Downloads/codesearch/data/zlib/zlib-O2'
    # for folder in os.listdir(path_lib):
    #     build3gram2(folder, 'libz')
    # path_lib = '/home/yijiufly/Downloads/codesearch/data/openssl'
    # for folder in os.listdir(path_lib):
    #     build3gram2(folder,'libcrypto')
    #     build3gram2(folder,'libssl')
    # hashMap3.grouping()

    ####build function lsh
    redis_object3 = Redis(host='localhost', port=6379, db=1)
    global hashMap3
    hashMap3 = getHashMap(["filted1-1gram-binary0", "filted1-1gram-binary1", "filted1-1gram-binary2"], redis_object3, dim=64)

    global path_lib
    path_lib = '/home/yijiufly/Downloads/codesearch/data/zlib/zlib-O2'
    for folder in os.listdir(path_lib):
        build1gram(folder, 'libz')
    path_lib = '/home/yijiufly/Downloads/codesearch/data/openssl'
    for folder in os.listdir(path_lib):
        build1gram(folder,'libcrypto')
        build1gram(folder,'libssl')
    hashMap3.grouping()
