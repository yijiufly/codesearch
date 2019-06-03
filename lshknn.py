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

def loadData(path):
    f = open(path, "rb")
    # load data
    newNameList = [line.rstrip() for line in f]
    f.close()
    # generate signature
    file11 = []
    for i in range(0, len(newNameList)):
        testsample = newNameList[i]
        PATH1 = "data/versiondetect/test3/funcemb_openssl/" + testsample
        f1 = open(PATH1, 'rb')
        file1 = p.load(f1).tolist()
        file11.append(file1)
        f1.close()
    data = np.array(file11)
    return data, newNameList


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
    # redis_object = Redis(host='localhost', port=6379, db=4)
    # hashMap = getHashMap("testtree-3gram", redis_object, dim=192)

    print("\nStart query for test data")
    testkNN = doSearch(hashMap3, qdata)
    p.dump(testkNN, open(outpath, "w"))

def queryForOneBinary2Gram(qdata, outpath):
    redis_object = Redis(host='localhost', port=6379, db=2)
    hashMap = getHashMap("test3-2gram", redis_object, dim=128)
    print("\nStart query for test data")
    testkNN = doSearch(hashMap, qdata)
    p.dump(testkNN, open(outpath, "w"))

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


def build2gramDB(configname):
    redis_object = Redis(host='localhost', port=6379, db=2)
    hashMap = getHashMap(configname, redis_object, dim=128)
    openssl_data, names = loadDataFromFolder(
        'data/versiondetect/test3/funcemb_openssl/')
    zlib_data, namelist = loadDataFromFolder(
        'data/versiondetect/test3/funcemb_zlibO2')

    #load zlib edges to LSH database
    path_zlib = '/home/yijiufly/Downloads/codesearch/data/zlib/zlib-O2'
    for folder in os.listdir(path_zlib):
        dotfile = loadFiles(os.path.join(path_zlib, folder), ext='.dot')[0]
        path11 = os.path.join(path_zlib, folder, dotfile)
        lib = Library(path11)
        lib.libraryName = folder + '_' + dotfile.rsplit('.',1)[0]
        linklistgraph = lib.callgraphEdges
        keylist = linklistgraph.keys()
        for src in keylist:
            for (des, distance) in linklistgraph[src]:
                try:
                    srcname = lib.libraryName + '{' + lib.ind2FuncName[src] + '}.emb'
                    srcemb = zlib_data[namelist.index(srcname)]
                    desname = lib.libraryName + '{' + lib.ind2FuncName[des] + '}.emb'
                    desemb = zlib_data[namelist.index(desname)]
                    # print(np.concatenate((srcemb, desemb)))
                    #print((srcname, desname))
                    hashMap = addToHashMap(hashMap, [np.concatenate((srcemb, desemb))], [[path_zlib.split('/')[-1], folder, (srcname, desname)]])
                except:
                    #print(traceback.format_exc())
                    pass

        print('load ' + lib.libraryName)

    # load openssl edges to LSH database
    path_openssl = '/home/yijiufly/Downloads/codesearch/data/openssl'
    for folder in os.listdir(path_openssl):
        path11 = os.path.join(path_openssl, folder, 'libcrypto.so.dot')
        lib = Library(path11)
        lib.libraryName = folder.split('-')[1] + '_libcrypto.so'
        linklistgraph = lib.callgraphEdges
        keylist = linklistgraph.keys()
        for src in keylist:
            for (des, distance) in linklistgraph[src]:
                try:
                    srcname = lib.libraryName + '{' + lib.ind2FuncName[src] + '}.emb'
                    srcemb = openssl_data[names.index(srcname)]
                    desname = lib.libraryName + '{' + lib.ind2FuncName[des] + '}.emb'
                    desemb = openssl_data[names.index(desname)]
                    #print((srcname, desname))
                    hashMap = addToHashMap(hashMap, [np.concatenate((srcemb, desemb))], [[path_openssl.split('/')[-1], folder, (srcname, desname)]])
                except:
                    pass
        print('load ' + lib.libraryName)

        path11 = os.path.join(path_openssl, folder, 'libssl.so.dot')
        lib = Library(path11)
        lib.libraryName = folder.split('-')[1]+'_libssl.so'
        linklistgraph = lib.callgraphEdges
        keylist = linklistgraph.keys()
        for src in keylist:
            for (des, distance) in linklistgraph[src]:
                try:
                    srcname = lib.libraryName + '{' + lib.ind2FuncName[src] + '}.emb'
                    srcemb = openssl_data[names.index(srcname)]
                    desname = lib.libraryName + '{' + lib.ind2FuncName[des] + '}.emb'
                    desemb = openssl_data[names.index(desname)]
                    #print((srcname, desname))
                    hashMap = addToHashMap(hashMap, [np.concatenate((srcemb, desemb))], [[path_openssl.split('/')[-1], folder, (srcname, desname)]])
                except:
                    pass

        print('load ' + lib.libraryName)
    return hashMap


def build3gram(folder):

    dotfile = loadFiles(os.path.join(path_lib, folder), ext='.dot')[0]
    #dotfile = 'libcrypto.so.dot'
    path11 = os.path.join(path_lib, folder, dotfile)
    lib = Library(path11)
    #lib.libraryName = folder.split('-')[1] + '_' + dotfile.rsplit('.',1)[0]
    lib.libraryName = folder + '_' + dotfile.rsplit('.',1)[0]
    linklistgraph = lib.callgraphEdges
    keylist = linklistgraph.keys()
    for src in keylist:
        for (des, distance) in linklistgraph[src]:
            if des in keylist:
                for (des2, distance2) in linklistgraph[des]:
                    try:
                        srcname = lib.libraryName + '{' + lib.ind2FuncName[src] + '}.emb'
                        srcemb = lib_data[namelist.index(srcname)]
                        desname = lib.libraryName + '{' + lib.ind2FuncName[des] + '}.emb'
                        desemb = lib_data[namelist.index(desname)]
                        desname2 = lib.libraryName + '{' + lib.ind2FuncName[des2] + '}.emb'
                        desemb2 = lib_data[namelist.index(desname2)]
                        addToHashMap(hashMap3, [np.concatenate((srcemb, desemb, desemb2))], [[path_lib.split('/')[-1], folder, (srcname, desname, desname2)]])
                    except:
                        #print(traceback.format_exc())
                        pass

    print('load ' + lib.libraryName)

def build3gram2(folder):

    #dotfile = loadFiles(os.path.join(path_lib, folder), ext='.dot')[0]
    dotfile = 'libcrypto.so.dot'
    path11 = os.path.join(path_lib, folder, dotfile)
    lib = Library(path11)
    lib.libraryName = folder.split('-')[1] + '_' + dotfile.rsplit('.',1)[0]
    #lib.libraryName = folder + '_' + dotfile.rsplit('.',1)[0]
    linklistgraph = lib.callgraphEdges
    keylist = linklistgraph.keys()
    for src in keylist:
        for (des, distance) in linklistgraph[src]:
            if des in keylist:
                for (des2, distance2) in linklistgraph[des]:
                    try:
                        srcname = lib.libraryName + '{' + lib.ind2FuncName[src] + '}.emb'
                        srcemb = lib_data[namelist.index(srcname)]
                        desname = lib.libraryName + '{' + lib.ind2FuncName[des] + '}.emb'
                        desemb = lib_data[namelist.index(desname)]
                        desname2 = lib.libraryName + '{' + lib.ind2FuncName[des2] + '}.emb'
                        desemb2 = lib_data[namelist.index(desname2)]
                        addToHashMap(hashMap3, [np.concatenate((srcemb, desemb, desemb2))], [[path_lib.split('/')[-1], folder, (srcname, desname, desname2)]])
                    except:
                        #print(traceback.format_exc())
                        pass

    print('load ' + lib.libraryName)
'''
    # load openssl edges to LSH database
    path_openssl = '/home/yijiufly/Downloads/codesearch/data/openssl'
    folders = os.listdir(path_openssl)
    folders.sort()
    for folder in folders[:10]:
        path11 = os.path.join(path_openssl, folder, 'libcrypto.so.dot')
        lib = Library(path11)
        lib.libraryName = folder.split('-')[1] + '_libcrypto.so'
        linklistgraph = lib.callgraphEdges
        keylist = linklistgraph.keys()
        for src in keylist:
            for (des, distance) in linklistgraph[src]:
                if des in keylist:
                    for (des2, distance2) in linklistgraph[des]:
                        try:
                            srcname = lib.libraryName + '{' + lib.ind2FuncName[src] + '}.emb'
                            srcemb = openssl_data[names.index(srcname)]
                            desname = lib.libraryName + '{' + lib.ind2FuncName[des] + '}.emb'
                            desemb = openssl_data[names.index(desname)]
                            desname2 = lib.libraryName + '{' + lib.ind2FuncName[des2] + '}.emb'
                            desemb2 = openssl_data[names.index(desname2)]
                            hashMap = addToHashMap(hashMap, [np.concatenate((srcemb, desemb, desemb2))], [[path_openssl.split('/')[-1], folder, (srcname, desname, desname2)]])
                        except:
                            pass
        print('load ' + lib.libraryName)

        path11 = os.path.join(path_openssl, folder, 'libssl.so.dot')
        lib = Library(path11)
        lib.libraryName = folder.split('-')[1]+'_libssl.so'
        linklistgraph = lib.callgraphEdges
        keylist = linklistgraph.keys()
        for src in keylist:
            for (des, distance) in linklistgraph[src]:
                if des in keylist:
                    for (des2, distance2) in linklistgraph[des]:
                        try:
                            srcname = lib.libraryName + '{' + lib.ind2FuncName[src] + '}.emb'
                            srcemb = openssl_data[names.index(srcname)]
                            desname = lib.libraryName + '{' + lib.ind2FuncName[des] + '}.emb'
                            desemb = openssl_data[names.index(desname)]
                            desname2 = lib.libraryName + '{' + lib.ind2FuncName[des2] + '}.emb'
                            desemb2 = openssl_data[names.index(desname2)]
                            hashMap = addToHashMap(hashMap, [np.concatenate((srcemb, desemb, desemb2))], [[path_openssl.split('/')[-1], folder, (srcname, desname, desname2)]])
                        except:
                            pass

        print('load ' + lib.libraryName)
   '''


# def build3gramDB():
#     try:
#         pool = Pool(processes=1)
#         folders = os.listdir(path_lib)
#         print(len(folders))
#         pool.map(build3gram, folders)
#     except:
#         print(traceback.format_exc())
#         pass

def parallelQuery():
    dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx'
    funcembFolder = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/funcemb_testing'
    folders = os.listdir(dir)
    folders.sort()
    l = len(folders)
    cores = 8#multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    num = l/cores
    start_time = time.time()
    for i in range(cores):
        pool.apply_async(test_some_binary_ngram, (i*num, (i+1)*num, ))
    if num*cores < l:
        pool.apply_async(test_some_binary_ngram, (num*cores, l, ))

    pool.close()
    pool.join()
    print "parallelQuery done"
    print("--- totoal time: %s seconds ---" % (time.time() - start_time))

def test_some_binary_ngram(i, j):
    hashMap3.buildTree()
    dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx'
    funcembFolder = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/funcemb_testing'
    folders = os.listdir(dir)
    folders.sort()
    print i,j
    for folder in folders[i:j]:
        print(folder)
        start_time = time.time()
        try:
            dotfile = loadFiles(os.path.join(dir, folder), ext='.dot')[0]
            namfile = loadFiles(os.path.join(dir, folder), ext='.nam')[0]
        except Exception:
            print traceback.format_exc()
            continue

        binaryName = folder
        dotPath = os.path.join(dir, folder, dotfile)
        namPath = os.path.join(dir, folder, namfile)

        #calculate 3gram
        candidate_output3 = os.path.join(dir, folder, 'test_kNN_0508_3gram.p')
        count_output3 = os.path.join(dir, folder, 'out_0508_3gram.p')
        '''
        candidate_output2 = os.path.join(dir, folder, 'test_kNN_0504_2gram.p')
        count_output2 = os.path.join(dir, folder, 'out_0507_2gram.p')
        '''
        if os.path.isfile(count_output3):
            continue

        testbin = TestBinary(binaryName, dotPath, funcembFolder)
        testbin.buildNGram(namPath)
        if os.path.isfile(count_output3):
            #result3gram = p.load(open(count_output3, 'rb'))
            pass
        else:
            if os.path.isfile(candidate_output3):
                pass
            else:
                queryForOneBinary3Gram(testbin.threeGramList, candidate_output3)

            result3gram = testbin.count(candidate_output3)
            p.dump(result3gram, open(count_output3, 'w'))
        print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    configname = ["test3-one", "test3-2gram", "test3-3gram"]
    grouped_configname = ["1gram-grouped", "2gram-grouped", "3gram-grouped"]
    #build1gramDB(configname[0])
    #hashMap1 = build2gramDB(configname[1])
    #hashMap1.grouping()
    redis_object3 = Redis(host='localhost', port=6379, db=4)
    global hashMap3
    hashMap3 = getHashMap("testtree-3gram", redis_object3, dim=192)
    hashMap3.ungrouping()
    #hashMap3.buildTree()
    #pdb.set_trace()
    # global lib_data, namelist
    # lib_data, namelist = loadDataFromFolder(
    #    '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/funcemb_zlibO2')
    # global path_lib
    # path_lib = '/home/yijiufly/Downloads/codesearch/data/zlib/zlib-O2'
    # for folder in os.listdir(path_lib):
    #     build3gram(folder)
    # lib_data, namelist = loadDataFromFolder(
    #        '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/funcemb_openssl/')
    # path_lib = '/home/yijiufly/Downloads/codesearch/data/openssl'
    # for folder in os.listdir(path_lib):
    #     build3gram2(folder)
    # hashMap3.grouping()
    #parallelQuery()
    #test_some_binary_ngram(0,90)

    # redis_object = Redis(host='localhost', port=6379, db=3)
    # hashMap = getHashMap(configname[2], redis_object, dim = 192)
    #cleanAllBuckets(hashMap)
    # redis_object2 = Redis(host='localhost', port=6379, db=4)
    # grouped_hashMap = getHashMap(grouped_configname[2], redis_object2, dim = 192)
    #grouped_hashMap = addToHashMap(grouped_hashMap, [], [])
    #hashMap2.grouping()
    # hashMap = getHashMap()
    # data, newNameList = loadData("data/versiondetect/test2/versiondetect_func_list.txt")
    # print "Start add data to hash map"
    # hashMap = addToHashMap(hashMap, data, newNameList)
    # add_data, newNameList = loadQueryData("data/versiondetect/test1/versiondetect_addfunc_list.txt")
    # hashMap = addToHashMap(hashMap, add_data, newNameList)
    # hashMap = addToHashMap(hashMap, [])
    # zlib_data, namelist = loadDataFromFolder(
    #     'data/versiondetect/test2/funcemb_output_zlib_O2')
    # hashMap = addToHashMap(hashMap, zlib_data, namelist)

    #hashMap = addToHashMap(hashMap, [], [])

    # qdata, newNameList = loadQueryData("data/versiondetect/test1/versiondetect_query_list.txt")
    # funcnamepath = os.path.join('data/versiondetect/test2/idafiles/0acc5283147612b2abd11d606d5585ac8370fc33567f7f77c0b312c207af3bf9', 'nginx-{openssl-0.9.8r}{zlib-1.2.9}.ida.nam')
    #funcnamepath = 'data/versiondetect/test2/idafiles/0acc5283147612b2abd11d606d5585ac8370fc33567f7f77c0b312c207af3bf9/nginx-{openssl-0.9.8r}{zlib-1.2.9}.ida.nam'
    # qdata, newNameList = loadOneQueryBinary(
    #    funcnamepath, 'data/versiondetect/test2/funcemb_output_testing/')
    # print qdata, newNameList
    #print "Start query for test data"
    #testkNN = doSearch(hashMap, qdata, newNameList)
    # for knn in testkNN:
    #    print knn
    # hashMap = addToHashMap(hashMap, [])
    # testkNN = p.load(open("data/versiondetect/kNN.p","r"))
    # N = output_format("data/versiondetect/versiondetect_func_list.txt", testkNN)
    #p.dump(testkNN, open("data/versiondetect/test2/test_kNN.p", "w"))
