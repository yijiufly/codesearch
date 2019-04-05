from sslh.test_SSLH_inference import *
from sslh.analysis import *
import numpy as np
from labels import Labels
import cPickle as p
import sys
import operator
import pdb
from binary import TestBinary
from library import Library
from lshknn import queryForOneBinary
import os
import time
import traceback

def loadkNNGraph(querykNNPath=None):
    knn = []
    for j in range(38):
        filename = 'data/versiondetect/test1/train_kNN_' + \
            str(j * 10000 + 9999) + '.p'
        data = p.load(open(filename, 'r'))
        for i in data:
            if i[2] > 0.9999 and i[0][0] != i[1][0]:
                knn.append([i[0][0], i[1][0], i[2]])
        print j

    data = p.load(open('data/versiondetect/test1/train_kNN_end.p', 'r'))
    for i in data:
        knn.append([i[0][0], i[1][0], i[2]])
    print knn[0]
    if querykNNPath == None:
        return kNN
    else:
        data = p.load(open(querykNNPath, 'r'))
        for i in data:
            knn.append([i[0][0], i[1][0], i[2]])
        return knn


def train():
    adjMatPath = loadkNNGraph()
    priorsBelievePath = 'data/versiondetect/test1/seedlabels.p'
    resultsPath = 'data/versiondetect/test1/train_belief_addthreshold.txt'
    allLabelNamePath = 'data/versiondetect/test1/alllabels.p'
    runBP(adjMatPath, priorsBelievePath, allLabelNamePath, resultsPath)


def query():
    #priorsBelievePath = 'data/versiondetect/test1/train_belief.txt'
    priorsBelievePath = 'data/versiondetect/test1/beliefAfterFirstRound.p'
    resultsPath = 'data/versiondetect/test1/beliefEnd.txt'
    queryPath = 'data/versiondetect/test1/test_kNN.p'
    queryList = 'data/versiondetect/test1/versiondetect_query_list.txt'
    queryListSize = len(open(queryList, 'r').readlines())
    addFuncList = 'data/versiondetect/test1/versiondetect_addfunc_list.txt'
    addFuncListSize = len(open(addFuncList, 'r').readlines())
    adjMatPath = loadkNNGraph(querykNNPath=queryPath)
    runBPforQuery(adjMatPath, priorsBelievePath, resultsPath,
                  queryPath, queryListSize, addFuncListSize)


def identity_key_functions():
    find_higheset_beliefs_versiondetect()


def analyse():
    results = np.loadtxt(
        open("data/versiondetect/test1/beliefEnd.txt", "r"))[390569:]

    groundTruth = open(
        "data/versiondetect/test1/groundTruth.txt", "r").readlines()
    alllabels = p.load(open("data/versiondetect/test1/alllabels.p", "r"))
    funcNameList = open(
        "data/versiondetect/test1/versiondetect_query_list.txt", "r").readlines()
    label = Labels(funcNameList=funcNameList, finalLabels=results,
                   groundTruth=groundTruth, alllabels=alllabels)
    label.compareOutWithGroundTruth()

## analyse according to the key functions identified by BP algorithm
def analyse_naive():
    #results = np.loadtxt(open("data/versiondetect/beliefEnd.txt", "r"))[603962:]

    groundTruth = open(
        "data/versiondetect/test1/groundTruth.txt", "r").readlines()
    alllabels = p.load(
        open("data/versiondetect/test1/alllabels_withnginx.p", "r"))
    funcNameList = open(
        "data/versiondetect/test1/versiondetect_query_list.txt", "r").readlines()
    label = Labels(funcNameList=funcNameList,
                   groundTruth=groundTruth, alllabels=alllabels)
    label.analyse_naive()


def preprocessing_label():
    funcNameList = open(
        "data/versiondetect/test1/versiondetect_func_sublist.txt", "r").readlines()
    addfuncNameList = open(
        "data/versiondetect/test1/versiondetect_addfunc_list.txt", "r").readlines()
    label = Labels(funcNameList=funcNameList)

    seedlabels = label.funcNameList2seedlabels()
    for i in range(len(addfuncNameList)):
        seedlabels.append(['nginx', 10])

    p.dump(seedlabels, open("data/versiondetect/test1/seedlabels_withnginx.p", "w"))
    label.load_seedlabels(seedlabels)
    allLabel = label.generate_label_vector()
    p.dump(allLabel, open("data/versiondetect/test1/alllabels_withnginx.p", "w"))



## for each function, choose similar functions according to the similarity distribution of candidiate similar functions
## choose the cluster of functions with the highest similarity scores
def choose_threshold(queryPath, threshold=0.999, verbose=True):
    #queryPath = 'data/versiondetect/test2/test_kNN.p'
    #query stores the query knn: each line is [(query_idx, query_func_name), (func_idx, func_name), similarity]
    query = p.load(open(queryPath, 'r'))

    #temporarily stores the similarity distribution of candidiate similar functions
    funcs = []
    #for each function, choose similar functions according to the knn
    chosenFuncs = []

    #stores the name of "each function"
    funcList = []

    lastFunc = ''
    for i in query:
        if i[0][1] != lastFunc and lastFunc != '':
            chosenFuncs.append(funcs)
            funcList.append(lastFunc)
            if verbose:
                print lastFunc
                print chosenFuncs[-1]
                print '\n'
            funcs = []

        if i[2] > threshold:
            funcs.append(i[1][1])
        lastFunc = i[0][1]


    chosenFuncs.append(funcs)
    funcList.append(lastFunc)
    return chosenFuncs, funcList

## detect the version simply according to the vote for each version
def analyse_labelcount(queryPath):
    chosenFuncs, funcList = choose_threshold(queryPath, verbose=True)
    #alllabels = p.load(
    #    open("data/versiondetect/test2/alllabels.p", "r"))

    #label = Labels(funcNameList=funcList)
    #label.analyse_onebinary_labelcount(chosenFuncs)
    return chosenFuncs, funcList

def loadFiles(PATH, ext=None):  # use .ida or .emb for ida file and embedding file
    filenames = []
    filenames = [f for f in os.listdir(PATH) if f.endswith(ext)]
    return filenames

def test_one_binary(path, path2, queryPath, libs):
    #path = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test2/idafiles/0acc5283147612b2abd11d606d5585ac8370fc33567f7f77c0b312c207af3bf9/nginx-{openssl-0.9.8r}{zlib-1.2.9}.dot'
    #path2 = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test2/idafiles/0acc5283147612b2abd11d606d5585ac8370fc33567f7f77c0b312c207af3bf9/nginx-{openssl-0.9.8r}{zlib-1.2.9}.ida.nam'
    testbin = TestBinary(path, path2)
    chosenFuncs, funcList = choose_threshold(queryPath, verbose=False)
    testbin.getRank1Neighbors(chosenFuncs, funcList)
    results=[]
    for lib in libs:
        libraryName, edgeCount = testbin.compareSameEdges(lib)
        results.append([libraryName, edgeCount])
    return results

def load_libs():
    libs=[]
    path_zlib = '/home/yijiufly/Downloads/codesearch/data/zlib/idafiles'
    for folder in os.listdir(path_zlib):
        dotfile = loadFiles(os.path.join(path_zlib, folder), ext='.dot')[0]
        namfile = loadFiles(os.path.join(path_zlib, folder), ext='.nam')[0]
        path11 = os.path.join(path_zlib, folder, dotfile)
        path12 = os.path.join(path_zlib, folder, namfile)
        lib = Library(path11, path12)
        lib.libraryName = dotfile.rsplit('.',1)[0]
        libs.append(lib)
        print 'load ' + lib.libraryName

    path_openssl = '/home/yijiufly/Downloads/codesearch/data/openssl/'
    for folder in os.listdir(path_openssl):
        path11 = os.path.join(path_openssl, folder, 'libcrypto.so.dot')
        path12 = os.path.join(path_openssl, folder, 'libcrypto.so.ida.nam')
        lib = Library(path11, path12)
        lib.libraryName = folder.split('-')[1]+'_libcrypto.so'
        libs.append(lib)
        print 'load ' + lib.libraryName
    return libs

def test_some_binary():
    libs = load_libs()
    dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test2/idafiles'
    for folder in os.listdir(dir):
        start_time = time.time()
        try:
            dotfile = loadFiles(os.path.join(dir, folder), ext='.dot')[0]
            namfile = loadFiles(os.path.join(dir, folder), ext='.nam')[0]
        except Exception:
            print traceback.format_exc()
            continue
        path = os.path.join(dir, folder, dotfile)
        path2 = os.path.join(dir, folder, namfile)
        outpath = os.path.join(dir, folder, 'test_kNN_0403.p')
        outpath2 = os.path.join(dir, folder, 'out_0403.p')
        if os.path.isfile(outpath2):
            continue
        if os.path.isfile(outpath):
            pass
        else:
            queryForOneBinary(path2, outpath)
        results = test_one_binary(path, path2, outpath, libs)
        p.dump(results, open(outpath2, 'w'))
        print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    choice = int(sys.argv[1])
    if choice == 0:
        preprocessing_label()
    elif choice == 1:
        train()
    elif choice == 2:
        identity_key_functions()
    elif choice == 3:
        query()
    elif choice == 4:
        path2 = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test2/idafiles/3bafa6ff19784d18850f468b2813c5634a1f9b82de2c5db8f68e0b96bd57d7ea/nginx-{openssl-0.9.8t}{zlib-1.2.8}.ida.nam'
        queryPath = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test2/idafiles/3bafa6ff19784d18850f468b2813c5634a1f9b82de2c5db8f68e0b96bd57d7ea/test_kNN_0403.p'
        queryForOneBinary(path2, queryPath)
    elif choice == 5:
        analyse_naive()
    elif choice == 6:
        queryPath = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test2/idafiles/3bafa6ff19784d18850f468b2813c5634a1f9b82de2c5db8f68e0b96bd57d7ea/test_kNN_0403.p'
        analyse_labelcount(queryPath)
    elif choice == 7:
        path = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test2/idafiles/3bafa6ff19784d18850f468b2813c5634a1f9b82de2c5db8f68e0b96bd57d7ea/nginx-{openssl-0.9.8t}{zlib-1.2.8}.dot'
        path2 = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test2/idafiles/3bafa6ff19784d18850f468b2813c5634a1f9b82de2c5db8f68e0b96bd57d7ea/nginx-{openssl-0.9.8t}{zlib-1.2.8}.ida.nam'
        queryPath = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test2/idafiles/3bafa6ff19784d18850f468b2813c5634a1f9b82de2c5db8f68e0b96bd57d7ea/test_kNN_0403.p'
        folder = 'openssl-OpenSSL_0_9_8u'
        path11 = os.path.join('/home/yijiufly/Downloads/codesearch/data/openssl/', folder, 'libcrypto.so.dot')
        path12 = os.path.join('/home/yijiufly/Downloads/codesearch/data/openssl/', folder, 'libcrypto.so.ida.nam')
        lib = Library(path11, path12)
        lib.libraryName = folder.split('-')[1]+'_libcrypto.so'
        test_one_binary(path, path2, queryPath, [lib])
    elif choice == 8:
        test_some_binary()
