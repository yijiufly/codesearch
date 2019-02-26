from sslh.test_SSLH_inference import *
from sslh.analysis import *
import numpy as np
from labels import Labels
import cPickle as p
import sys
import operator
import pdb


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
def choose_threshold():
    queryPath = 'data/versiondetect/test1/test_kNN.p'
    #query stores the query knn: each line is [(query_idx, query_func_name), (func_idx, func_name), similarity]
    query = p.load(open(queryPath, 'r'))

    #temporarily stores the similarity distribution of candidiate similar functions
    funcs = dict()
    #for each function, choose similar functions according to the knn
    chosenFuncs = []

    #stores the name of "each function"
    funcList = []

    lastFunc = ''
    for i in query:
        if i[0][1] != lastFunc and lastFunc != '':
            keylist = funcs.keys()
            #sort according to keys
            keylist.sort()
            if len(funcs[keylist[-1]]) <= 67:
                #append the functions with highest similarity score
                chosenFuncs.append(funcs[keylist[-1]])
            else:
                #ignore the short functions which is simiar to too many functions
                chosenFuncs.append([])
            funcList.append(lastFunc)
            print lastFunc
            print chosenFuncs[-1]
            print '\n'
            funcs = dict()

        key = round(i[2], 8)
        #pdb.set_trace()
        if key in funcs:
            funcs[key].append(i[1][1])
        else:
            funcs[key] = [i[1][1]]

        lastFunc = i[0][1]

    keylist = funcs.keys()
    keylist.sort()
    if len(funcs[keylist[-1]]) <= 67:
        chosenFuncs.append(funcs[keylist[-1]])
    else:
        chosenFuncs.append([])
    funcList.append(lastFunc)
    return chosenFuncs, funcList

## detect the version simply according to the vote for each version
def analyse_labelcount():
    chosenFuncs, funcList = choose_threshold()
    alllabels = p.load(
        open("data/versiondetect/test1/alllabels_withnginx.p", "r"))

    label = Labels(funcNameList=funcList, alllabels=alllabels)
    label.analyse_labelcount(chosenFuncs)



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
        analyse()
    elif choice == 5:
        analyse_naive()
    elif choice == 6:
        analyse_labelcount()
