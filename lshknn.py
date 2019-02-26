from db import db
import numpy as np
import pickle as p
configname = "openssl-halftest"
def loadData(path):
    f = open(path, "rb")
    ## load data
    newNameList = [line.rstrip() for line in f]
    f.close()
    ##generate signature
    file11 = []
    for i in range(0, len(newNameList)):
        testsample = newNameList[i]
        PATH1 = "data/versiondetect/func_emb_withname/" + testsample
        f1 = open(PATH1, 'rb')
        file1 = p.load(f1).tolist()
        file11.append(file1)
        f1.close()
    data = np.array(file11)
    return data, newNameList

def loadQueryData(path):
    f = open(path, "rb")
    ## load data
    newNameList = [line.rstrip() for line in f]
    f.close()
    ##generate signature
    file11 = []
    for i in range(0, len(newNameList)):
        testsample = newNameList[i]
        PATH1 = "data/versiondetect/func_emb_nginx/" + testsample
        f1 = open(PATH1, 'rb')
        file1 = p.load(f1).tolist()
        file11.append(file1)
        f1.close()
    data = np.array(file11)
    return data, newNameList

def getHashMap():
    db_instance = db()
    db_instance.loadHashMap(configname)
    #db_instance.engine.clean_all_buckets()

    return db_instance

def addToHashMap(db_instance, data, funcList):
    db_instance.indexing(data, funcList)

    return db_instance

def doSearch(db_instance, query, name):
    return db_instance.querying(query, name)

def output_format(queryList, testkNN):
    f = open(queryList, "r")
    list = [line.rstrip() for line in f]
    querydata = []
    for idx, testcases in enumerate(testkNN):
        for testcase in testcases:
            id = list.index(testcase[1])
            #print str(idx + len(list)), str(id), str(1-testcase[2])
            querydata.append([idx + len(list), id, 1-testcase[2]])
    return querydata

if __name__ == '__main__':
    hashMap = getHashMap()
    #data, newNameList = loadData("data/versiondetect/test1/versiondetect_func_sublist.txt")
    #print "Start add data to hash map"
    #hashMap = addToHashMap(hashMap, data, newNameList)
    add_data, newNameList = loadQueryData("data/versiondetect/test1/versiondetect_addfunc_list.txt")
    hashMap = addToHashMap(hashMap, add_data, newNameList)
    #hashMap = addToHashMap(hashMap, [])
    qdata, newNameList = loadQueryData("data/versiondetect/test1/versiondetect_query_list.txt")
    print "Start query for test data"
    testkNN = doSearch(hashMap, qdata, newNameList)
    #for knn in testkNN:
    #    print knn
    #hashMap = addToHashMap(hashMap, [])
    #testkNN = p.load(open("data/versiondetect/kNN.p","r"))
    #N = output_format("data/versiondetect/versiondetect_func_list.txt", testkNN)
    p.dump(testkNN, open("data/versiondetect/test1/test_kNN.p","w"))
