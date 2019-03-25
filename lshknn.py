from db import db
import numpy as np
import pickle as p
configname = "openssl-0306"
def _get_binary_hash(file_path):
	fd = open(file_path, 'rb')
	md5 = hashlib.sha256()
	data = fd.read()
	md5.update(data)
	return md5.hexdigest()
def loadData(path):
    f = open(path, "rb")
    ## load data
    newNameList = [line.rstrip() for line in f]
    f.close()
    ##generate signature
    file11 = []
    for i in range(0, len(newNameList)):
        testsample = newNameList[i]
        PATH1 = "data/versiondetect/test2/funcemb_output_openssl" + testsample
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
    #data, newNameList = loadData("data/versiondetect/test2/versiondetect_func_list.txt")
    #print "Start add data to hash map"
    #hashMap = addToHashMap(hashMap, data, newNameList)
    #add_data, newNameList = loadQueryData("data/versiondetect/test1/versiondetect_addfunc_list.txt")
    #hashMap = addToHashMap(hashMap, add_data, newNameList)
    #hashMap = addToHashMap(hashMap, [])
    #qdata, newNameList = loadQueryData("data/versiondetect/test1/versiondetect_query_list.txt")
    #funcnamepath = os.path.join('data/versiondetect/test2/idafiles/0acc5283147612b2abd11d606d5585ac8370fc33567f7f77c0b312c207af3bf9', 'nginx-{openssl-0.9.8r}{zlib-1.2.9}.ida.nam')
    funcnamepath = 'data/versiondetect/test2/idafiles/0acc5283147612b2abd11d606d5585ac8370fc33567f7f77c0b312c207af3bf9/nginx-{openssl-0.9.8r}{zlib-1.2.9}.ida.nam'
    qdata, newNameList = loadOneQueryBinary(funcnamepath, 'data/versiondetect/test2/funcemb_output_testing/')
    #print qdata, newNameList
    #print "Start query for test data"
    testkNN = doSearch(hashMap, qdata, newNameList)
    #for knn in testkNN:
    #    print knn
    #hashMap = addToHashMap(hashMap, [])
    #testkNN = p.load(open("data/versiondetect/kNN.p","r"))
    #N = output_format("data/versiondetect/versiondetect_func_list.txt", testkNN)
    p.dump(testkNN, open("data/versiondetect/test2/test_kNN.p","w"))
