'''
decompose binary embedding
or generate similarity matrix
'''
import os
import subprocess
import shlex
import sys
import numpy as NP
import pickle as p
from pprint import pprint
from scipy import linalg as LA
#from embedding import *
#from embedding import Embedding
from multiprocessing import Pool
import traceback
import tables as tb
from scipy import spatial
from numpy.linalg import norm
import math
import json
import lshknn

trainingmFilenameList = []
EMBPATH     = "/rhome/lgao027/bigdata/binary/openssl/func_emb/"
#EMBPATH = "/rhome/lgao027/bigdata/out_analysis/functionembs/"
RAWBINPATH  = "/rhome/lgao027/bigdata/binary/alltestsamples/" ## add raw file path here
alpha = 80
posCounts        = 0
negCounts        = 0
posCounts_raw = 0
negCounts_raw = 0
#emb = Embedding()
newNameList = []


def loadFiles(PATH, ext = None): ## use .ida or .emb for ida file and embedding file
    filenames = []
    if   ext == ".ida":
        filenames = [f for f in os.listdir(PATH) if f.endswith(ext)]
    elif ext == ".emb":
        filenames = [f for f in os.listdir(PATH) if f.endswith(ext)]
    else:
        filenames = [f for f in os.listdir(PATH)]

    return filenames


def genIDA(bin, PATH, SCRIPTPATH): # extract ida script
    for i in range(len(bin)):
        cmd = '/home/yzheng04/ida-7.1/ida64 -c -A -S\"' + SCRIPTPATH + ' /home/yzheng04/Downloads/wannacry_ida/\" ' + PATH + bin[i]
        subprocess.call(shlex.split(cmd))

def printMatrix(filenameList, PATH): # inputlist with filenames, PATH for directory
    results = []
    for i, value1 in enumerate(filenameList):
        results_inner = []
        for j, value2 in enumerate(filenameList):
            PATH1 = PATH + value1
            PATH2 = PATH + value2
            hash1 = ssdeep.hash_from_file(PATH1)
            hash2 = ssdeep.hash_from_file(PATH2)
            result= ssdeep.compare(hash1, hash2)
            results_inner.append(result)
            # print("%3d"% (ssdeep.compare(hash1, hash2))), ## print ssdep compeare score
            print("%3d"% (ssdeep.compare(hash1, hash2))),
        print '\n'
        results.append(results_inner)
    return results


def compareEMB():
    ###load binary files
    BINPATH = WANNACRYPATH
    bin = loadFiles(BINPATH)
    # printMatrix(bin, BINPATH)

    ## extract IDA files from script
    SCRIPTFILE = PREPROCESSPATH
    genIDA(bin, BINPATH, SCRIPTFILE)

    ##load ida files
    IDAPATH = WANNACRYIDA
    ida = loadFiles(IDAPATH, ".ida")
    ##intialtize embedding
    emb = Embedding()
    for i, value in enumerate(ida):
        # print i
        idapath = IDAPATH + value
        ### generage embedding for all .emb files
        embedding = Embedding.embed_a_binary(emb, idapath)[1]
        embfile = idapath + ".emb"
        file = open(embfile ,'wb')
        p.dump(embedding, file)
        file.close

    ###load emb files
    EMBPATH = WANNACRYIDA
    emb = loadFiles(EMBPATH, ".emb")
    # print emb
    # ###compare emb with ssdeep
    # printMatrix(emb, EMBPATH)
    ###apply PCA on embedding
    PCAResults = genPCA(emb, EMBPATH)



def calcsim_cos(testsample):
    global newNameList
    global emb
    results_inner = []
    for i, value1 in enumerate(newNameList):
        try:
            PATH1 = EMBPATH + testsample
            PATH2 = EMBPATH + value1
        ## compare similarity score
            #emb = Embedding()
            f1 = open(PATH1, 'rb')
            f2 = open(PATH2, 'rb')
        ###load data
            file1 = p.load(f1)
            file2 = p.load(f2)
            result = NP.inner(file1, file2) / (norm(file1) * norm(file2))
            #result = 1 - spatial.distance.cosine(file1, file2)
            #print result
            results_inner.append(result)
            f1.close
            f2.close

        except Exception:
            print traceback.format_exc()

    return results_inner

def calcsim_tensorflow(testsample):
    global newNameList
    global emb
    results_inner = []
    for i, value1 in enumerate(newNameList):
        try:
            PATH1 = EMBPATH + testsample
            PATH2 = EMBPATH + value1
        ## compare similarity score
            #emb = Embedding()
            f1 = open(PATH1, 'rb')
            f2 = open(PATH2, 'rb')
        ###load data
            file1 = p.load(f1).tolist()
            file2 = p.load(f2).tolist()

            result = emb.test_similarity(file1, file2)

            results_inner.append(result)
            f1.close
            f2.close

        except Exception:
            print traceback.format_exc()

    return results_inner


def calc_tp(node_id):
    ### separate malicious and benign
    ### intilize variables
    mFilenameList = []
    bFilenameList = []
    for i, filenames in enumerate(newNameList):
        ## generate filenameList
        if len(filenames) == 72:
            mFilenameList.append(filenames) ## malicious list
        else:
            bFilenameList.append(filenames) ## benign list

    # print len(mFilenameList)
    # print len(bFilenameList)
    '''
    '''
    ### calculate truth prositive
    global trainingmFilenameList

    ### separate training/testing sample
    testingmFilenameList  = mFilenameList[:500] # first 500 malicious for test
    trainingmFilenameList = mFilenameList[500:] # 5381 for training

    testingbFilenameList  = bFilenameList[:500] # first 500 benign for test
    trainingbFilenameList = bFilenameList[500:] # 3001 for training

    # print len(testingbFilenameList)
    # print len(trainingbFilenameList)

    ### counters for positive and negative
    global posCounts
    global negCounts
    global posCounts_raw

    try:
        ### matching testing and training
        print(node_id)
        pool = Pool(processes=20)
        tofs = pool.map(calcsim_ssdeep, testingmFilenameList[node_id*20:(node_id+1)*20])

    except Exception:
        print traceback.format_exc()
    finally:
        print("final results:")
        print posCounts
        print posCounts_raw
        print posCounts/500*1.0
        print posCounts_raw/500*1.0

def calc_sample_similarity(testsample):
    global newNameList
    results_inner = []
    for i, value1 in enumerate(newNameList):
        PATH1 = EMBPATH + testsample
        PATH2 = EMBPATH + value1
        try:
            hash1 = ssdeep.hash_from_file(PATH1)
            hash2 = ssdeep.hash_from_file(PATH2)
            result= ssdeep.compare(hash1, hash2)
            #print result
            results_inner.append(result)

        except Exception:
            print traceback.format_exc()
            print PATH1
            print PATH2
    return results_inner

def calc_similarity_matrix(node_id):
    try:
        #print(node_id)
        #pool = Pool(processes=20)
        #quota = int(math.ceil(len(newNameList)/25))
        #print(len(newNameList))
        #print(quota)
        #res = pool.map(calcsim_cos, newNameList[node_id * quota : min((node_id + 1) * quota, len(newNameList))])

        size = int(math.ceil(len(newNameList)/20.0))
        #f = open(str(node_id) + ".txt", "w")
        file11 = []
        file22 = []
        row = node_id/20
        column = node_id%20
        for i in range(row * size, min((row + 1) * size, len(newNameList))):
            testsample = newNameList[i]
            PATH1 = EMBPATH + testsample
            f1 = open(PATH1, 'rb')
            file1 = p.load(f1).tolist()
            file11.append(file1)
            f1.close()

        for j in range(column * size, min(len(newNameList), (column + 1) * size)):
            value1 = newNameList[j]
            PATH2 = EMBPATH + value1
            f2 = open(PATH2, 'rb')
            file2 = p.load(f2).tolist()
            file22.append(file2)
            f2.close()

        result = emb.test_similarity(file11, file22)
        #p.dump(result[0],f)
        #f.write(str(result) + '\n')
        print len(result)
        print len(result[0])

        f = tb.open_file('versiondetect_funcsimilarity.h5', 'a')
        out = f.root.data
        out[row * size:min((row + 1) * size, len(newNameList)), column * size:min(len(newNameList), (column + 1) * size)] = result
        f.close()
    except Exception:
        print traceback.format_exc()


def decompose(embsample):
    PATH = "/rhome/lgao027/bigdata/binaryssl/" + embsample
    NAM_PATH = "/rhome/lgao027/bigdata/binaryssl/" + embsample[:-3] + 'nam'
    OUTPATH = "/rhome/lgao027/bigdata/binaryssl/funcemb_output/"
    f = open(PATH, "rb")
    funcs = p.load(f)
    nams = p.load(open(NAM_PATH, "rb"))
    for i in xrange(len(funcs)):
        OUTFILE = OUTPATH + embsample[8:-8] + "{" + nams[i] + "}.emb"
        #print OUTFILE
        file = open(OUTFILE,'wb')
        p.dump(funcs[i], file)
        file.close

def decomposebinary():
    try:
        pool = Pool(processes=10)
        res = pool.map(decompose, newNameList)
    except Exception:
        print traceback.format_exc()

# def lshkNN_preprocessing():
#     f = open("data/versiondetect_func_list.txt", "rb")
#     ## load data
#     global newNameList
#     newNameList = [line.rstrip() for line in f]
#     f.close()
#     ##generate signature
#     file11 = []
#     for i in range(0, len(newNameList)):
#         testsample = newNameList[i]
#         PATH1 = EMBPATH + testsample
#         f1 = open(PATH1, 'rb')
#         file1 = p.load(f1).tolist()
#         file11.append(file1)
#         f1.close()
#     data = NP.array(file11).T
#     for i in range(1,2):
#         c = lshknn.Lshknn(data=data, k=100, threshold=0.8, m=64, slice_length=0,signature=None, set_index= i, query=0)
#         c()


    #if already have signature
    #f = open("/rhome/lgao027/bigdata/emb/knn_final/0/random_plane/signature.p","r")
    #signature = p.load(f)
    #print signature.size
    #c = lshknn.Lshknn(data=None, k=100, threshold=0.8, m=64, slice_length=0, signature=signature,set_index=1,query=0)
    #c()
    #knn, similarity, n_neighbors = c()
    #with open('knn.p', 'wb') as fp:
    #    pickle.dump(knn, fp)
    #with open('n_neighbors.p', 'wb') as fp:
    #    pickle.dump(n_neighbors, fp)
    #with open('similarity.p', 'wb') as fp:
    #    pickle.dump(similarity, fp)

# def lshkNN_query():
#     f = open("data/versiondetect_query_list.txt", "rb")
#     queryList = [line.rstrip() for line in f]
#     f.close()
#     #list = [[] for y in range(len(queryList))]
#     file11 = []
#     for i in range(0, len(queryList)):
#         testsample = queryList[i]
#         PATH1 = "/rhome/lgao027/bigdata/binaryssl/func_emb/" + testsample
#         f1 = open(PATH1, 'rb')
#         file1 = p.load(f1).tolist()
#         file11.append(file1)
#         f1.close()
#     data = NP.array(file11).T
#     for i in range(1):
#         c = lshknn.Lshknn(data=data, k=100, threshold=0.8, m=64, slice_length=0, signature=None, set_index= i, query = len(queryList))
#         c()
        #signature = c()
        #for idx, sig in enumerate(signature[0]):
        #    f = open("knn_final/buckets/" + str(sig) + ".txt","r")
        #    list[idx] = list[idx].extend([line.rstrip() for line in f])
        #    f.close()

    #f = open("data/knn_candidate.p","w")
    #p.dump(list, f)
    #f.close()

    #for idx, item in enumerate(list):
    #    filename1 = queryList[idx]
    #    emb1 = p.load(open("/rhome/lgao027/bigdata/binaryssl/func_emb/" + filename1, "r"))
    #    for file2idx in item:
    #        filename2 = newNameList[file2idx]
    #        emb2 = p.load(open(EMBPATH + filename2 , "r"))
    #        result = NP.inner(emb1, emb2) / (norm(emb1) * norm(emb2))




def main():
    #get node id
    #node_id = int(sys.argv[1])
    ###########################################
    # decompose embedding of binary to embedding of functions
    ###########################################
    # ### load embedding filenames listdir
    #embFilenames    = loadFiles("/rhome/lgao027/bigdata/binary/openssl/openssl_emb_exclude_smallfuncs/", ".emb")

    embFilenames    = loadFiles("/home/yijiufly/Downloads/codesearch/scripts/out/", ".emb")
    global newNameList
    newNameList = embFilenames

    decomposebinary()



    ###########################################
    # calculate similarity matrix
    ###########################################
    ### load saved namelists
    #f = open("subfunc.txt", 'rb')
    #f = open("data/versiondetect_func_list.txt", "rb")
    ## load data
    #global newNameList
    #newNameList = [line.rstrip() for line in f]
    #f.close()
    #print len(newNameList)

    #calc_tp(node_id)
    #for i in range(node_id * 20, (node_id + 1) * 20)
        #calc_similarity_matrix(i)

    #lshkNN_preprocessing()
    #lshkNN_query()
    ## add false positive, true negative




if __name__ == '__main__':
	main()
