'''
ssdeap test for
binary vs embedding
'''
import os
import subprocess
import ssdeep
import shlex
import sys
import numpy as NP
import pickle as p
from pprint import pprint
from scipy import linalg as LA
from embedding import Embedding

### define all PATH and test for filenames
MALWARE1        = "0fb900a14c943976cfc9029a3a710333f5434bea9e3cccacd0778bd25ccfa445"
MALWARE2        = "5f18adb787561f2f2d82dcd334dc77a9cac1982b33c52bd4abf977c1d94b5dd8"
MALWARE3        = "86e0eac8c5ce70c4b839ef18af5231b5f92e292b81e440193cdbdc7ed108049f"
WANNACRYPATH    = "/home/yzheng04/Downloads/wannacry/"
WANNACRYIDA     = "/home/yzheng04/Downloads/wannacry_ida/"
LUCYLOCKERPATH  = "/home/yzheng04/Downloads/lucylocker/"
PREPROCESSPATH  = "/home/yzheng04/Workspace/Deepbitstech/Allinone/rawfeatureextractor/extractor/preprocessing_ida.py"

PATHNAME        = "/home/yzheng04/Workspace/Deepbitstech/Allinone/embedding/"
EMBFILENAME     = "emb"
BINFILENAME     = "test/binary/openssl.x86_64_O1"
EMB01           = PATHNAME + BINFILENAME

SAMPLENAME      = "sample.exe"
TESTNAME        = "0.001_0"

EMBSAMPLENAME   = "emb_malware_1"
EMBTESTNAME     = "emb_malware_2"

EMBSAMPLENAME   = WANNACRYPATH + MALWARE1
EMBTESTNAME     = LUCYLOCKERPATH + MALWARE3

def genPCA(filenameList, PATH, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    results = []
    rmlist = []
    count1 = 0
    count2 = 0
    for i, value in enumerate(filenameList):
        # print i
        # print value
        ## load filename
        input = PATH + value
        f = open(input, 'rb')
        ## load data
        data = p.load(f)
        f.close
        ## convert to np array
        data = NP.array(data)

        ## remove shelled data
        try:
            m, n = data.shape
        except:
            count1 = count1 +1
            # print i, count1
            # print value
            rmlist.append(i)
            continue


        # m, n = data.shape
        ## check shape
        # print m, n
        # mean center the data
        data -= data.mean(axis=0)
        # calculate the covariance matrix
        R = NP.cov(data, rowvar=False)
        # calculate eigenvectors & eigenvalues of the covariance matrix
        # use 'eigh' rather than 'eig' since R is symmetric,count1 = count1 +1
        # the performance gain is substantial
        try:
            evals, evecs = LA.eigh(R)
        except:
            count2 = count2 + 1
            # print i, count2
            # print value
            rmlist.append(i)
            continue
        # sort eigenvalue in decreasing order
        idx = NP.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        # sort eigenvectors according to same index
        evals = evals[idx]
        # select the first n eigenvectors (n is desired dimension
        # of rescaled data array, or dims_rescaled_data)
        evecs = evecs[:, :dims_rescaled_data]
        # carry out the transformation on the data using eigenvectors
        # and return the re-scaled data, eigenvalues, and eigenvectors
        ###output resultsEMBPATH = "/home/yzheng04/Downloads/embs/"
        result = NP.dot(evecs.T, data.T).T, evals, evecs
        emb    = NP.dot(evecs.T, data.T).T
        results.append(emb)
        # print i, results
    print count1, count2
    # return results
    return rmlist, results

# def loadBinFiles(PATH): ### merged into loadFiles
#     filenames = []
#     filenames = [f for f in os.listdir(PATH)]
#     # for filename in os.listdir(WAN#NACRYPATH):
#     #     # print(os.path.join(WANNACRYPATH, filename))
#     #
#     #     binName = os.path.join(WANNACRYPATH, filename)
#     #     filenames.append(binName)
#     # # print filenames[0]
#     return filenames

def loadFiles(PATH, ext = None): ## use .ida or .emb for ida file and embedding file
    filenames = []
    if   ext == ".ida":
        filenames = [f for f in os.listdir(PATH) if f.endswith(ext)]
    elif ext == ".emb":
        filenames = [f for f in os.listdir(PATH) if f.endswith(ext)]
    else:
        filenames = [f for f in os.listdir(PATH)]

    return filenames

# def compareRawFiles(filenames):
#     n = len(filenames)filenameList, PATH
#     results = [[0] * n for i in range(n)]
#     filenames = loadBinFiles()
#     for i in range(len(filenames)):
#         for j in range(len(filenames)):
#             sampleHash = ssdeep.hash_from_file(filenames[i])
#             testHash   = ssdeep.hash_from_file(filenames[j])
#             results[i][j] = ssdeep.compare(sampleHash, testHash)
#             # results.append(result)
#             # print "======="
#             # print result
#     return results

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

def main():
    ### define PATH
    EMBPATH     = "/home/yzheng04/Downloads/embs/"
    RAWBINPATH  = "/home/yzheng04/Downloads/rawbinfile" ## add raw file path here
    alpha = 80
    ### intilize variables
    mFilenameList = []
    bFilenameList = []

    # ### load embedding filenames listdir
    # embFilenames    = loadFiles(EMBPATH, ".emb")

    ### load saved namelists
    f = open("newNameList.p", 'rb')
    ## load data
    newNameList = p.load(f)

    # print len(newNameList)

    ### separate malicious and benign
    for i, filenames in enumerate(newNameList):
        ## generate filenameList
        if len(filenames) == 72:
            mFilenameList.append(filenames) ## malicious list
        else:
            bFilenameList.append(filenames) ## benign list

    # print len(mFilenameList)
    # print len(bFilenameList)

    ### separate training/testing sample
    testingmFilenameList  = mFilenameList[:500] # first 500 malicious for test
    trainingmFilenameList = mFilenameList[500:] # 5381 for training

    testingbFilenameList  = bFilenameList[:500] # first 500 benign for test
    trainingbFilenameList = bFilenameList[500:] # 3001 for training

    # print len(testingbFilenameList)
    # print len(trainingbFilenameList)

    ### counters for positive and negative
    posCounts        = 0
    negCounts        = 0
    posCounts_ssdeep = 0
    negCounts_ssdeep = 0

    ### matching testing and training
    for testmsample in testingmFilenameList: ## true positive
        for trainmsample in trainingmFilenameList:
            ## combine file full name
            PATH1 = EMBPATH + testmsample
            PATH2 = EMBPATH + trainmsample
            ## get ssdeep hash value
            hash1 = ssdeep.hash_from_file(PATH1)
            hash2 = ssdeep.hash_from_file(PATH2)
            ## compare similarity score
            result= ssdeep.compare(hash1, hash2)

            PATH11= RAWBINPATH + testmsample[:-7]
            PATH22= RAWBINPATH + trainmsample[:-7]
            ## get ssdeep hash value
            hash11= ssdeep.hash_from_file(PATH11)
            hash22= ssdeep.hash_from_file(PATH22)
            ## compare similarity score
            result1= ssdeep.compare(hash11, hash22)

            if result >= alpha: ## true positive by embedding
                posCounts = posCounts + 1
                break
            if result1>= alpha: ## true positive by ssdeep
                posCounts_ssdeep = posCounts_ssdeep + 1
                break

    print posCounts/500*1.0
    print posCounts_ssdeep/500*1.0

    ## add false positive, true negative    






    # ###load binary files
    # BINPATH = WANNACRYPATH
    # bin = loadFiles(BINPATH)
    # # printMatrix(bin, BINPATH)
    #
    # ## extract IDA files from script
    # SCRIPTFILE = PREPROCESSPATH
    # genIDA(bin, BINPATH, SCRIPTFILE)
    #
    # ##load ida files
    # IDAPATH = WANNACRYIDA
    # ida = loadFiles(IDAPATH, ".ida")
    # ##intialtize embedding
    # emb = Embedding()
    # for i, value in enumerate(ida):
    #     # print i
    #     idapath = IDAPATH + value
    #     ### generage embedding for all .emb files
    #     embedding = Embedding.embed_a_binary(emb, idapath)[1]
    #     embfile = idapath + ".emb"
    #     file = open(embfile ,'wb')
    #     p.dump(embedding, file)
    #     file.close
    #
    # ###load emb files
    # EMBPATH = WANNACRYIDA
    # emb = loadFiles(EMBPATH, ".emb")
    # # print emb
    # # ###compare emb with ssdeep
    # # printMatrix(emb, EMBPATH)
    # ###apply PCA on embedding
    # PCAResults = genPCA(emb, EMBPATH)

if __name__ == '__main__':
	main()
# ### test for openssl_O0-O3
# for i in range( 4 ):
#     files_dict.append(ssdeep.hash_from_file( PATHNAME + BINFILENAME + str(i) ))
#     emb_dict.append(ssdeep.hash_from_file( PATHNAME + EMBFILENAME + str(i) ))
# print ssdeep.compare(files_dict[0], emb_dict[0])
# # print emb_dict
#
# for i in range(len(emb_dict)):
#     for j in range(len(emb_dict)):
#         result = ssdeep.compare(files_dict[i], emb_dict[j])
#         print result,
#     print "\n"

# ### test for sample.exe vs flip sample.exe
# NEWSAMPLEPATH   = "/home/yzheng04/Workspace/Deepbitstech/Allinone/malwareintelligence/test/sample.exe"
# # NEWTESTPATH     = "/home/yzheng04/Workspace/Deepbitstech/Allinone/embedding/0.001_0"
# NEWTESTPATH     = "/home/yzheng04/Workspace/Deepbitstech/Allinone/malwareintelligence/test/bin/0.001_3"
# NEWTESTPATH1     = "/home/yzheng04/Workspace/Deepbitstech/Allinone/malwareintelligence/test/bin/0.001_2"
#
# sampleHash = ssdeep.hash_from_file( NEWSAMPLEPATH )
# testHash   = ssdeep.hash_from_file( NEWTESTPATH )
# testHash1   = ssdeep.hash_from_file( NEWTESTPATH1 )
#
# # print sampleHash
# # print testHash
#
# result = ssdeep.compare(sampleHash, testHash)
# result1 = ssdeep.compare(testHash1, testHash)
#
# print result, result1
