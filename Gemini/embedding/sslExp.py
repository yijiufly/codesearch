import os
import sys
import subprocess
import shlex
import pickle as p
import ssdeep
import numpy as np
import scipy
import collections

from embedding import Embedding

### getting all file list in a directory
def get_files(filepath):
    dir_list  = []
    file_list = []
    for root, dirc, files in os.walk(filepath):
        for file in files:
            file_list.append(os.path.join(root, file))
            dir_list.append(root)
    return dir_list, file_list

def genIDA(outpath, binary_full_path): # extract ida script
    for i in range(len(binary_full_path)):
        #cmd = 'QT_X11_NO_MITSHM=1 ./ida/idaq -A -S\"' + cwd + '/raw-feature-extractor/preprocessing_ida.py ' +  + '\" ' + binary_full_path
        # cmd = './ida/idal64 -c -A -S\"' + cwd + '/raw_feature_extractor/preprocessing_ida.py ' + outpath + '\" ' + binary_full_path
        # cmd = '/home/yzheng04/ida-7.1/' + 'idat64 -c -A -S\"' + '/home/yzheng04/Workspace/Deepbitstech/Allinone/rawfeatureextractor' + '/extractor/gen_cfg.py ' + outpath + '\" ' + binary_full_path
        cmd = '/home/yzheng04/ida-7.1/ida64 -c -A -S\"' + '/home/yzheng04/Workspace/Deepbitstech/Allinone/rawfeatureextractor/extractor/preprocessing_ida.py ' + str(outpath[i]) + '\" ' + str(binary_full_path[i])
        print("cmd",cmd)

        # subprocess.call(shlex.split(cmd))
        # cmd = '/home/yzheng04/ida-7.1/ida64 -c -A -S\"' + SCRIPTPATH + ' /home/yzheng04/Downloads/wannacry_ida/\" ' + PATH + bin[i]
        subprocess.call(shlex.split(cmd))
        # if i == 1:
        #     break

def genEMB(ida_full_path_list):
    ##intialtize embedding
    emb = Embedding()
    for i, value in enumerate(ida_full_path_list):
        print value
        # idapath = value + ".ida"
        idapath = value
        ### generage embedding for all .emb files
        embs = Embedding.embed_a_binary(emb, idapath)[1]
        embfile   = value + ".emb"
        # funcfile  = value  + ".func"
        f1 = open(embfile ,'wb')
        # f2 = open(funcfile,'wb')
        p.dump(embs,f1)
        genEMB# p.dump(func,f2)
        f1.close
        # f2.close
        # if i == 0:
        #     break

def loadFiles(PATH, ext = None): ## use .ida or .emb for ida file and embedding file
    filenames = []
    if   ext == ".ida":
        filenames = [f for f in os.listdir(PATH) if f.endswith(ext)]
    elif ext == ".emb":
        filenames = [f for f in os.listdir(PATH) if f.endswith(ext)]
    else:
        filenames = [f for f in os.listdir(PATH)]

    return filenames

def printMatrix(filenameList): # inputlist with filenames, PATH for directory
    results = []
    for i, value1 in enumerate(filenameList):
        results_inner = []
        for j, value2 in enumerate(filenameList):
            hash1 = ssdeep.hash_from_file(value1)
            hash2 = ssdeep.hash_from_file(value2)
            result= ssdeep.compare(hash1, hash2)
            results_inner.append(result)
            # print("%3d"% (ssdeep.compare(hash1, hash2))), ## print ssdep compeare score
            print("%3d"% (ssdeep.compare(hash1, hash2))),
        print '\n'
        results.append(results_inner)
    return results

def printNewMatrix(filenameList): # inputlist with filenames, PATH for directory
    results = []
    for i, value1 in enumerate(filenameList):
        results_inner = []
        for j, value2 in enumerate(filenameList):
            # hash1 = ssdeep.hash_from_file(value1)
            # hash2 = ssdeep.hash_from_file(value2)
            result= ssdeep.compare(value1, value2)
            results_inner.append(result)
            # print("%3d"% (ssdeep.compare(hash1, hash2))), ## print ssdep compeare score
            print("%3d"% (ssdeep.compare(value1, value2))),
        print '\n'
        results.append(results_inner)
    return results

def gen_pca(emb, dims_rescaled_data=100):
	emb = np.array(emb).T
	emb -= emb.mean(axis=0)
	r = np.cov(emb, rowvar=False)
	evals, evecs = scipy.linalg.eigh(r)
	idx = np.argsort(evals)[::-1]
	evecs = evecs[:, idx]
	evecs = evecs[:, :dims_rescaled_data]
	return np.dot(emb, evecs).T.reshape(128)

def loadResult(name):
    file = open(name, 'rb')
    # load data
    result = p.load(file)
    # print "malicious", len(mFilenameList)
    file.close
    return result

def saveResult(name, var):
    file = open(name, 'wb')
    p.dump(var, file)
    file.close

def intersectionOfMats(matA,matB):
    nrows, ncols = matA.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],'formats':ncols * [matA.dtype]}

    C = np.intersect1d(matA.view(dtype), matB.view(dtype))

    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(matA.dtype).reshape(-1, ncols)
    return C

def setDiff(a1,a2): ## set difference, in a1 & not in a2
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
    result = np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])
    return result

def intersectWithThr(a1,a2,th): ## set difference with threshold, in a1 & not in a2
    nrows, ncols = a1.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],'formats':ncols * [a1.dtype]}
    C = np.intersect1d(a1.view(dtype), a2.view(dtype))

    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(a1.dtype).reshape(-1, ncols)
    return C

def main():

    ### define path
    DIRPATH = "/media/yzheng04/DATA/binary/binary/"
    dirList, fileList = get_files(DIRPATH)
    newdirList = list(set(dirList))# ssllistPCA
    TESTPATH = "/media/yzheng04/DATA/binary/"
    TESTFILE = ["110b.o","110c.o","110h.o"]
    testfilelist = []
    testpathlist = []
    for name in TESTFILE:
        testfilelist.append(TESTPATH+name)
        testpathlist.append(TESTPATH)

    # ### ida files for test samples
    # genIDA(testpathlist,testfilelist)

    ## generate emb for test samples
    idatestlist = []
    testfilename = loadFiles(TESTPATH,".ida")
    for name in testfilename:
        idatestlist.append(TESTPATH+name)
    print idatestlist
    genEMB(idatestlist) # uncomment when need to generate

    print "done"

    '''
    ### load emb files
    testemblist = []
    testembfiles = loadFiles(TESTPATH,".emb")
    # print testembfiles
    for name in testembfiles:
        testemblist.append(TESTPATH + name)
    # print testemblist




    # print len(newdirList)
    # print dirList, fileList
    # print len(dirList)
    # print len(fileList)

    ### generate IDA files
    # genIDA(dirList, fileList)

    ### extract func EMB from IDA files
    idafullpathlist = []
    for folderpath in newdirList:
        fileLists = loadFiles(folderpath,".ida")
        # print folderpath, fileLists
        # print len(dirList)
        # print len(fileLists)
        for filename in fileLists:
            idafullpath = folderpath + '/' + filename
            idafullpathlist.append(idafullpath)
    # print len(idafullpathlist)
    ### generate emb
    # genEMB(idafullpathlist) # uncomment when need to generate
    ### compare similiarity
    ssllist = idafullpathlist[1::2]
    cryptolist = idafullpathlist[0::2]
    # print len(cryptolist)
    # print len(ssllist)
    sslemblist = [s + '.emb' for s in ssllist]
    cryptoemblist = [t + '.emb' for t in cryptolist]
    sslpcaemblist = [s + '.pca.emb' for s in ssllist]
    cryptopcaemblist = [t + '.pca.emb' for t in cryptolist]

    EMB = Embedding()
    # print sslemblist
    ## resuults by ssdeep on raw files
    # sslresult    = printMatrix(sslemblist)
    # cryptoresult = printMatrix(cryptoemblist)

    ### get PCA on lists for embeddings & compare with test

    sslPCAlist = []
    for t_idx, t_emb in enumerate(testemblist):
        t_embresult = np.array(loadResult(t_emb))
        t_tmp = Embedding.gen_pca(EMB, t_embresult)
        t_pcaResult = np.expand_dims(t_tmp, axis=0)
        for idx, emb in enumerate(sslemblist):
            # print len(sslemblist)
            embresult = np.array(loadResult(emb))
            tmp = Embedding.gen_pca(EMB, embresult)
            pcaResult = np.expand_dims(tmp, axis=0)
            # print pcaResult.shape
            ## test similiarity between test samples and library
            sim_result = Embedding.test_similarity(EMB,t_pcaResult,pcaResult)
            # print sim_result
            if idx == 100:
                print "110b"
            if idx == 101:
                print "110c"
            if idx == 106:
                print "110h"
        # print t_emb


    ### stack all version library into a matrix
    # matVersionDB = []

    for idx, emb in enumerate(sslemblist):
        # print len(sslemblist)
        # print emb
        # if idx == 106:
        #     tmp = np.array(loadResult(emb))
        #     print tmp.shape
        if idx == 0:
            tmp = np.array(loadResult(emb))
            matVersionDB = tmp
        else:
            tmp = np.array(loadResult(emb))
            matVersionDB = np.vstack((matVersionDB,tmp))
    print matVersionDB.shape

        # embresult = np.array(loadResult(emb))
        # tmp = Embedding.gen_pca(EMB, embresult)

    ### intersection of all version DB
    for idx, emb in enumerate(sslemblist):
        if idx == 0:
            matIntersection = np.array(loadResult(emb))
        else:
            tmp = np.array(loadResult(emb))
            matIntersection = intersectionOfMats(tmp,matIntersection)
    print "intersection size:", matIntersection.shape

    ### save the diff mat for each version
    for idx, emb in enumerate(sslemblist):
        tmpsample = np.array(loadResult(emb))
        tmpdiff = setDiff(tmpsample,matIntersection)
        print "=================================="
        print "tmp sample size:", tmpsample.shape
        print "diff mat size:", tmpdiff.shape
        print "=================================="

    ### use diff version detect

    ### rank by occurance
    d = collections.OrderedDict()
    for a in matVersionDB:
        t = tuple(a)
        if t in d:
            d[t] += 1
        else:
            d[t] = 1

    tmpresult = []
    for (key, value) in d.items():
        tmpresult.append(list(key) + [value])

    B = np.asarray(tmpresult)
    print B.shape
    print min(B[:,64])
    print max(B[:,64])

    ### compare test sample vs diff version db



    # sslPCAlist = loadResult('sslPCAlist.p')
    # print len(sslPCAlist)
    # print len(sslPCAlist[0])
    # for idx, emb in enumerate(sslPCAlist):
    #     saveResult(sslpcaemblist[idx],emb)

    # sslresult  = printMatrix(sslpcaemblist)


    ### save PCA results
    # f3 = open("sslPCA.p" ,'wb')
    # p.dump(ssllistPCA, f3)
    # f3.close

    ### load pca results
    # ssllistPCA= loadResult("sslPCA.p")
    # print ssllistPCA

    # sslresult  = printNewMatrix(ssllistPCA)



    # ssllistPCA = genpCA(ssllist)
    # cryptolistPCA = genpCA(cryptolist)
    # f3 = open("sslPCA.p" ,'wb')
    # p.dump(ssllistPCA, f3)
    # f3.close
    # f4 = open("cryptoPCA.p" ,'wb')
    # p.dump(cryptolistPCA, f4)
    # f4.close


    '''





if __name__ == '__main__':
    main()
