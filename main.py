import numpy as np
from labels import Labels
import cPickle as p
import sys
import operator
import pdb
from binary import TestBinary
from library import Library
from lshknn import queryForOneBinary3Gram, queryForOneBinary2Gram
import sys
sys.path.append('./SPTAG/Release')
#from searchSPTAG import queryForOneBinary3Gram
import os
import time
import traceback
import multiprocessing
import factorgraph as fg

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

def test_one_binary(binaryName, dotPath, funcembFolder, queryPath, namPath, libs=None):
    #path = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test2/idafiles/0acc5283147612b2abd11d606d5585ac8370fc33567f7f77c0b312c207af3bf9/nginx-{openssl-0.9.8r}{zlib-1.2.9}.dot'
    #path2 = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test2/idafiles/0acc5283147612b2abd11d606d5585ac8370fc33567f7f77c0b312c207af3bf9/nginx-{openssl-0.9.8r}{zlib-1.2.9}.ida.nam'
    testbin = TestBinary(binaryName, dotPath, funcembFolder)
    testbin.buildNGram(namPath)
    if os.path.isfile(queryPath):
        pass
    else:
        queryForOneBinary(testbin.threeGramList, queryPath)
    testbin.count(queryPath)

def load_libs():
    libs=[]
    path_zlib = '/home/yijiufly/Downloads/codesearch/data/zlib/zlib-O2'
    for folder in os.listdir(path_zlib):
        dotfile = loadFiles(os.path.join(path_zlib, folder), ext='.dot')[0]
        namfile = loadFiles(os.path.join(path_zlib, folder), ext='.nam')[0]
        path11 = os.path.join(path_zlib, folder, dotfile)
        path12 = os.path.join(path_zlib, folder, namfile)
        lib = Library(path11, path12)
        lib.libraryName = dotfile.rsplit('.',1)[0]+'O2'
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

        path11 = os.path.join(path_openssl, folder, 'libssl.so.dot')
        path12 = os.path.join(path_openssl, folder, 'libssl.so.ida.nam')
        lib = Library(path11, path12)
        lib.libraryName = folder.split('-')[1]+'_libssl.so'
        libs.append(lib)
        print 'load ' + lib.libraryName
    return libs

def test3gram():
    dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx'
    folders = os.listdir(dir)
    try:
        pool = Pool(processes=1)
        pool.map(test_some_binary_ngram, folders)
    except:
        print(traceback.format_exc())
        pass
    pool.close()
    pool.join()

def parallelQuery():
    dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx'
    #funcembFolder = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/funcemb_testing'
    folders = os.listdir(dir)
    folders.sort()
    l = len(folders)
    cores = 4#multiprocessing.cpu_count()
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
    dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx'
    #funcembFolder = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/funcemb_testing'
    folders = os.listdir(dir)
    folders.sort()
    print "PID:", os.getpid()
    print i,j
    for folder in folders[i:j]:
        print(folder)
        start_time = time.time()
        try:
            dotfiles = loadFiles(os.path.join(dir, folder), ext='_bn.dot')
            #dotfiles.sort()
            namfiles = loadFiles(os.path.join(dir, folder), ext='.nam')
            embfiles = loadFiles(os.path.join(dir, folder), ext='.emb')
        except Exception:
            print traceback.format_exc()
            continue
        #pdb.set_trace()
        for i in range(len(dotfiles)):

            dotfile = dotfiles[i]
            namfile = namfiles[i]
            embfile = embfiles[i]
            binaryName = dotfile.split('.')[0]
            dotPath = os.path.join(dir, folder, dotfile)
            namPath = os.path.join(dir, folder, namfile)
            embPath = os.path.join(dir, folder, embfile)

            #calculate 3gram
            # candidate_output3 = os.path.join(dir, folder, binaryName+'_test_kNN_0521_3gram.p')
            # count_output3 = os.path.join(dir, folder, binaryName+'_BP_undirected.txt')
            binary_dir = os.path.join(dir, folder)
            candidate_output = os.path.join(dir, folder, 'test_kNN_factorgraph_3gram.p')
            count_output = os.path.join(dir, folder, 'out_factorgraph0620_3gram.p')
            #if os.path.isfile(count_output):
                #result3gram = p.load(open(count_output3, 'rb'))
                #continue

            testbin = TestBinary(binaryName, dotPath, embPath)

            if os.path.isfile(candidate_output):
                pass
            else:
                testbin.buildNGram(namPath)
                queryForOneBinary3Gram(testbin.threeGramList, candidate_output)
                #queryForOneBinary2Gram(testbin.twoGramList, candidate_output)

            #testbin.callBP(candidate_output3, candidate_output2, namPath, binary_dir)
            result = runFactorGraph3(candidate_output)
            #result = testbin.count(candidate_output)
            p.dump(result, open(count_output, 'w'))
        print("--- %s seconds ---" % (time.time() - start_time))
'''
        #2gram
        if os.path.isfile(count_output2):
            result2gram = p.load(open(count_output2, 'rb'))
        else:
            if os.path.isfile(candidate_output2):
                pass
            else:
                queryForOneBinary2Gram(testbin.twoGramList, candidate_output2)

            result2gram = testbin.count(candidate_output2)
            p.dump(result2gram, open(count_output2, 'w'))
'''

def runFactorGraph(queryPath_3gram, threshold=0.999):
    potential1 = np.array([
            [0.99, 0.01],
            [0.01, 0.99],
    ])
    # Make an empty graph
    g = fg.Graph()
    global_dict = dict()
    query_3gram = p.load(open(queryPath_3gram, 'rb'))
    for i in query_3gram:
        if i[2] > threshold and len(i[1]) < 220:
            test_src = i[0][0]
            test_des = i[0][1]
            test_des2 = i[0][2]
            if test_src == test_des == test_des2:
                continue

            funcList = i[1]
            for predicted_func in funcList:
                # src = predicted_func[0]
                # des = predicted_func[1]
                # des2 = predicted_func[2]
                src = predicted_func[1] + '{' + predicted_func[2][0] + '}'
                des = predicted_func[1] + '{' + predicted_func[2][1] + '}'
                des2 = predicted_func[1] + '{' + predicted_func[2][2] + '}'
                # find the src function if it's already in the dict,
                # otherwise, add it to the dict
                if test_src in global_dict:
                    if src in global_dict[test_src]:
                        pass
                    else:
                        global_dict[test_src].append(src)
                else:
                    global_dict[test_src] = [src]
                # do the same thing to des function
                if test_des in global_dict:
                    if des in global_dict[test_des]:
                        pass
                    else:
                        global_dict[test_des].append(des)
                else:
                    global_dict[test_des] = [des]
                if test_des2 in global_dict:
                    if des2 in global_dict[test_des2]:
                        pass
                    else:
                        global_dict[test_des2].append(des2)
                else:
                    global_dict[test_des2] = [des2]

    for i in query_3gram:
        if i[2] > threshold and len(i[1]) < 220:
            test_src = i[0][0]
            test_des = i[0][1]
            test_des2 = i[0][2]
            if test_src == test_des == test_des2:
                continue

            funcList = i[1]
            for predicted_func in funcList:
                # src = predicted_func[0]
                # des = predicted_func[1]
                # des2 = predicted_func[2]
                src = predicted_func[1] + '{' + predicted_func[2][0] + '}'
                des = predicted_func[1] + '{' + predicted_func[2][1] + '}'
                des2 = predicted_func[1] + '{' + predicted_func[2][2] + '}'
                
    print 'Add Factors Between Nodes'
    keylist = global_dict.keys()
    print(len(keylist))
    for key in keylist:
        funclist = global_dict[key]
        idxs=[]
        if len(funclist) > 0:
            for id1 in range(len(funclist)):
                idxs.append(key+'##'+funclist[id1])
            g.factor(idxs, potential=None, ftype='2')
    '''
    from collections import Counter
    votes = Counter()
    for key in keylist:
        print key
        funclist = global_dict[key]
        label=set()
        idxs=[]
        for id1 in range(len(funclist)):
            print funclist[id1]
            predicted_label = funclist[id1].split('{')[0]
            label.add(predicted_label)
        # if 'openssl-OpenSSL_0_9_7h' in label and 'openssl-OpenSSL_0_9_7i' not in label:
        #     print key
        votes += Counter(label)

    sorted_count = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    print(sorted_count)
    '''
    print 'Begin Running LBP'
    iters, converged = g.lbp(normalize=True, progress=False)
    print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
    print
    # Print out the final marginals
    counts = g.count_rv_marginals(normalize=True)
    return counts

def runFactorGraph3(queryPath_3gram, threshold=0.999):
    potential1 = np.array([
            [0.99, 0.01],
            [0.01, 0.99],
    ])
    # Make an empty graph
    g = fg.Graph()
    global_dict = dict()
    query_3gram = p.load(open(queryPath_3gram, 'rb'))
    for i in query_3gram:
        if i[2] > threshold and len(i[1]) < 220:
            test_src = i[0][0]
            test_des = i[0][1]
            test_des2 = i[0][2]
            if test_src == test_des == test_des2:
                continue
            funcList = i[1]
            for predicted_func in funcList:
                # src = predicted_func[0]
                # des = predicted_func[1]
                # des2 = predicted_func[2]
                src = predicted_func[1] + '{' + predicted_func[2][0] + '}'
                des = predicted_func[1] + '{' + predicted_func[2][1] + '}'
                des2 = predicted_func[1] + '{' + predicted_func[2][2] + '}'
                # find the src function if it's already in the dict,
                # otherwise, add it to the dict, and add it to graph
                if test_src in global_dict:
                    if src in global_dict[test_src]:
                        pass
                    else:
                        global_dict[test_src].append(src)
                        g.rv(test_src+'##'+src, 2)
                else:
                    global_dict[test_src] = [src]
                    g.rv(test_src+'##'+src, 2)
                # do the same thing to des function
                if test_des in global_dict:
                    if des in global_dict[test_des]:
                        pass
                    else:
                        global_dict[test_des].append(des)
                        g.rv(test_des+'##'+des, 2)
                else:
                    global_dict[test_des] = [des]
                    g.rv(test_des+'##'+des, 2)
                if test_des2 in global_dict:
                    if des2 in global_dict[test_des2]:
                        pass
                    else:
                        global_dict[test_des2].append(des2)
                        g.rv(test_des2+'##'+des2, 2)
                else:
                    global_dict[test_des2] = [des2]
                    g.rv(test_des2+'##'+des2, 2)
                #add factor between the two rvs
                # if test_src == 'nginx-{openssl-0.9.7e}{zlib-1.2.7.3}{app_info_free}.emb':
                #     pdb.set_trace()
                if test_src != test_des:
                    g.factor([test_src+'##'+src, test_des+'##'+des], potential=potential1)
                if test_des != test_des2:
                    g.factor([test_des+'##'+des, test_des2+'##'+des2], potential=potential1)

    print 'Add Factors Between Nodes'
    keylist = global_dict.keys()
    print(len(keylist))
    for key in keylist:
        funclist = global_dict[key]
        idxs=[]
        if len(funclist) > 0:
            for id1 in range(len(funclist)):
                idxs.append(key+'##'+funclist[id1])
            g.factor(idxs, potential=None, ftype='2')

    from collections import Counter
    votes = Counter()
    for key in keylist:
        #print key
        funclist = global_dict[key]
        label=set()
        idxs=[]
        for id1 in range(len(funclist)):
            #print funclist[id1]
            predicted_label = funclist[id1].split('{')[0]
            label.add(predicted_label)
        # if 'openssl-OpenSSL_0_9_7h' in label and 'openssl-OpenSSL_0_9_7i' not in label:
        #     print key
        votes += Counter(label)

    sorted_count = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    print(sorted_count)

    print 'Begin Running LBP'
    iters, converged = g.lbp(normalize=True, progress=False)
    print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
    print
    # Print out the final marginals
    counts = g.count_rv_marginals(normalize=True)
    return counts

def runFactorGraph2(queryPath_2gram, threshold=0.999):
    potential1 = np.array([
            [0.999, 0.001],
            [0.001, 0.999],
    ])
    # Make an empty graph
    g = fg.Graph()
    global_dict = dict()
    #pdb.set_trace()
    query_2gram = p.load(open(queryPath_2gram, 'rb'))
    for i in query_2gram:
        if i[2] > threshold and len(i[1]) < 220:
            test_src = i[0][0]
            test_des = i[0][1]
            # if test_src == test_des:
            #     continue
            funcList = i[1]
            for predicted_func in funcList:
                src = predicted_func[1] + '{' + predicted_func[2][0] + '}'
                des = predicted_func[1] + '{' + predicted_func[2][1] + '}'
                # find the src function if it's already in the dict,
                # otherwise, add it to the dict, and add it to graph
                if test_src in global_dict:
                    if src in global_dict[test_src]:
                        pass
                    else:
                        global_dict[test_src].append(src)
                        g.rv(test_src+'##'+src, 2)
                else:
                    global_dict[test_src] = [src]
                    g.rv(test_src+'##'+src, 2)
                # do the same thing to des function
                if test_des in global_dict:
                    if des in global_dict[test_des]:
                        pass
                    else:
                        global_dict[test_des].append(des)
                        g.rv(test_des+'##'+des, 2)
                else:
                    global_dict[test_des] = [des]
                    g.rv(test_des+'##'+des, 2)
                #add factor between the two rvs
                # if test_src == 'nginx-{openssl-0.9.7e}{zlib-1.2.7.3}{app_info_free}.emb':
                #     pdb.set_trace()
                if test_src != test_des or src != des:
                    g.factor([test_src+'##'+src, test_des+'##'+des], potential=potential1, ftype='1')

    print 'Add Factors Between Nodes'
    keylist = global_dict.keys()
    print(len(keylist))
    for key in keylist:
        funclist = global_dict[key]
        idxs=[]
        for id1 in range(len(funclist)):
            idxs.append(key+'##'+funclist[id1])
        g.factor(idxs, potential=None, ftype='2')
    print 'Begin Running LBP'
    iters, converged = g.lbp(normalize=True, progress=False)
    print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
    #g.print_messages()
    print
    # Print out the final marginals
    counts = g.count_rv_marginals(normalize=True)
    return g


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
        outpath = os.path.join(dir, folder, 'test_kNN_0414.p')
        outpath2 = os.path.join(dir, folder, 'out_0414.p')
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
    test_some_binary_ngram(11,12)
    #parallelQuery()
