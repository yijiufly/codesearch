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
import matplotlib.pyplot as plt
from Quick_Find import *
from Quick_Union import *
from collections import defaultdict
try:
    import queue
except ImportError:
    import Queue as queue
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

def parallelQuery(folders):
    #dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx'
    #funcembFolder = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/funcemb_testing'
    #folders = os.listdir(dir)
    folders.sort()
    l = len(folders)
    cores = 2#multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    num = l/cores
    start_time = time.time()
    for i in range(cores):
        pool.apply_async(test_some_binary_ngram, (i*num, (i+1)*num, folders, ))
    if num*cores < l:
        pool.apply_async(test_some_binary_ngram, (num*cores, l, folders, ))

    pool.close()
    pool.join()
    print "parallelQuery done"
    print("--- totoal time: %s seconds ---" % (time.time() - start_time))

def test_some_binary_ngram(i, j, folders):
    dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx'
    #print "PID:", os.getpid()
    #print i,j
    for folder in folders[i:j]:
# def test_some_binary_ngram(folders):
#     for folder in folders:
        #print(folder)
        dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx'
        start_time = time.time()
        try:
            dotfiles = loadFiles(os.path.join(dir, folder), ext='_bn.dot')
            #dotfiles.sort()
            namfiles = loadFiles(os.path.join(dir, folder), ext='_filted1.nam')
            embfiles = loadFiles(os.path.join(dir, folder), ext='.emb')
            embfiles.sort()
        except Exception:
            print traceback.format_exc()
            continue

        for i in range(len(dotfiles)):
            dotfile = dotfiles[i]
            namfile = namfiles[i]
            embfile = embfiles[i]
            binaryName = dotfile.split('.')[0]
            dotPath = os.path.join(dir, folder, dotfile)
            namPath = os.path.join(dir, folder, namfile)
            namPathFull = os.path.join(dir, folder, namfile[:-12]+'.nam')
            embPath = os.path.join(dir, folder, embfile)

            #calculate 3gram
            # candidate_output3 = os.path.join(dir, folder, binaryName+'_test_kNN_0521_3gram.p')
            # count_output3 = os.path.join(dir, folder, binaryName+'_BP_undirected.txt')
            binary_dir = os.path.join(dir, folder)
            candidate_output = os.path.join(dir, folder, 'test_kNN_0716_2gram.p')
            count_output = os.path.join(dir, folder, 'out_test0716_2gram.p')
            #if os.path.isfile(count_output):
                #result3gram = p.load(open(count_output3, 'rb'))
                #continue
            #testbin = TestBinary(binaryName, dotPath, embPath, namPath)
            if os.path.isfile(candidate_output):
                pass
            else:
                #continue
                testbin = TestBinary(binaryName, dotPath, embPath, namPath)
                testbin.buildNGram(namPathFull)
                #queryForOneBinary3Gram(testbin.threeGramList, candidate_output)
                queryForOneBinary2Gram(testbin.twoGramList, candidate_output)

            #testbin.callBP(candidate_output3, candidate_output2, namPath, binary_dir)
            result = runFactorGraph2gram(candidate_output, folder)
            #result = testbin.count(candidate_output)
            #result = getcomponentsbyquickunionwithexcude(candidate_output)
            #result = getcomponentsbyquickunion(candidate_output)
            p.dump(result, open(count_output, 'w'))
        #print("--- %s seconds ---" % (time.time() - start_time))
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

def findLargest(adj, contra):
    largest = 0
    largestitem = None
    #
    for item in contra:
        q = queue.Queue()
        q.put(item)
        visited = dict()
        visited[item]=1
        count = 0
        while not q.empty():
            top = q.get()
            count += 1
            if top in adj:
                for son in adj[top]:
                    if son not in visited:
                        q.put(son)
                        visited[son]=1

        if count > largest:
            largest = count
            largestitem = [item]
        elif count == largest:
            largestitem.append(item)

    return largestitem

def getcomponentsbyquickunionwithexcude(twogramspath):
    twograms = p.load(open(twogramspath,'r'))

    # get all the possible functions
    element = set()
    adj = dict()
    contradictory = []

    hasDistance = False
    if type(twograms[0][0][0]) == type((1,2)):
        hasDistance = True
    for twogram in twograms:
        if twogram[2] > 0.9999:
            if len(twogram[1]) < 5 and twogram[2]<0.99995:
                continue
            if not hasDistance:
                test_src = twogram[0][0]
                test_des = twogram[0][1]
            else:
                test_src = twogram[0][0][0]
                test_des = twogram[0][0][1]
            for pairs in twogram[1]:
                if hasDistance:
                    querydistance = twogram[0][1]
                    resultdistance = pairs[3]
                    if querydistance != resultdistance:
                        continue
                #src = pairs[0]
                #des = pairs[1]
                src = pairs[0] + pairs[1] + '{' + pairs[2][0] + '}'
                des = pairs[0] + pairs[1] + '{' + pairs[2][1] + '}'
                if test_src + '##' + src not in adj:
                    adj[test_src + '##' + src] = set()
                adj[test_src + '##' + src].add(test_des + '##' + des)

                if test_des + '##' + des not in adj:
                    adj[test_des + '##' + des] = set()
                adj[test_des + '##' + des].add(test_src + '##' + src)
                element.add(test_src + '##' + src)
                element.add(test_des + '##' + des)

    # find the contradictories, remove it from adjecent dictionary
    for key in adj:
        temp = defaultdict(list)
        for item in adj[key]:
            temp[item.split('##')[0]].append(item)
        for node in temp:
            if len(temp[node]) > 1:
                contradictory.append([key, node, temp[node]])
                adj[key]-=set(temp[node])
                for n in temp[node]:
                    adj[n].remove(key)

    # find the largest component for each set of contradictory
    for contra in contradictory:
        largest = findLargest(adj, contra[2])
        remove = set(contra[2])-set(largest)
        for re in remove:
            if re in element:
                element.remove(re)


    # at the beginning, each possible function is a component
    eleNodeMap = genNodeList(list(element))

#     # for each two gram, merge the two possible functions into one component
#     for entry in adj:
#         for des in adj[entry]:
#             if des in eleNodeMap and entry in eleNodeMap:
#                 quickUnion((entry, des), eleNodeMap)

    for twogram in twograms:
        if twogram[2] > 0.9999:
            if len(twogram[1]) < 5 and twogram[2]<0.99995:
                continue
            if not hasDistance:
                test_src = twogram[0][0]
                test_des = twogram[0][1]
            else:
                test_src = twogram[0][0][0]
                test_des = twogram[0][0][1]

            for pairs in twogram[1]:
                # src = pairs[0]
                # des = pairs[1]
                if hasDistance:
                    querydistance = twogram[0][1]
                    resultdistance = pairs[3]
                    if querydistance != resultdistance:
                        continue
                src = test_src + '##' + pairs[0] + pairs[1] + '{' + pairs[2][0] + '}'
                des = test_des + '##' + pairs[0] + pairs[1] + '{' + pairs[2][1] + '}'
                if des in eleNodeMap and src in eleNodeMap:
                    quickUnion((src, des), eleNodeMap)


    # get the groups
    groups=dict()
    for i in eleNodeMap.keys():
        if eleNodeMap[i].parent.num not in groups:
            groups[eleNodeMap[i].parent.num] = set()

        nodename = i.split('##')[0]
        groups[eleNodeMap[i].parent.num].add(nodename)

    # get the components
    components = dict()
    for component in groups:
        label = component.split('##')[1].split('{')[0]
        if label not in components:
            components[label]=[]
        components[label].append(groups[component])

    # delete overlapped component
    for label in components:
        pre = components[label]
        post = []
        for i in pre:
            add = True
            for j in post:
                if len(i.intersection(j)) > 0:
                    if len(i) < len(j):
                        add = False
                        break
                    elif len(i) == len(j):
                        continue
                    else:
                        post.remove(j)
            if add:
                post.append(i)
        components[label]=[]

        for i in post:
            components[label].append(len(i))

        # sort to get the predicted label
        components[label].sort(reverse=True)
    components = sorted(components.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    #print components
    return components

def getcomponentsbyquickunion(twogramspath):
    twograms = p.load(open(twogramspath,'r'))

    # get all the possible functions
    element = set()
    for twogram in twograms:
        if twogram[2] > 0.9999:
            if len(twogram[1]) < 5 and twogram[2]<THRES:
                continue
            test_src = twogram[0][0]
            test_des = twogram[0][1]
            for pairs in twogram[1]:
                # src = pairs[0]
                # des = pairs[1]
                src = pairs[0] + pairs[1] + '{' + pairs[2][0] + '}'
                des = pairs[0] + pairs[1] + '{' + pairs[2][1] + '}'
                element.add(test_src + '##' + src)
                element.add(test_des + '##' + des)

    # at the beginning, each possible function is a component
    eleNodeMap = genNodeList(list(element))

    # for each two gram, merge the two possible functions into one component
    for twogram in twograms:
        if twogram[2] > 0.9999:
            if len(twogram[1]) < 5 and twogram[2]<THRES:
                continue
            test_src = twogram[0][0]
            test_des = twogram[0][1]

            for pairs in twogram[1]:
                # src = pairs[0]
                # des = pairs[1]
                src = test_src + '##' + pairs[0] + pairs[1] + '{' + pairs[2][0] + '}'
                des = test_des + '##' + pairs[0] + pairs[1] + '{' + pairs[2][1] + '}'
                quickUnion((src, des), eleNodeMap)
                #result = [(i, eleNodeMap[i].num) for i in eleNodeMap.keys()]

    # get the groups
    groups=dict()
    for i in eleNodeMap.keys():
        if eleNodeMap[i].parent.num not in groups:
            groups[eleNodeMap[i].parent.num] = set()

        nodename = i.split('##')[0]
        groups[eleNodeMap[i].parent.num].add(nodename)

    # get the components
    components = dict()
    for component in groups:
        label = component.split('##')[1].split('{')[0]
        if label not in components:
            components[label]=[]
        #components[label].append(groups[component])
        components[label].append(len(groups[component]))

    # delete overlapped components
    # largest = []
    # for label in components:
    #     pre = components[label]
    #     post = []
    #     for i in pre:
    #         add = True
    #         for j in post:
    #             if len(i.intersection(j)) > 0:
    #                 if len(i) < len(j):
    #                     add = False
    #                     break
    #                 elif len(i) == len(j):
    #                     continue
    #                 else:
    #                     post.remove(j)
    #         if add:
    #             post.append(i)
    #     components[label]=[]
    #
    #     for i in post:
    #         components[label].append(len(i))

        # sort to get the predicted label
    for label in components:
        components[label].sort(reverse=True)
    components = sorted(components.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    #print components
    return components

def runFactorGraph2gram(queryPath_2gram, folder, threshold=0.9999):
    global LBP_MAX_ITERS
    global THRES
    # Make an empty graph
    g = fg.Graph()

    # build global dict according to candidate 2-gram
    # global dict stores the candidate version and function for each function in testing binary
    global_dict = dict()
    query_2gram = p.load(open(queryPath_2gram, 'rb'))
    hasDistance = False
    if type(query_2gram[0][0][0]) == type((1,2)):
        hasDistance = True
    for i in query_2gram:
        if i[2] > threshold:
            if len(i[1]) < 5 and i[2] < THRES:
                continue
            if not hasDistance:
                test_src = i[0][0]
                test_des = i[0][1]
            else:
                test_src = i[0][0][0]
                test_des = i[0][0][1]
            if test_src == test_des:
                continue

            funcList = i[1]
            if len(funcList) > 200:
                continue
            #hasTruth = False
            #for predicted_func in funcList:
        #        version = folder.split('{')[1].split('}')[0].split('-')[1].replace('.','_')
        #        truth = 'libcryptoopenssl-OpenSSL_'+version
        #        truth2 = 'libsslopenssl-OpenSSL_'+version
        #        if truth == (predicted_func[0] + predicted_func[1]) or truth2 == (predicted_func[0] + predicted_func[1]):
        #            hasTruth = True
        #            break
        #    if not hasTruth:
        #        continue
            for predicted_func in funcList:
                if hasDistance:
                    querydistance = i[0][1]
                    resultdistance = predicted_func[3]
                    if querydistance != resultdistance:
                        continue

                # src = predicted_func[0]
                # des = predicted_func[1]
                src = predicted_func[0] + predicted_func[1] + '{' + predicted_func[2][0] + '}'
                des = predicted_func[0] + predicted_func[1] + '{' + predicted_func[2][1] + '}'
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

    # add functions as random variables
    for key in global_dict:
        version = folder.split('{')[1].split('}')[0].split('-')[1].replace('.','_')
        truth = 'libcryptoopenssl-OpenSSL_'+version
        truth2 = 'libsslopenssl-OpenSSL_'+version
        src_list = global_dict[key]
        #if truth + '{' + key + '}' in src_list or truth2 + '{' + key + '}' in src_list:
        g.rv(key, len(src_list)+1) # the last label is none

    # add factors between each pair of rvs
    for i in query_2gram:
        if i[2] > threshold:
            if len(i[1]) < 5 and i[2] < THRES:
                continue
            if not hasDistance:
                test_src = i[0][0]
                test_des = i[0][1]
            else:
                test_src = i[0][0][0]
                test_des = i[0][0][1]
            if test_src == test_des:
                continue
            if len(i[1]) > 200:
                continue
            #if test_src not in global_dict or test_des not in global_dict:
            #    continue
            src_list = global_dict[test_src]
            des_list = global_dict[test_des]
            #if test_src not in g.get_rvs() or test_des not in g.get_rvs():
            #    continue
            src_rv = g.get_rvs()[test_src]
            des_rv = g.get_rvs()[test_des]

            # set the probability for src_rv and des_rv
            # first find if there is an existing probability, if so, add to it
            # otherwise, initialize a new probability matrix
            factor1 = None
            probability1 = None
            des_factor = des_rv.get_factors()
            for factor in des_factor:
                if src_rv in factor.get_rvs():
                    factor1 = factor
                    probability1 = factor1.get_potential()

            if factor1 == None and test_src != test_des:
                pot=np.full((len(src_list)+1, len(des_list)+1), 0.3)
                #pot=np.full((len(src_list), len(des_list)), 0.3)
                pot[-1,:] = 0.8 # set probability for none
                pot[:,-1] = 0.8
                #pot[-1,-1] = 0.8
                factor1 = g.factor([src_rv, des_rv], potential=pot)
                probability1 = factor1.get_potential()

            funcList = i[1]
            #hasTruth = False
            #for predicted_func in funcList:
            #    version = folder.split('{')[1].split('}')[0].split('-')[1].replace('.','_')
            #    truth = 'libcryptoopenssl-OpenSSL_'+version
            #    truth2 = 'libsslopenssl-OpenSSL_'+version
            #    if truth == (predicted_func[0] + predicted_func[1]) or truth2 == (predicted_func[0] + predicted_func[1]):
            #        hasTruth = True
            #        break
            #if not hasTruth:
            #    continue
            for predicted_func in funcList:
                # src = predicted_func[0]
                # des = predicted_func[1]
                if hasDistance:
                    querydistance = i[0][1]
                    resultdistance = predicted_func[3]
                    if querydistance != resultdistance:
                        continue

                src = predicted_func[0] + predicted_func[1] + '{' + predicted_func[2][0] + '}'
                des = predicted_func[0] + predicted_func[1] + '{' + predicted_func[2][1] + '}'
                src_ind = src_list.index(src)
                des_ind = des_list.index(des)
                if src_rv == factor1.get_rvs()[0]:
                    probability1[src_ind, des_ind] = 0.99
                    for ind in range(len(probability1)):
                        if ind != src_ind:
                            probability1[ind, des_ind] = 0.01
                    for ind in range(len(probability1[0])):
                        if ind != des_ind:
                            probability1[src_ind, ind] = 0.01
                    # probability1[src_ind, -1] = 0.01
                    # probability1[-1, des_ind] = 0.01
                else:
                    probability1[des_ind, src_ind] = 0.99
                    for ind in range(len(probability1)):
                        if ind != des_ind:
                            probability1[ind, src_ind] = 0.01
                    for ind in range(len(probability1[0])):
                        if ind != src_ind:
                            probability1[des_ind, ind] = 0.01
                    # probability1[des_ind, -1] = 0.01
                    # probability1[-1, src_ind] = 0.01

            factor1.set_potential(probability1)

    '''
    from collections import Counter
    votes = Counter()
    for key in global_dict:
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
    #print 'Begin Running LBP'
    iters, converged = g.lbp(global_dict, normalize=True, max_iters=LBP_MAX_ITERS, progress=False)
    #print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
    #print
    # Print out the final marginals
    #counts = g.print_rv_marginals(global_dict, normalize=True)
    counts = g.count_rv_marginals2(global_dict, normalize=True)
    #counts = g.print_None_accuracy(global_dict, folder, normalize=True)
    return counts


def runFactorGraph(queryPath_3gram, threshold=0.999):
    global LBP_MAX_ITERS
    global THRES
    # Make an empty graph
    g = fg.Graph()
    global_dict = dict()
    query_3gram = p.load(open(queryPath_3gram, 'rb'))
    for i in query_3gram:
        if i[2] > threshold and len(i[1]) < 220:
            if len(i[1]) < 5 and i[2]<THRES:
                continue
            test_src = i[0][0]
            test_des = i[0][1]
            test_des2 = i[0][2]
            if test_src == test_des == test_des2:
                continue

            funcList = i[1]

            for predicted_func in funcList:
                src = predicted_func[0]
                des = predicted_func[1]
                des2 = predicted_func[2]
                # src = predicted_func[1] + '{' + predicted_func[2][0] + '}'
                # des = predicted_func[1] + '{' + predicted_func[2][1] + '}'
                # des2 = predicted_func[1] + '{' + predicted_func[2][2] + '}'
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

    # add functions as random variables
    for key in global_dict:
        src_list = global_dict[key]
        g.rv(key, len(src_list))

    # add factors between each pair of rvs
    for i in query_3gram:
        if i[2] > threshold and len(i[1]) < 220:
            if len(i[1]) < 5 and i[2]<THRES:
                continue
            test_src = i[0][0]
            test_des = i[0][1]
            test_des2 = i[0][2]
            if test_src == test_des == test_des2:
                continue
            src_list = global_dict[test_src]
            des_list = global_dict[test_des]
            des2_list = global_dict[test_des2]

            src_rv = g.get_rvs()[test_src]
            des_rv = g.get_rvs()[test_des]
            des2_rv = g.get_rvs()[test_des2]

            factor1 = None
            factor2 = None
            probability1 = None
            probability2 = None
            des_factor = des_rv.get_factors()
            for factor in des_factor:
                if src_rv in factor.get_rvs():
                    factor1 = factor
                    probability1 = factor1.get_potential()
                if des2_rv in factor.get_rvs():
                    factor2 = factor
                    probability2 = factor2.get_potential()

            if factor1 == None and test_src != test_des:
                factor1 = g.factor([src_rv, des_rv], potential=np.full((len(src_list)+1, len(des_list)+1), 0.1))
                probability1 = factor1.get_potential()
            if factor2 == None and test_des != test_des2:
                factor2 = g.factor([des_rv, des2_rv], potential=np.full((len(des_list)+1, len(des2_list)+1), 0.1))
                probability2 = factor2.get_potential()

            funcList = i[1]
            for predicted_func in funcList:
                src = predicted_func[0]
                des = predicted_func[1]
                des2 = predicted_func[2]
                # src = predicted_func[1] + '{' + predicted_func[2][0] + '}'
                # des = predicted_func[1] + '{' + predicted_func[2][1] + '}'
                # des2 = predicted_func[1] + '{' + predicted_func[2][2] + '}'
                src_ind = src_list.index(src)
                des_ind = des_list.index(des)
                des2_ind = des2_list.index(des2)
                if test_src != test_des:
                    if src_rv == factor1.get_rvs()[0]:
                        probability1[src_ind, des_ind] = 0.8
                    else:
                        probability1[des_ind, src_ind] = 0.8
                if test_des != test_des2:
                    if des_rv == factor2.get_rvs()[0]:
                        probability2[des_ind, des2_ind] = 0.8
                    else:
                        probability2[des2_ind, des_ind] = 0.8
            if test_src != test_des:
                factor1.set_potential(probability1)
            if test_des != test_des2:
                factor2.set_potential(probability2)

    '''
    from collections import Counter
    votes = Counter()
    for key in global_dict:
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
    iters, converged = g.lbp(normalize=True, max_iters=LBP_MAX_ITERS, progress=False)
    print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
    print
    # Print out the final marginals
    counts = g.count_rv_marginals2(global_dict, normalize=True)
    return counts
#
# def runFactorGraph3(queryPath_3gram, threshold=0.999):
#     potential1 = np.array([
#             [0.95, 0.05],
#             [0.05, 0.95],
#     ])
#     # Make an empty graph
#     g = fg.Graph()
#     global_dict = dict()
#     query_3gram = p.load(open(queryPath_3gram, 'rb'))
#     for i in query_3gram:
#         if i[2] > threshold and len(i[1]) < 220:
#             if len(i[1]) < 5 and i[2]<THRES:
#                 continue
#             test_src = i[0][0]
#             test_des = i[0][1]
#             test_des2 = i[0][2]
#             if test_src == test_des == test_des2:
#                 continue
#             funcList = i[1]
#             for predicted_func in funcList:
#                 # src = predicted_func[0]
#                 # des = predicted_func[1]
#                 # des2 = predicted_func[2]
#                 src = predicted_func[1] + '{' + predicted_func[2][0] + '}'
#                 des = predicted_func[1] + '{' + predicted_func[2][1] + '}'
#                 des2 = predicted_func[1] + '{' + predicted_func[2][2] + '}'
#                 # find the src function if it's already in the dict,
#                 # otherwise, add it to the dict, and add it to graph
#                 if test_src in global_dict:
#                     if src in global_dict[test_src]:
#                         pass
#                     else:
#                         global_dict[test_src].append(src)
#                         g.rv(test_src+'##'+src, 2)
#                 else:
#                     global_dict[test_src] = [src]
#                     g.rv(test_src+'##'+src, 2)
#                 # do the same thing to des function
#                 if test_des in global_dict:
#                     if des in global_dict[test_des]:
#                         pass
#                     else:
#                         global_dict[test_des].append(des)
#                         g.rv(test_des+'##'+des, 2)
#                 else:
#                     global_dict[test_des] = [des]
#                     g.rv(test_des+'##'+des, 2)
#                 if test_des2 in global_dict:
#                     if des2 in global_dict[test_des2]:
#                         pass
#                     else:
#                         global_dict[test_des2].append(des2)
#                         g.rv(test_des2+'##'+des2, 2)
#                 else:
#                     global_dict[test_des2] = [des2]
#                     g.rv(test_des2+'##'+des2, 2)
#                 #add factor between the two rvs
#                 # if test_src == 'nginx-{openssl-0.9.7e}{zlib-1.2.7.3}{app_info_free}.emb':
#                 #     pdb.set_trace()
#                 if test_src != test_des:
#                     g.factor([test_src+'##'+src, test_des+'##'+des], potential=potential1)
#                 if test_des != test_des2:
#                     g.factor([test_des+'##'+des, test_des2+'##'+des2], potential=potential1)
#
#     print 'Add Factors Between Nodes'
#     keylist = global_dict.keys()
#     print(len(keylist))
#     for key in keylist:
#         funclist = global_dict[key]
#         idxs=[]
#         if len(funclist) > 0:
#             for id1 in range(len(funclist)):
#                 idxs.append(key+'##'+funclist[id1])
#             g.factor(idxs, potential=None, ftype='2')
#     '''
#     from collections import Counter
#     votes = Counter()
#     for key in keylist:
#         #print key
#         funclist = global_dict[key]
#         label=set()
#         idxs=[]
#         for id1 in range(len(funclist)):
#             #print funclist[id1]
#             predicted_label = funclist[id1].split('{')[0]
#             label.add(predicted_label)
#         # if 'openssl-OpenSSL_0_9_7h' in label and 'openssl-OpenSSL_0_9_7i' not in label:
#         #     print key
#         votes += Counter(label)
#
#     sorted_count = sorted(votes.items(), key=lambda x: x[1], reverse=True)
#     print(sorted_count)
#     '''
#     print 'Begin Running LBP'
#     iters, converged = g.lbp(normalize=True, max_iters=LBP_MAX_ITERS, progress=False)
#     print 'LBP ran for %d iterations. Converged percentage = %f' % (iters, converged)
#     print
#     # Print out the final marginals
#     counts = g.count_rv_marginals(normalize=True)
#     return counts
#
# def runFactorGraph2(queryPath_2gram, threshold=0.999):
#     potential1 = np.array([
#             [0.999, 0.001],
#             [0.001, 0.999],
#     ])
#     # Make an empty graph
#     g = fg.Graph()
#     global_dict = dict()
#     #pdb.set_trace()
#     query_2gram = p.load(open(queryPath_2gram, 'rb'))
#     for i in query_2gram:
#         if i[2] > threshold and len(i[1]) < 220:
#             test_src = i[0][0][0]
#             test_des = i[0][0][1]
#             # if test_src == test_des:
#             #     continue
#             funcList = i[1]
#             for predicted_func in funcList:
#                 querydistance = i[0][1]
#                 resultdistance = predicted_func[3]
#                 if querydistance != resultdistance:
#                     continue
#                 src = predicted_func[1] + '{' + predicted_func[2][0] + '}'
#                 des = predicted_func[1] + '{' + predicted_func[2][1] + '}'
#                 # find the src function if it's already in the dict,
#                 # otherwise, add it to the dict, and add it to graph
#                 if test_src in global_dict:
#                     if src in global_dict[test_src]:
#                         pass
#                     else:
#                         global_dict[test_src].append(src)
#                         g.rv(test_src+'##'+src, 2)
#                 else:
#                     global_dict[test_src] = [src]
#                     g.rv(test_src+'##'+src, 2)
#                 # do the same thing to des function
#                 if test_des in global_dict:
#                     if des in global_dict[test_des]:
#                         pass
#                     else:
#                         global_dict[test_des].append(des)
#                         g.rv(test_des+'##'+des, 2)
#                 else:
#                     global_dict[test_des] = [des]
#                     g.rv(test_des+'##'+des, 2)
#                 #add factor between the two rvs
#                 # if test_src == 'nginx-{openssl-0.9.7e}{zlib-1.2.7.3}{app_info_free}.emb':
#                 #     pdb.set_trace()
#                 if test_src != test_des or src != des:
#                     g.factor([test_src+'##'+src, test_des+'##'+des], potential=potential1, ftype='1')
#
#     print 'Add Factors Between Nodes'
#     keylist = global_dict.keys()
#     print(len(keylist))
#     for key in keylist:
#         funclist = global_dict[key]
#         idxs=[]
#         for id1 in range(len(funclist)):
#             idxs.append(key+'##'+funclist[id1])
#         g.factor(idxs, potential=None, ftype='2')
#     print 'Begin Running LBP'
#     iters, converged = g.lbp(normalize=True, progress=False)
#     print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
#     #g.print_messages()
#     print
#     # Print out the final marginals
#     counts = g.count_rv_marginals(normalize=True)
#     return counts


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

def analysing_results(folders):
    dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx'
    count = 0
    count2 = 0
    for folder in folders:
        path = os.path.join(dir, folder, 'out_factorgraph_3gram.p')
        if os.path.isfile(path):
            count2 += 1
            a=p.load(open(path ,'r'))
            #a.sort(key = lambda x: x[1])
            #print os.listdir(path[:-16])
            #print folder
            truth=folder.split('{')[1].split('}')[0].split('-')[1].replace('.','')
            if a != None and len(a)>0:
                #print a[:10]
                #prediction = a[0][0][-6:].replace('_','')

                prediction = a[0][0][8:-13].replace('_','')
                if prediction == truth:
                    count += 1
    return count*1.0/count2

def train(folders, threshold):
    global LBP_MAX_ITERS
    global THRES
    THRES = threshold
    max_accuracy = -1
    max_iter = -1
    for iter in range(1,10,2):
        LBP_MAX_ITERS = iter
        parallelQuery(folders)
        accuracy = analysing_results(folders)
        print 'iteration: ' + str(iter) + ' ' + str(accuracy)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_iter = iter
    print 'max iteration, accuracy:' + str(max_iter) + ' ' + str(max_accuracy)
    return max_iter

def errorrate(folders, max_iter):
    global LBP_MAX_ITERS
    LBP_MAX_ITERS = max_iter
    parallelQuery(folders)
    accuracy = analysing_results(folders)
    return 1-accuracy

def three_fold_cross_validation():
    dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx'
    folders = os.listdir(dir)
    fold_len = len(folders)/3
    train1 = folders[:2*fold_len]
    test1 = folders[2*fold_len:]
    train2 = folders[:fold_len]
    train2.extend(folders[2*fold_len:])
    test2 = folders[fold_len:2*fold_len]
    train3 = folders[fold_len:]
    test3 = folders[:fold_len]
    x_axis = np.linspace(0.999, 1, num=9)
    y1 = np.zeros(len(x_axis))
    for i, threshold in enumerate(x_axis):
        max_iter = train(train1,threshold)
        er1 = errorrate(test1,max_iter)
        print(er1)
        max_iter = train(train2,threshold)
        er2 = errorrate(test2,max_iter)
        print(er2)
        max_iter = train(train3,threshold)
        er3 = errorrate(test3,max_iter)
        print(er3)
        y1[i]=np.average([er1,er2,er3])

    plt.plot(x_axis, y1, linewidth=1)
    #plt.legend(["use extra features", "don\'t use extra features"])
    plt.xlabel('threshold')
    plt.ylabel('cross-validation error')
    plt.show()
    p.dump(y1, open('crossvalidationresults.p'))


if __name__ == '__main__':
    global LBP_MAX_ITERS
    global THRES
    LBP_MAX_ITERS = 1
    THRES = 0.9999
    dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx'
    folders = os.listdir(dir)
    folders.sort()
    parallelQuery(folders)
    #test_some_binary_ngram(9,10,folders)
    #three_fold_cross_validation()
