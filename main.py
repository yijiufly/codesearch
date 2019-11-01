import numpy as np
from labels import Labels
import cPickle as p
import sys
import operator
import pdb
from binary import TestBinary
from library import Library
from lshknn import queryForOneBinary3Gram, queryForOneBinary2Gram, queryForOneBinary1Gram
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
import mongowrapper.MongoWrapper as mdb
from collections import Counter

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
    libraryName = 'library'
    db = mdb('oss', libraryName + '_stringtable')
    #print "PID:", os.getpid()
    #print i,j
    for folder in folders[i:j]:
        print folder
        dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx'
        start_time = time.time()
        try:
            dotfiles = loadFiles(os.path.join(dir, folder), ext='_bn.dot')
            #dotfiles.sort()
            namfiles = loadFiles(os.path.join(dir, folder), ext='.ida_newmodel_withsize.nam')
            embfiles = loadFiles(os.path.join(dir, folder), ext='.ida_newmodel.emb')
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
            namPathFull = os.path.join(dir, folder, namfile[:-13]+'.nam')
            embPath = os.path.join(dir, folder, embfile)

            #calculate 3gram
            # candidate_output3 = os.path.join(dir, folder, binaryName+'_test_kNN_0521_3gram.p')
            # count_output3 = os.path.join(dir, folder, binaryName+'_BP_undirected.txt')
            binary_dir = os.path.join(dir, folder)
            candidate_output = os.path.join(dir, folder, 'test_kNN_1030_1gram.p')
            string_table = p.load(open('string_dict.p','rb'))
            count_output = os.path.join(dir, folder, 'out_count1030_1gram.p')
            #if os.path.isfile(count_output):
                #result3gram = p.load(open(count_output3, 'rb'))
                #continue
            #testbin = TestBinary(binaryName, dotPath, embPath, namPath, 1)
            #pdb.set_trace()
            #testbin.loadOneBinary(namPathFull, embPath)
            if os.path.isfile(candidate_output):
                #g = BP_with_penalty(candidate_output)
                #count = runBPversions(candidate_output)
                count = counting(candidate_output)
                #g = BP_with_strings(candidate_output, string_table, db)
            else:
                #continue
                testbin = TestBinary(binaryName, dotPath, embPath, namPath, 1)
                testbin.buildNGram(namPathFull)
                #queryForOneBinary3Gram(testbin.threeGramList, candidate_output)
                #kNN = queryForOneBinary2Gram(testbin.twoGramList, candidate_output)
                kNN = queryForOneBinary1Gram(testbin, candidate_output)
                #g = BP_with_penalty(kNN)
                #count = runBPversions(kNN)
                count = counting(kNN)
                #g = BP_with_strings(kNN, string_table, db)
            #accuracy_of_functions(g)
            #testbin.callBP(candidate_output3, candidate_output2, namPath, binary_dir)
            #result = runFactorGraph2gram(candidate_output, folder)
            #funcprediction = runBPfunctionname(kNN, folder)
            #result = runBPversions(graph, testbin)
            #result = testbin.count(candidate_output)
            #result = getcomponentsbyquickunionwithexcude(candidate_output, funcprediction)
            #result = getcomponentsbyquickunion(candidate_output)
            p.dump(count, open(count_output, 'w'))
            #print folder, ', ', result
        print("--- %s seconds ---" % (time.time() - start_time))

def counting(queryPath_1gram):
    global THRES
    threshold=THRES
    if type(queryPath_1gram) == type(''):
        query_1gram = p.load(open(queryPath_1gram, 'rb'))
    else:
        query_1gram = queryPath_1gram
    hasDistance = False
    if type(query_1gram[0][0][0]) == type((1,2)):
        hasDistance = True

    prediction = defaultdict(set)
    for [(name, distance), predict, sim] in query_1gram:
        if sim > threshold:
            for [lib, version, func] in predict:
                prediction[name].add(version)

    votes = Counter()
    for key in prediction:
        votes += Counter(prediction[key])

    sorted_count = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    print sorted_count
    return sorted_count

def accuracy_of_functions(g, rvs=None, normalize=False):
        # Extract
        global_dict = g.global_dict
        tuples = g.rv_marginals(rvs, normalize)
        prediction = dict()
        count = 0
        count2 = 0
        prior = defaultdict(list)
        # Display
        for rv, marg in tuples:
            rv_label = str(rv).split(' ')[0]
            if rv_label[:4] == 'str_':
                continue
            largest = -1
            for label in marg:
                if label > largest:
                    largest = label
            names = global_dict[rv_label]

            # if np.isclose(largest, 1.0/len(names)):
            #     continue
            for i, label in enumerate(marg):
                prior[rv_label].append((names[i], label))
                if np.isclose(label, largest):
                    #pdb.set_trace()
                    if rv_label in prediction:
                        prediction[rv_label].append(names[i])
                    else:
                        prediction[rv_label]=[names[i]]

        keylist = prior.keys()
        # for key in keylist:
        #     print key
        #     print prior[key]
        #     print
        keylist = prediction.keys()
        correct_set = set()
        for key in keylist:
            funclist = prediction[key]
            label=set()
            #print key
            for (func, lib) in funclist:
                if len(func.split('{')) > 1:
                    predicted_label = func.split('{')[1].split('}')[0]
                else:
                    predicted_label = func
                label.add(predicted_label)
            if 'None' in label and len(label)==1:
                count2 += 1
            elif key in label:
                count += 1
                correct_set.add(key)
            # else:
            #     print key

        #         if len(label) > 1:
        #             count2+=1
            # else:
            #     print key, prior[key], label
        #print count, count2,len(keylist)
        print "correct prediction, None prediction, total prediction, precision:"
        print count, count2, len(keylist), count*1.0/(len(keylist)-count2)
        return prediction, prior, correct_set

def BP_with_strings(queryPath_2gram, string_dict, db):
    global LBP_MAX_ITERS
    global THRES
    threshold=THRES
    g = fg.Graph()
    global_node_dict = dict()
    global_edge_dict = dict()
    global_sim_dict = defaultdict(float)
    rvlabels = defaultdict(list)
    if type(queryPath_2gram) == type(''):
        query_2gram = p.load(open(queryPath_2gram, 'rb'))
    else:
        query_2gram = queryPath_2gram
    hasDistance = False
    if type(query_2gram[0][0][0]) == type((1,2)):
        hasDistance = True

    # # first identify the name that are very sure
    sure_dict = defaultdict(list)
    for i in query_2gram:
        if i[2] > 0.99:
            if not hasDistance:
                test_src = i[0][0]
                test_des = i[0][1]
            else:
                test_src = i[0][0][0]
                test_des = i[0][0][1]
            if test_src == test_des:
                continue
#             # initialize the dictionary
            if not global_node_dict.has_key(test_src):
                global_node_dict[test_src] = []
            if not global_node_dict.has_key(test_des):
                global_node_dict[test_des] = []
            if not global_edge_dict.has_key((test_src, test_des)):
                global_edge_dict[(test_src, test_des)] = []
            funcList = i[1]
            # if len(funcList) > 200:
            #     continue
            for predicted_func in funcList:
                if hasDistance:
                    querydistance = i[0][1]
                    resultdistance = predicted_func[3]
                    if querydistance != resultdistance:
                        continue

                src_funcname = predicted_func[2][0]
                des_funcname = predicted_func[2][1]
                libraryname = predicted_func[0]
                # find the src function if it's already in the dict,
                # otherwise, add it to the dict
                if (test_src, libraryname) not in sure_dict[test_src]:
                    sure_dict[test_src].append((src_funcname, libraryname))
                    global_node_dict[test_src].append((src_funcname, libraryname))
                    rvlabels[test_src].append(src_funcname + ', ' + '(' + src_funcname + ', ' + des_funcname + ')')
                # do the same thing to des function
                if (test_des, libraryname) not in sure_dict[test_des]:
                    sure_dict[test_des].append((des_funcname, libraryname))
                    global_node_dict[test_des].append((des_funcname, libraryname))
                    rvlabels[test_des].append(des_funcname + ', ' + '(' + src_funcname + ', ' + des_funcname + ')')
    p.dump(sure_dict, open('sure_dict','w'))

    for i in query_2gram:
        if i[2] > threshold:
            if not hasDistance:
                test_src = i[0][0]
                test_des = i[0][1]
            else:
                test_src = i[0][0][0]
                test_des = i[0][0][1]
            if test_src == test_des:
                continue
#             # initialize the dictionary
            if not global_node_dict.has_key(test_src):
                global_node_dict[test_src] = []
            if not global_node_dict.has_key(test_des):
                global_node_dict[test_des] = []
            if not global_edge_dict.has_key((test_src, test_des)):
                global_edge_dict[(test_src, test_des)] = []
            funcList = i[1]
            for predicted_func in funcList:
                if hasDistance:
                    querydistance = i[0][1]
                    resultdistance = predicted_func[3]
                    if querydistance != resultdistance:
                        continue

                src_funcname = predicted_func[2][0]
                des_funcname = predicted_func[2][1]
                libraryname = predicted_func[0]
                if test_src in sure_dict and (src_funcname, libraryname) not in sure_dict[test_src]:
                    continue
                if test_des in sure_dict and (des_funcname, libraryname) not in sure_dict[test_des]:
                    continue
                # find the src function if it's already in the dict,
                # otherwise, add it to the dict
                if (src_funcname, libraryname) not in global_node_dict[test_src]:
                    global_node_dict[test_src].append((src_funcname, libraryname))
                    rvlabels[test_src].append(src_funcname + ', ' + '(' + src_funcname + ', ' + des_funcname + ')')
                # do the same thing to des function
                if (des_funcname, libraryname) not in global_node_dict[test_des]:
                    global_node_dict[test_des].append((des_funcname, libraryname))
                    rvlabels[test_des].append(des_funcname + ', ' + '(' + src_funcname + ', ' + des_funcname + ')')

                # add this edge to the edge dictionary
                if (src_funcname, des_funcname, libraryname) not in global_edge_dict[(test_src, test_des)]:
                    global_edge_dict[(test_src, test_des)].append((src_funcname, des_funcname, libraryname))
                    if global_sim_dict[(src_funcname, des_funcname, libraryname)] < i[2]:
                        global_sim_dict[(src_funcname, des_funcname, libraryname)] = i[2]

    string_count = defaultdict(int)
    # add functions as random variables and add factors between function and string nodes
    for key in global_node_dict:
        rvlabels[test_src].append('None')
        length = len(global_node_dict[key])
        rv_func = g.rv(key, length + 1) # None label is the last item
        for idx, (prediction, library) in enumerate(global_node_dict[key]):
            query = db.load({"name":prediction})
            if not query:
                string_list = []
            else:
                string_list = query['strings']
            string_shared = set(string_dict[key]) & set(string_list)
            string_differ = set(string_dict[key]) | set(string_list) - string_shared
            #string_differ = set(string_list) - set(string_dict[key])
            for string in string_shared:
                count = string_count[string]
                string_count[string] += 1
                rv_string = g.rv('str_' + string + str(count), 1)
                fillvalue = 1.0/(length + 10)
                pot=np.full((length + 1, 1), fillvalue)
                pot[idx, 0] = 10 * fillvalue
                factor = g.factor([rv_func, rv_string], potential=pot, name = '')

            for string in string_differ:
                count = string_count[string]
                string_count[string] += 1
                rv_string = g.rv('str_' + string + str(count), 1)
                fillvalue = 1.0/(10 * length + 1)
                pot=np.full((length + 1, 1), 10 * fillvalue)
                pot[idx, 0] = fillvalue
                factor = g.factor([rv_func, rv_string], potential=pot, name = '')
        global_node_dict[key].append(('None', 'None'))
    p.dump(global_node_dict, open('global_node_dict','w'))


    # add factors
    for (test_src, test_des) in global_edge_dict.keys():
        src_rv = g.get_rvs()[test_src]
        des_rv = g.get_rvs()[test_des]
        src_list = global_node_dict[test_src]
        des_list = global_node_dict[test_des]
        # set the probability for src_rv and des_rv
        # first find if there is an existing probability, if so, add to it
        # otherwise, initialize a new probability matrix
        factor1 = None
        probability1 = None

        if test_src != test_des:
            pot=np.full((len(src_list), len(des_list)), 0.01)
            #pot[-1, -1] = 0.1
            factor1 = g.factor([src_rv, des_rv], potential=pot, name = '')
            probability1 = factor1.get_potential()

        # if there is an matching edge
        if len(global_edge_dict[(test_src, test_des)]) > 0:
            for (src_funcname, des_funcname, libraryname) in global_edge_dict[(test_src, test_des)]:
                sim = global_sim_dict[(src_funcname, des_funcname, libraryname)]
                src_ind = src_list.index((src_funcname, libraryname))
                des_ind = des_list.index((des_funcname, libraryname))
                if src_rv == factor1.get_rvs()[0]:
                    #probability1[src_ind, des_ind] = 2**(sim + 2) - 7
                    probability1[src_ind, des_ind] = 0.9
                else:
                    #probability1[des_ind, src_ind] = 2**(sim + 2) - 7
                    probability1[des_ind, src_ind] = 0.9
        # else:
        #     if len(src_list) > 0 and len(des_list) > 0:
        #         probability1[:, -1] = 0.3
        #         probability1[-1, :] = 0.3
        factor1.set_potential(probability1)
        #factor1.name += src_funcname + '->' + des_funcname + '\n'

    iters, converged = g.lbp(global_node_dict, normalize=True, max_iters=LBP_MAX_ITERS, progress=False)
    g.global_dict = global_node_dict
    print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
#     results = g.get_func_prediction(normalize=True)
#     p.dump(results, open('prediction','w'))
#     results = g.accuracy_of_functions(normalize=True)
    return g


#############################################################################
# get a list of components for each version
# we compare this method with the BP methods
#############################################################################
def getcomponentsbyquickunionwithexcude(twogramspath, funcprediction=None):
    if type(twogramspath) == type(''):
        twograms = p.load(open(twogramspath, 'rb'))
    else:
        twograms = twogramspath
    #global THRES
    # get all the possible functions
    element = set()
    adj = dict()
    contradictory = []

    hasDistance = False
    if type(twograms[0][0][0]) == type((1,2)):
        hasDistance = True
    for twogram in twograms:
        if twogram[2] > 0.95:
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
                # src = pairs[0] + pairs[1] + '{' + pairs[2][0] + '}'
                # des = pairs[0] + pairs[1] + '{' + pairs[2][1] + '}'
                src = pairs[2][0]
                des = pairs[2][1]
                #pdb.set_trace()
                if funcprediction is not None:
                    if (src, pairs[0]) not in funcprediction[test_src] or (des, pairs[0]) not in funcprediction[test_des]:
                        continue
                if test_src + '##' + src not in adj:
                    adj[test_src + '##' + src] = set()
                adj[test_src + '##' + src].add(test_des + '##' + des)

                if test_des + '##' + des not in adj:
                    adj[test_des + '##' + des] = set()
                adj[test_des + '##' + des].add(test_src + '##' + src)
                element.add(test_src + '##' + pairs[0] + '#' + src)
                element.add(test_des + '##' + pairs[0] + '#' + des)

    # at the beginning, each possible function is a component
    eleNodeMap = genNodeList(list(element))

    for twogram in twograms:
        if twogram[2] > 0.95:
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
                # src = test_src + '##' + pairs[0] + pairs[1] + '{' + pairs[2][0] + '}'
                # des = test_des + '##' + pairs[0] + pairs[1] + '{' + pairs[2][1] + '}'
                src = pairs[2][0]
                des = pairs[2][1]
                if funcprediction is not None:
                    if (src, pairs[0]) not in funcprediction[test_src] or (des, pairs[0]) not in funcprediction[test_des]:
                        continue

                if test_des + '##' + pairs[0] + '#' + des in eleNodeMap and test_src + '##' + pairs[0] + '#' + src in eleNodeMap:
                    quickUnion((test_src + '##' + pairs[0] + '#' + src, test_des + '##' + pairs[0] + '#' + des), eleNodeMap)
                    print test_src + '##' + pairs[0] + '#' + src, test_des + '##' + pairs[0] + '#' + des

    # get the groups
    groups=dict()
    for i in eleNodeMap.keys():
        if eleNodeMap[i].parent.num not in groups:
            groups[eleNodeMap[i].parent.num] = set()

        groups[eleNodeMap[i].parent.num].add(i)

    # get the components
    components = []
    for component in groups:
        components.append(groups[component])
    components = sorted(components, key=lambda x: len(x))
    #p.dump(components, open('component','w'))

    # finalprediction = defaultdict(set)
    # for component in groups:
    #     if len(groups[component]) > 0:
    #         for element in groups[component]:
    #             test_func = element.split('##')[0]
    #             prediction = element.split('##')[1].split('#')[1]
    #             library = element.split('##')[1].split('#')[0]
    #             finalprediction[test_func].add((prediction, library))
    #
    # count = 0
    # count_total=0
    # for key in finalprediction:
    #     if 'libcrypto' in [i[1] for i in finalprediction[key]]:
    #         count_total += 1
    #     if (key, 'libcrypto') in finalprediction[key]:
    #         count += 1
    # print count, count_total
    return components


#############################################################################
#run 2-rounds of BP algorithm, firstly for function names, then for versions
#############################################################################
def runBPversions(queryPath_2gram):
    global LBP_MAX_ITERS
    global THRES
    threshold = THRES
    if type(queryPath_2gram) == type(''):
        query_2gram = p.load(open(queryPath_2gram, 'rb'))
    else:
        query_2gram = queryPath_2gram
    hasDistance = False
    if type(query_2gram[0][0][0]) == type((1,2)):
        hasDistance = True

    global_node_dict = defaultdict(list)
    version_set = set()
    g = fg.Graph()
    global_edge_set = set()
    for i in query_2gram:
        if i[2] > threshold:
            if not hasDistance:
                test_src = i[0][0]
                test_des = i[0][1]
            else:
                test_src = i[0][0][0]
                test_des = i[0][0][1]
            if test_src == test_des:
                continue

            funcList = i[1]
            for predicted_func in funcList:
                if hasDistance:
                    querydistance = i[0][1]
                    resultdistance = predicted_func[3]
                    if querydistance != resultdistance:
                        continue

                version = predicted_func[1]
                libraryname = predicted_func[0]
                version_set.add((libraryname, version))
                if (libraryname, version) not in global_node_dict[test_src]:
                    global_node_dict[test_src].append((libraryname, version))

                if (libraryname, version) not in global_node_dict[test_des]:
                    global_node_dict[test_des].append((libraryname, version))

                global_edge_set.add((test_src, test_des))

    version_list = list(version_set)
    for key in global_node_dict:
        length = len(version_set)
        priorsBelieve = np.zeros(len(version_set))
        for (libraryname, version) in global_node_dict[key]:
            ind = version_list.index((libraryname, version))
            priorsBelieve[ind] = 1
        rv_func = g.rv(key, length, prior = priorsBelieve)

    for (test_src, test_des) in global_edge_set:
        src_rv = g.get_rvs()[test_src]
        des_rv = g.get_rvs()[test_des]
        factor_probability = np.full((len(version_list), len(version_list)), 0.01)
        np.fill_diagonal(factor_probability, 0.99)
        factor1 = g.factor([src_rv, des_rv], potential=factor_probability, name = '')

    # run BP
    #pdb.set_trace()
    iters, converged = g.lbp(global_node_dict, normalize=True, max_iters=LBP_MAX_ITERS, progress=False)
    print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
    #print
    # Print out the final marginals
    #counts = g.print_rv_marginals(global_dict, normalize=True)
    counts = g.count_rv_marginals2(version_list, normalize=True)
    #counts = g.accuracy_of_functions(global_dict, normalize=True)
    #counts = g.print_None_accuracy(global_dict, folder, normalize=True)
    return counts


def runBPfunctionname(queryPath_2gram, folder):
    global LBP_MAX_ITERS
    global THRES
    threshold=THRES
    # Make an empty graph
    g = fg.Graph()
    # build global dict according to candidate 2-gram
    # global dict stores the candidate version and function for each function in testing binary
    global_dict = defaultdict(list)
    rvlabels = defaultdict(list)
    if type(queryPath_2gram) == type(''):
        query_2gram = p.load(open(queryPath_2gram, 'rb'))
    else:
        query_2gram = queryPath_2gram
    hasDistance = False
    if type(query_2gram[0][0][0]) == type((1,2)):
        hasDistance = True

    # # first identify the name that are very sure
    sure_dict = defaultdict(list)
    for i in query_2gram:
        if i[2] > 0.999:
            if not hasDistance:
                test_src = i[0][0]
                test_des = i[0][1]
            else:
                test_src = i[0][0][0]
                test_des = i[0][0][1]
            if test_src == test_des:
                continue

            funcList = i[1]
            # if len(funcList) > 200:
            #     continue
            for predicted_func in funcList:
                if hasDistance:
                    querydistance = i[0][1]
                    resultdistance = predicted_func[3]
                    if querydistance != resultdistance:
                        continue

                src_funcname = predicted_func[2][0]
                des_funcname = predicted_func[2][1]
                libraryname = predicted_func[0]
                # find the src function if it's already in the dict,
                # otherwise, add it to the dict
                if (test_src, libraryname) not in sure_dict[test_src]:
                    sure_dict[test_src].append((src_funcname, libraryname))
                    global_dict[test_src].append((src_funcname, libraryname))
                    rvlabels[test_src].append(src_funcname + ', ' + '(' + src_funcname + ', ' + des_funcname + ')')
                # do the same thing to des function
                if (test_des, libraryname) not in sure_dict[test_des]:
                    sure_dict[test_des].append((des_funcname, libraryname))
                    global_dict[test_des].append((des_funcname, libraryname))
                    rvlabels[test_des].append(des_funcname + ', ' + '(' + src_funcname + ', ' + des_funcname + ')')
    p.dump(sure_dict, open('sure_dict','w'))

    # add other labels, if it contradict to the very sure labels, don't add it
    for i in query_2gram:
        if i[2] > threshold:
            if not hasDistance:
                test_src = i[0][0]
                test_des = i[0][1]
            else:
                test_src = i[0][0][0]
                test_des = i[0][0][1]
            if test_src == test_des:
                continue

            funcList = i[1]
            # if len(funcList) > 200:
            #     continue
            for predicted_func in funcList:
                if hasDistance:
                    querydistance = i[0][1]
                    resultdistance = predicted_func[3]
                    if querydistance != resultdistance:
                        continue

                src_funcname = predicted_func[2][0]
                des_funcname = predicted_func[2][1]
                libraryname = predicted_func[0]

                if test_src in sure_dict and (src_funcname, libraryname) not in sure_dict[test_src]:
                    continue
                if test_des in sure_dict and (des_funcname, libraryname) not in sure_dict[test_des]:
                    continue
                # find the src function if it's already in the dict,
                # otherwise, add it to the dict
                if (src_funcname, libraryname) not in global_dict[test_src]:
                    global_dict[test_src].append((src_funcname, libraryname))
                    rvlabels[test_src].append(src_funcname + ', ' + '(' + src_funcname + ', ' + des_funcname + ')')
                # do the same thing to des function
                if (des_funcname, libraryname) not in global_dict[test_des]:
                    global_dict[test_des].append((des_funcname, libraryname))
                    rvlabels[test_des].append(des_funcname + ', ' + '(' + src_funcname + ', ' + des_funcname + ')')

    # add functions as random variables
    for key in global_dict:
        #src_list = global_dict[key]
        #a = [list(i) for i in src_list]
        #if key not in [row[0] for row in a]:
        #global_dict[key] = list(set((item[0], item[1]) for item in global_dict[key]))
        g.rv(key, len(global_dict[key])) # we don't need None label this time
    p.dump(global_dict, open('global_dict','w'))

    # add factors between each pair of rvs
    for i in query_2gram:
        if i[2] > threshold:
            if not hasDistance:
                test_src = i[0][0]
                test_des = i[0][1]
            else:
                test_src = i[0][0][0]
                test_des = i[0][0][1]
            if test_src == test_des:
                continue
            # if len(i[1]) > 200:
            #     continue
            if test_src not in global_dict or test_des not in global_dict:
                continue
            if test_src not in g.get_rvs() or test_des not in g.get_rvs():
                continue
            src_list = global_dict[test_src]
            des_list = global_dict[test_des]
            src_rv = g.get_rvs()[test_src]
            des_rv = g.get_rvs()[test_des]
            funcList = i[1]
            if len(funcList) == 0:
                continue
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
                pot=np.full((len(src_list), len(des_list)), 0.1)
                factor1 = g.factor([src_rv, des_rv], potential=pot, name = '')
                probability1 = factor1.get_potential()


            for predicted_func in funcList:
                if hasDistance:
                    querydistance = i[0][1]
                    resultdistance = predicted_func[3]
                    if querydistance != resultdistance:
                        continue

                src = predicted_func[2][0]
                des = predicted_func[2][1]
                libraryName = predicted_func[0]
                if (src, libraryName) in src_list and (des, libraryName) in des_list:
                    src_ind = src_list.index((src, libraryName))
                    des_ind = des_list.index((des, libraryName))
                else:
                    continue
                if src_rv == factor1.get_rvs()[0]:
                    probability1[src_ind, des_ind] = 0.99
                else:
                    probability1[des_ind, src_ind] = 0.99

            factor1.set_potential(probability1)
            factor1.name += src + '->' + des + '\n'
    #print 'Begin Running LBP'
    iters, converged = g.lbp(global_dict, normalize=True, max_iters=LBP_MAX_ITERS, progress=False)
    g.global_dict = global_dict
    print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
    #g.print_messages()
    # Print out the final marginals
    results = g.get_func_prediction(normalize=True)
    p.dump(results, open('prediction','w'))
    none_nodes = set()
    for key in results.keys():
        predictions = [i[0][0] for i in results[key]]
        if key not in predictions:
            none_nodes.add(key)
    p.dump(none_nodes, open('none_nodes', 'w'))
    #results = g.print_rv_marginals(global_dict, normalize=True)
    #results = g.count_rv_marginals2(global_dict, normalize=True)
    results = g.accuracy_of_functions(normalize=True)
    g.print_graph('BP.dot')
    #results = g.print_None_accuracy(global_dict, folder, normalize=True)
    return results


def BP_with_penalty(queryPath_2gram):
    global LBP_MAX_ITERS
    global THRES
    threshold=THRES
    g = fg.Graph()
    global_node_dict = dict()
    global_edge_dict = dict()
    rvlabels = defaultdict(list)
    if type(queryPath_2gram) == type(''):
        query_2gram = p.load(open(queryPath_2gram, 'rb'))
    else:
        query_2gram = queryPath_2gram
    hasDistance = False
    if type(query_2gram[0][0][0]) == type((1,2)):
        hasDistance = True

    for i in query_2gram:
        if i[2] > threshold:
            if not hasDistance:
                test_src = i[0][0]
                test_des = i[0][1]
            else:
                test_src = i[0][0][0]
                test_des = i[0][0][1]
            if test_src == test_des:
                continue
            # initialize the dictionary
            if not global_node_dict.has_key(test_src):
                global_node_dict[test_src] = []
            if not global_node_dict.has_key(test_des):
                global_node_dict[test_des] = []
            if not global_edge_dict.has_key((test_src, test_des)):
                global_edge_dict[(test_src, test_des)] = []


            funcList = i[1]
            for predicted_func in funcList:
                if hasDistance:
                    querydistance = i[0][1]
                    resultdistance = predicted_func[3]
                    if querydistance != resultdistance:
                        continue

                src_funcname = predicted_func[2][0]
                des_funcname = predicted_func[2][1]
                libraryname = predicted_func[0]

                # find the src function if it's already in the dict,
                # otherwise, add it to the dict
                if (src_funcname, libraryname) not in global_node_dict[test_src]:
                    global_node_dict[test_src].append((src_funcname, libraryname))
                    rvlabels[test_src].append(src_funcname + ', ' + '(' + src_funcname + ', ' + des_funcname + ')')
                # do the same thing to des function
                if (des_funcname, libraryname) not in global_node_dict[test_des]:
                    global_node_dict[test_des].append((des_funcname, libraryname))
                    rvlabels[test_des].append(des_funcname + ', ' + '(' + src_funcname + ', ' + des_funcname + ')')

                # add this edge to the edge dictionary
                if (src_funcname, des_funcname, libraryname) not in global_edge_dict[(test_src, test_des)]:
                    global_edge_dict[(test_src, test_des)].append((src_funcname, des_funcname, libraryname))

    # add functions as random variables
    for key in global_node_dict:
        #src_list = global_dict[key]
        #a = [list(i) for i in src_list]
        #if key not in [row[0] for row in a]:
        #global_dict[key] = list(set((item[0], item[1]) for item in global_dict[key]))
        global_node_dict[key].append(('None', 'None'))
        rvlabels[test_src].append('None')
        g.rv(key, len(global_node_dict[key])) # None label is the last item
    p.dump(global_node_dict, open('global_node_dict','w'))

    # add factors
    for (test_src, test_des) in global_edge_dict.keys():
        src_rv = g.get_rvs()[test_src]
        des_rv = g.get_rvs()[test_des]
        src_list = global_node_dict[test_src]
        des_list = global_node_dict[test_des]
        # set the probability for src_rv and des_rv
        # first find if there is an existing probability, if so, add to it
        # otherwise, initialize a new probability matrix
        factor1 = None
        probability1 = None

        if test_src != test_des:
            pot=np.full((len(src_list), len(des_list)), 0.1)
            #pot[-1, -1] = 0.2
            factor1 = g.factor([src_rv, des_rv], potential=pot, name = '')
            probability1 = factor1.get_potential()

        # if there is an matching edge
        if len(global_edge_dict[(test_src, test_des)]) > 0:
            for (src_funcname, des_funcname, libraryname) in global_edge_dict[(test_src, test_des)]:
                src_ind = src_list.index((src_funcname, libraryname))
                des_ind = des_list.index((des_funcname, libraryname))
                if src_rv == factor1.get_rvs()[0]:
                    probability1[src_ind, des_ind] = 0.99
                else:
                    probability1[des_ind, src_ind] = 0.99

        else:
            if len(src_list) > 0 and len(des_list) > 0:
                probability1[:, -1] = 0.9
                probability1[-1, :] = 0.9
#             elif len(src_list) > 0 and len(des_list) == 0:
#                 if src_rv == factor1.get_rvs()[0]:
#                     probability1[-1, 0] = 0.3
#                 else:
#                     probability1[0, -1] = 0.3
#             elif len(src_list) == 0 and len(des_list) > 0:
#                 if src_rv == factor1.get_rvs()[0]:
#                     probability1[0, -1] = 0.3
#                 else:
#                     probability1[-1, 0] = 0.3
            else:
                continue
        factor1.set_potential(probability1)
        #factor1.name += src_funcname + '->' + des_funcname + '\n'

    iters, converged = g.lbp(global_node_dict, normalize=True, max_iters=LBP_MAX_ITERS, progress=False)
    g.global_dict = global_node_dict
    print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
    results = g.get_func_prediction(normalize=True)
    p.dump(results, open('prediction','w'))
    results = g.accuracy_of_functions(normalize=True)
    return g


#############################################################################
# run BP algorithm, label is library plus version plus function name
#############################################################################
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
                pot=np.full((len(src_list)+1, len(des_list)+1), 0.01)
                #pot=np.full((len(src_list), len(des_list)), 0.3)
                #pot[-1,:] = 0.8 # set probability for none
                #pot[:,-1] = 0.8
                pot[-1,-1] = 0.8
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
    print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
    #print
    # Print out the final marginals
    #counts = g.print_rv_marginals(global_dict, normalize=True)
    #counts = g.count_rv_marginals2(global_dict, normalize=True)
    counts = g.accuracy_of_functions(global_dict, normalize=True)

    #counts = g.print_None_accuracy(global_dict, folder, normalize=True)
    return counts


################################################################################
#cross validation part, use it when we want to determine some parameters
################################################################################
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

#######################################################################
#Main
#######################################################################
if __name__ == '__main__':
    global LBP_MAX_ITERS
    global THRES
    LBP_MAX_ITERS = 40
    THRES = 0.9999
    dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx'
    folders = os.listdir(dir)
    folders.sort()
    #print folders.index('nginx-{openssl-1.0.1j}{zlib-1.2.7.3}')
    #parallelQuery(folders)
    selected = ['nginx-{openssl-0.9.8p}{zlib-1.2.11}', 'nginx-{openssl-0.9.8t}{zlib-1.2.8}', 'nginx-{openssl-0.9.8w}{zlib-1.2.11}', 'nginx-{openssl-1.0.0a}{zlib-1.2.10}', 'nginx-{openssl-1.0.0g}{zlib-1.2.7.3}', 'nginx-{openssl-1.0.0i}{zlib-1.2.7.2}', 'nginx-{openssl-1.0.0j}{zlib-1.2.8}', 'nginx-{openssl-1.0.0k}{zlib-1.2.7.1}', 'nginx-{openssl-1.0.0p}{zlib-1.2.8}', 'nginx-{openssl-1.0.1a}{zlib-1.2.7}', 'nginx-{openssl-1.0.1b}{zlib-1.2.7.3}', 'nginx-{openssl-1.0.1e}{zlib-1.2.7.3}', 'nginx-{openssl-1.0.1s}{zlib-1.2.7}', 'nginx-{openssl-1.0.2g}{zlib-1.2.8}', 'nginx-{openssl-1.0.2m}{zlib-1.2.9}', 'nginx-{openssl-1.1.0b}{zlib-1.2.7.1}']
    test_some_binary_ngram(0,len(folders),folders)
    #three_fold_cross_validation()
