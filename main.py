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
import mongowrapper.MongoWrapper as mdb

def loadFiles(PATH, ext=None):  # use .ida or .emb for ida file and embedding file
    filenames = []
    filenames = [f for f in os.listdir(PATH) if f.endswith(ext)]
    return filenames

def test_one_binary(binaryName, dotPath, funcembFolder, queryPath, namPath, libs=None):
    testbin = TestBinary(binaryName, dotPath, funcembFolder)
    testbin.buildNGram(namPath)
    if os.path.isfile(queryPath):
        pass
    else:
        queryForOneBinary(testbin.threeGramList, queryPath)
    testbin.count(queryPath)


def parallelQuery(folders):
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

def test_some_binary_ngram(i, j, folders, dir):
    libraryName = 'library'
    db = mdb('oss', libraryName + '_stringtable')
    #print "PID:", os.getpid()
    #print i,j
    for folder in folders[i:j]:
        print folder
        start_time = time.time()
        try:
            dotfiles = loadFiles(os.path.join(dir, folder), ext='_bn.dot')
            namfiles = loadFiles(os.path.join(dir, folder), ext='.ida_newmodel_withsize.nam')
            embfiles = loadFiles(os.path.join(dir, folder), ext='.ida_newmodel.emb')
            strfiles = loadFiles(os.path.join(dir, folder), ext='.str')
            embfiles.sort()
        except Exception:
            print traceback.format_exc()
            continue

        for i in range(len(dotfiles)):
            dotfile = dotfiles[i]
            namfile = namfiles[i]
            embfile = embfiles[i]
            strfile = strfiles[i]
            binaryName = dotfile.split('.')[0]
            dotPath = os.path.join(dir, folder, dotfile)
            namPath = os.path.join(dir, folder, namfile)
            namPathFull = os.path.join(dir, folder, namfile[:-13]+'.nam')
            embPath = os.path.join(dir, folder, embfile)
            strPath = os.path.join(dir, folder, strfile)

            binary_dir = os.path.join(dir, folder)
            candidate_output_1gram = os.path.join(dir, folder, 'test_kNN_1030_1gram.p')
            candidate_output_2gram = os.path.join(dir, folder, 'test_kNN_1030_2gram.p')
            count_output = os.path.join(dir, folder, 'out_count1102_1gram.p')
            prediction_output = os.path.join(dir, folder, 'out_prediction1102_1gram.p')

            try:
                string_table = p.load(open(strPath, 'r'))
                mongodb = mdb('oss',  'library_stringtable')
            except Exception:
                mongodb = None

            if os.path.isfile(candidate_output_1gram) and os.path.isfile(candidate_output_2gram):
                funcprediction, count = BP_with_strings(candidate_output_1gram, candidate_output_2gram, string_table, mongodb)
            else:
                testbin = TestBinary(binaryName, dotPath, embPath, namPath, 1)
                testbin.buildNGram(namPathFull)
                kNN_1gram = queryForOneBinary1Gram(testbin, candidate_output_1gram)
                kNN_2gram = queryForOneBinary1Gram(testbin, candidate_output_2gram)
                funcprediction, count = BP_with_strings(kNN_1gram, kNN_2gram, string_table, mongodb)

            p.dump(count, open(count_output, 'w'))
            p.dump(funcprediction, open(prediction_output, 'w'))
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
            for i, label in enumerate(marg):
                prior[rv_label].append((names[i], label))
                if np.isclose(label, largest):
                    #pdb.set_trace()
                    if rv_label in prediction:
                        prediction[rv_label].append(names[i])
                    else:
                        prediction[rv_label]=[names[i]]

        keylist = prior.keys()
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
        print "correct prediction, None prediction, total prediction, precision:"
        print count, count2, len(keylist), count*1.0/(len(keylist)-count2)
        return prediction, prior, correct_set

def BP_with_strings(queryPath_1gram, queryPath_2gram, string_dict, db = None):
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
        query_1gram = p.load(open(queryPath_1gram, 'rb'))
    else:
        query_2gram = queryPath_2gram
        query_1gram = queryPath_1gram
    hasDistance = False
    if type(query_2gram[0][0][0]) == type((1,2)):
        hasDistance = True

    # count for libraries and versions
    counts = counting(query_1gram)
    global_node_dict, global_prior_dict = get_node_prior_belief(query_1gram)
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

                # find the src function if it's already in the dict,
                # otherwise, add it to the dict
                if (src_funcname, libraryname) not in global_node_dict[test_src]:
                    continue
                # do the same thing to des function
                if (des_funcname, libraryname) not in global_node_dict[test_des]:
                    continue

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
        if length > 0:
            #rv_func = g.rv(key, length + 1, prior = global_prior_dict[key])
            rv_func = g.rv(key, length + 1)
        else:
            rv_func = g.rv(key, length + 1) # None label is the last item
        for idx, (prediction, library) in enumerate(global_node_dict[key]):
            if db is None:
                string_shared = set()
                string_differ = set()
            else:
                query = db.load({"name":prediction})
                if not query:
                    string_list = []
                else:
                    string_list = query['strings']
                string_shared = set(string_dict[key]) & set(string_list)
                string_differ = set(string_dict[key]) | set(string_list) - string_shared

            for string in string_shared:
                count = string_count[string]
                string_count[string] += 1
                rv_string = g.rv('str_' + string + str(count), 1)
                fillvalue = 1.0
                pot=np.full((length + 1, 1), fillvalue)
                pot[idx, 0] = 5 * fillvalue
                factor = g.factor([rv_func, rv_string], potential=pot, name = '')

            for string in string_differ:
                count = string_count[string]
                string_count[string] += 1
                rv_string = g.rv('str_' + string + str(count), 1)
                fillvalue = 1.0
                pot=np.full((length + 1, 1), 5 * fillvalue)
                pot[idx, 0] = fillvalue
                factor = g.factor([rv_func, rv_string], potential=pot, name = '')
        global_node_dict[key].append(('None', 'None'))
    #p.dump(global_node_dict, open('global_node_dict','w'))


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
                    probability1[src_ind, des_ind] = 0.9
                else:
                    probability1[des_ind, src_ind] = 0.9
#         else:
#             if len(src_list) > 0 and len(des_list) > 0:
#                 probability1[:, -1] = 0.3
#                 probability1[-1, :] = 0.3

        factor1.set_potential(probability1)

    iters, converged = g.lbp(global_node_dict, normalize=True, max_iters=LBP_MAX_ITERS, progress=False)
    g.global_dict = global_node_dict
    print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
    results = g.get_func_prediction(normalize=True)
    return results, counts


def getRank(prediction):
    predict = sorted(prediction, key=lambda x: x[1])
    rank = 1
    lastsim = -1
    predicts = []
    ranks = []
    for (func, sim) in predict:
        if np.isclose(sim, lastsim):
            predicts.append(func)
            ranks.append(rank)
        else:
            predicts.append(func)
            ranks.append(rank)
            rank += 1
        lastsim = sim
    sm = softmax(range(ranks[-1]+1))
    for i in range(len(ranks)):
        ranks[i] = sm[ranks[i]]
    ranks.append(sm[0])
    return predicts, ranks

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def get_node_prior_belief(knn_all):
    prediction = defaultdict(set)
    sim_dict = defaultdict(float)
    global THRES
    for [(name, distance), predict, sim] in knn_all:
        if sim > THRES:
            for [lib, version, func] in predict:
                prediction[name].add((func, lib))
                if sim_dict[(name, func, lib)] < sim:
                    sim_dict[(name, func, lib)] = sim

    prediction_with_sim = defaultdict(list)

    for name in prediction.keys():
        for (func, lib) in prediction[name]:
            prediction_with_sim[name].append(((func, lib), sim_dict[(name, func, lib)]))

    global_dict = defaultdict(list)
    global_beliefs = defaultdict(list)
    for func in prediction_with_sim.keys():
        predicts, beliefs = getRank(prediction_with_sim[func])
        global_dict[func] = predicts
        global_beliefs[func] = beliefs

    return global_dict, global_beliefs

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
def runBPversions(queryPath_1gram, global_edge_dict, funcprediction):
    global LBP_MAX_ITERS
    global THRES
    threshold = THRES
    if type(queryPath_1gram) == type(''):
        query_1gram = p.load(open(queryPath_1gram, 'rb'))
    else:
        query_1gram = queryPath_1gram
    hasDistance = False
    if type(query_1gram[0][0]) == type((1,2)):
        hasDistance = True

    global_node_dict = defaultdict(list)
    version_set = set()
    global_edge_set = set()
    for i in query_1gram:
        if i[2] > threshold:
            if not hasDistance:
                test_func = i[0]
            else:
                test_func = i[0][0]

            funcList = i[1]
            if hasDistance:
                querydistance = i[0][1]
                resultdistance = i[2]
                if querydistance != resultdistance:
                    continue

            for predicted_func in funcList:
                version = predicted_func[1]
                libraryname = predicted_func[0]
                funcname = predicted_func[2]

                # if (funcname, libraryname) not in funcprediction[test_func]:
                #     continue
                version_set.add((libraryname, version))
                if (libraryname, version) not in global_node_dict[test_func]:
                    global_node_dict[test_func].append((libraryname, version))

    g = fg.Graph()
    version_list = list(version_set)
    for key in global_node_dict:
        length = len(version_set)
        priorsBelieve = np.zeros(len(version_set))
        for (libraryname, version) in global_node_dict[key]:
            ind = version_list.index((libraryname, version))
            priorsBelieve[ind] = 1
        g.rv(key, length, priorsBelieve)

    for (test_src, test_des) in global_edge_dict.keys():
        if test_src not in g.get_rvs() or test_des not in g.get_rvs():
            continue
        src_rv = g.get_rvs()[test_src]
        des_rv = g.get_rvs()[test_des]
        factor_probability = np.full((len(version_list), len(version_list)), 0.01)
        np.fill_diagonal(factor_probability, 0.99)
        g.factor([src_rv, des_rv], potential=factor_probability)
    #g.remove_loner_rvs()
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
    #parallelQuery(folders)
    test_some_binary_ngram(0, len(folders), folders, dir)
    #three_fold_cross_validation()
