from binary import Binary
import ConfigParser
import ast
import os
import mongowrapper.MongoWrapper as mdb
import pickle as p
from redis import Redis
import traceback
from db import db
import factorgraph as fg
import matplotlib.pyplot as plt
from Quick_Find import *
from Quick_Union import *
from collections import defaultdict
from collections import Counter
import numpy as np
import pdb
class TestBinary(Binary):
    def __init__(self, binaryName, binFolder, dotPath, embFile, namFile, strFile, filterSize=0, configPath='config'):
        print 'init testing binary'
        self.binaryName = binaryName
        self.dotPath = dotPath
        self.embFile = embFile
        self.namFile = namFile
        self.filterSize = filterSize
        self.binFolder = binFolder
        self.strFile = strFile

        # get configuration
        configParser = ConfigParser.RawConfigParser()
        configParser.read('config')
        self.config = configParser

    def loadBinary(self):
        self.loadCallGraph(self.dotPath, self.namFile)
        self.generatefuncNameFilted(self.namFile, self.filterSize)
        self.getGraphFromPathfilted()
        self.buildNGram(self.namFile, self.embFile)

    def search(self):
        try:
            stringTable = p.load(open(self.strFile, 'r'))
            mongodb = mdb('oss',  'library_stringtable')
        except Exception:
            print "Don't have string database, will not use string information"
            mongodb = None
            stringTable = None

        self.stringTable = stringTable
        self.mongodb = mongodb
        candidateOutput1Gram = os.path.join(self.binFolder, 'test_kNN_1112_1gram.p')
        candidateOutput2Gram = os.path.join(self.binFolder, 'test_kNN_1112_2gram.p')
        countOutput = os.path.join(self.binFolder, 'out_count1102_1gram.p')
        predictionOutput = os.path.join(self.binFolder, 'out_prediction1102_1gram.p')

        if os.path.isfile(candidateOutput1Gram) and os.path.isfile(candidateOutput2Gram):
            knn1Gram = p.load(open(candidateOutput1Gram, 'rb'))
            knn2Gram = p.load(open(candidateOutput2Gram, 'rb'))
        else:
            self.loadBinary()
            knn1Gram = self.queryForOneBinary1Gram(candidateOutput1Gram)
            knn2Gram = self.queryForOneBinary2Gram(candidateOutput2Gram)
        self.knn1Gram = knn1Gram
        self.knn2Gram = knn2Gram

        funcprediction, count = self.BP_with_strings()

        p.dump(count, open(countOutput, 'w'))
        p.dump(funcprediction, open(predictionOutput, 'w'))


    def queryForOneBinary3Gram(self, outpath):
        qdata = self.threeGramList
        redis_object = Redis(host='localhost', port=6379, db=int(self.config.get("ThreeGramRedis", "DATABASE")))
        hashMap = db()
        hashMap.loadHashMap(ast.literal_eval(self.config.get("ThreeGramRedis", "CONFIG")), redis_object, int(self.config.get("ThreeGramRedis", "DIM")), int(self.config.get("ThreeGramRedis", "PROJECTIONS")))

        print("\nStart query for test data")
        testkNN = hashMap.querying(qdata)
        p.dump(testkNN, open(outpath, "w"))
        return testkNN


    def queryForOneBinary2Gram(self, outpath):
        qdata = self.twoGramList
        redis_object = Redis(host='localhost', port=6379, db=self.config.get("TwoGramRedis", "DATABASE"))
        hashMap = db()
        hashMap.loadHashMap(ast.literal_eval(self.config.get("TwoGramRedis", "CONFIG")), redis_object, int(self.config.get("TwoGramRedis", "DIM")), int(self.config.get("TwoGramRedis", "PROJECTIONS")))

        print("\nStart query for 2-gram data")
        testkNN = hashMap.querying(qdata)
        p.dump(testkNN, open(outpath, "w"))
        return testkNN


    def queryForOneBinary1Gram(self, outpath):
        redis_object = Redis(host='localhost', port=6379, db=self.config.get("FuncRedis", "DATABASE"))
        hashMap = db()
        hashMap.loadHashMap(ast.literal_eval(self.config.get("FuncRedis", "CONFIG")), redis_object, int(self.config.get("FuncRedis", "DIM")), int(self.config.get("FuncRedis", "PROJECTIONS")))
        nodes = set()
        for key in self.funcNameFilted.keys():
            if self.funcNameFilted[key] != -1:
                nodes.add(key)
        query = []
        for node in nodes:
            emb = self.funcName2emb[node]
            query.append([emb, node, 1])
        print("\nStart query for 1-gram data")
        testkNN = hashMap.querying(query)
        p.dump(testkNN, open(outpath, "w"))
        return testkNN


    def BP_with_strings(self):
        print "running BP"
        LBP_MAX_ITERS = int(self.config.get("FunctionName", "LBP_MAX_ITERS"))
        threshold = float(self.config.get("FunctionName", "THRESHOLD"))
        g = fg.Graph()
        global_node_dict = dict()
        global_edge_dict = dict()
        global_sim_dict = defaultdict(float)
        rvlabels = defaultdict(list)
        query_2gram = self.knn2Gram
        query_1gram = self.knn1Gram
        hasDistance = True

        global_node_dict, global_prior_dict = get_node_prior_belief(query_1gram, threshold)
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
                if self.mongodb is None:
                    string_shared = set()
                    string_differ = set()
                else:
                    query = self.mongodb.load({"name": prediction})
                    if not query:
                        string_list = []
                    else:
                        string_list = query['strings']
                    string_shared = set(self.stringTable[key]) & set(string_list)
                    string_differ = set(self.stringTable[key]) | set(string_list) - string_shared

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
        # count for libraries and versions
        counts = counting(query_1gram, results, threshold)
        self.get_versions_through_components(query_1gram, results, global_edge_dict, threshold)
        return results, counts

    def shortest_distance(self, component1, component2):
        shortest_distance = 1000000
        for (v, version1) in component1:
            for (w, version2) in component2:
                if abs(self.funcName2Ind[v] - self.funcName2Ind[w]) < shortest_distance:
                    shortest_distance = abs(self.funcName2Ind[v] - self.funcName2Ind[w])
        return shortest_distance

    def get_versions_through_components(self, query_1gram, funcprediction, global_edge_dict, threshold):
        hasDistance = False
        if type(query_1gram[0][0][0]) == type((1,2)):
            hasDistance = True

        prediction = defaultdict(set)
        for [(name, distance), predict, sim] in query_1gram:
            if sim > threshold:
                for [lib, version, func] in predict:
                    if (func, lib) not in funcprediction[name]:
                        continue
                    prediction[name].add(version)

        eleNodeMap = genNodeList(list(prediction.keys()))
        for (src, des) in global_edge_dict.keys():
            if src in prediction.keys() and des in prediction.keys():
                quickUnion((src, des), eleNodeMap)

        name_size = p.load(open(self.namFile, 'r'))
        all_funcs = [i[0] for i in name_size]
        slide_window = 10
        for i, src in enumerate(all_funcs[:-slide_window]):
            for j in range(1, slide_window):
                des = all_funcs[i + j]
                if src in prediction.keys() and des in prediction.keys():
                    if prediction[src] == prediction[des]:
                        quickUnion((src, des), eleNodeMap)

        # get the groups
        groups = dict()
        for i in eleNodeMap.keys():
            if eleNodeMap[i].parent.num not in groups:
                groups[eleNodeMap[i].parent.num] = set()

            groups[eleNodeMap[i].parent.num].add(i)

        # get the components
        components = []
        for component in groups:
            votes = Counter()
            for element in groups[component]:
                votes += Counter(prediction[element])
            sorted_count = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            versions = list(filter(lambda x: x[1] >= sorted_count[0][1], sorted_count))
            print len(groups[component]), versions
            components.append((groups[component], versions))

        components = sorted(components, key=lambda x: len(x[0]))
        return components

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




def counting(query_1gram, funcprediction, threshold):
    hasDistance = True

    prediction = defaultdict(set)
    for [(name, distance), predict, sim] in query_1gram:
        if sim > threshold:
            for [lib, version, func] in predict:
                if (func, lib) not in funcprediction[name]:
                    continue
                prediction[name].add(version)

    votes = Counter()
    for key in prediction:
        votes += Counter(prediction[key])

    sorted_count = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    print sorted_count
    return sorted_count


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

def get_node_prior_belief(knn_all, threshold):
    prediction = defaultdict(set)
    sim_dict = defaultdict(float)
    for [(name, distance), predict, sim] in knn_all:
        if sim > threshold:
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
