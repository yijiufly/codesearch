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
import csv
import sys
sys.path.insert(0, "faiss/python")
import faiss
import time
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
            mongodb = mdb(self.config.get("Mongodb", "DBNAME"),  self.config.get("Mongodb", "TABLENAME"))
            #mongodb = mdb('oss', 'library' + '_stringtable')
        except Exception:
            print "Don't have string database, will not use string information"
            mongodb = None
            stringTable = None

        self.stringTable = stringTable
        self.mongodb = mongodb
        candidateOutput1Gram = os.path.join(self.binFolder, 'test_kNN_1125_1gram.p')
        candidateOutput2Gram = os.path.join(self.binFolder, 'test_kNN_1127_2gram.p')
        predictionOutput = os.path.join(self.binFolder, 'out_prediction1120_BP.csv')

        if os.path.isfile(candidateOutput1Gram) and os.path.isfile(candidateOutput2Gram):
            knn1Gram = p.load(open(candidateOutput1Gram, 'rb'))
            knn2Gram = p.load(open(candidateOutput2Gram, 'rb'))
        else:
            self.loadBinary()
            start_time = time.time()
            knn1Gram = self.queryForOneBinary1GramFaiss(candidateOutput1Gram)
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            knn2Gram = self.queryForOneBinary2GramFaiss(candidateOutput2Gram)
            print("--- %s seconds ---" % (time.time() - start_time))
        self.knn1Gram = knn1Gram
        self.knn2Gram = knn2Gram

        start_time = time.time()
        funcprediction = self.BP_with_strings()
        print("--- %s seconds ---" % (time.time() - start_time))
        print_CSV(funcprediction, predictionOutput)


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


    def queryForOneBinary1GramFaiss(self, outpath):
        index = faiss.read_index("funcemb_aftergrouping_cosine2.index")
        meta = p.load(open('funcmetadata_aftergrouping_cosine2.p', 'r'))
        nodes = []
        query = []
        for key in self.funcNameFilted.keys():
            if self.funcNameFilted[key] != -1:
                nodes.append(key)
                emb = self.funcName2emb[key]
                query.append(emb)
        query_norm = query / np.linalg.norm(query, axis=1)[:,None]
        D, I = index.search(np.array(query_norm), 10)
        N_query = []
        for query_node, distances, indexes in zip(nodes, D, I):
            for d, i in zip(distances, indexes):
                N_query.append([(query_node, 1), meta[i], d])
        p.dump(N_query, open(outpath, "w"))
        return N_query

    def queryForOneBinary2GramFaiss(self, outpath):
        index = faiss.read_index("2gram_mupdf.index")
        meta = p.load(open('2grammetadata_mupdf.p', 'r'))
        # index = faiss.read_index("tmp/2gramemb.index")
        # meta = p.load(open('tmp/2grammetadata.p', 'r'))
        nodes = []
        query = []
        for [twogram, name, distance] in self.twoGramList:
            nodes.append((name, distance))
            query.append(twogram)
        query_norm = query / np.linalg.norm(query, axis=1)[:,None]
        D, I = index.search(np.array(query_norm), 20)
        N_query = []
        for query_node, distances, indexes in zip(nodes, D, I):
            for d, i in zip(distances, indexes):
                N_query.append([query_node, meta[i], d])
        p.dump(N_query, open(outpath, "w"))
        return N_query

    def BP_with_strings(self):
        print "running BP"
        LBP_MAX_ITERS = int(self.config.get("FunctionName", "LBP_MAX_ITERS"))
        threshold = float(self.config.get("FunctionName", "THRESHOLD"))
        threshold_2gram = float(self.config.get("FunctionName", "THRESHOLD_2GRAM"))
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
            if not hasDistance:
                test_src = i[0][0]
                test_des = i[0][1]
                #test_des2 = i[0][2]
            else:
                test_src = i[0][0][0]
                test_des = i[0][0][1]
                #test_des2 = i[0][0][2]
            # if test_src == test_des:
            #     continue
              # initialize the dictionary
            if not global_node_dict.has_key(test_src):
                global_node_dict[test_src] = []
            if not global_node_dict.has_key(test_des):
                global_node_dict[test_des] = []
            # if not global_node_dict.has_key(test_des2):
            #     global_node_dict[test_des2] = []
            if not global_edge_dict.has_key((test_src, test_des)):
                global_edge_dict[(test_src, test_des)] = []
            # if not global_edge_dict.has_key((test_des, test_des2)):
            #     global_edge_dict[(test_des, test_des2)] = []
            if i[2] > threshold_2gram:
                funcList = i[1]
                for predicted_func in funcList:
                    if hasDistance:
                        querydistance = i[0][1]
                        resultdistance = predicted_func[3]
                        if querydistance != resultdistance:
                            continue

                    src_funcname = predicted_func[2][0]
                    des_funcname = predicted_func[2][1]
                    #des2_funcname = predicted_func[2][2]
                    libraryname = predicted_func[0]

                    # find the src function if it's already in the dict,
                    # if it's not, meaning that 1-gram matching didn't find it, so don't add it
                    if (src_funcname, libraryname) not in global_node_dict[test_src]:
                        continue
                    # do the same thing to des function
                    if (des_funcname, libraryname) not in global_node_dict[test_des]:
                        continue
                    
                    # if (des2_funcname, libraryname) not in global_node_dict[test_des2]:
                    #     continue

                    # add this edge to the edge dictionary
                    if test_src != test_des:
                        if (src_funcname, des_funcname, libraryname) not in global_edge_dict[(test_src, test_des)]:
                            global_edge_dict[(test_src, test_des)].append((src_funcname, des_funcname, libraryname))
                            if global_sim_dict[(src_funcname, des_funcname, libraryname)] < i[2]:
                                global_sim_dict[(src_funcname, des_funcname, libraryname)] = i[2]

                    # if test_des != test_des2:
                    #     if (des_funcname, des2_funcname, libraryname) not in global_edge_dict[(test_des, test_des2)]:
                    #         global_edge_dict[(test_des, test_des2)].append((des_funcname, des2_funcname, libraryname))
                    #         if global_sim_dict[(des_funcname, des2_funcname, libraryname)] < i[2]:
                    #             global_sim_dict[(des_funcname, des2_funcname, libraryname)] = i[2]
                        
        edges = [['test_src', 'test_des', 'src_funcname', 'des_funcname', 'libraryname', 'sim']]
        for (test_src, test_des) in global_edge_dict.keys():
            edges.append([test_src, test_des, '', '', '', ''])
            for (src_funcname, des_funcname, libraryname) in global_edge_dict[(test_src, test_des)]:
                edges.append([test_src, test_des, src_funcname, des_funcname, libraryname, global_sim_dict[(src_funcname, des_funcname, libraryname)]])
        
        print_CSV(edges, 'tmp/edges.csv')
        string_count = defaultdict(int)
        # add functions as random variables and add factors between function and string nodes
        for key in global_node_dict:
            rvlabels[test_src].append('None')
            length = len(global_node_dict[key])
            if length > 0:
                rv_func = g.rv(key, length + 1, prior = global_prior_dict[key])
                #rv_func = g.rv(key, length + 1)
            else:
                rv_func = g.rv(key, length + 1) # None label is the last item
            for idx, (prediction, library) in enumerate(global_node_dict[key]):
                if self.mongodb is None:
                    string_shared = set()
                    string_differ = set()
                else:
                    query = self.mongodb.load({"name": prediction})
                    if not query:
                        string_set = set()
                    else:
                        string_set = set()
                        for item in query:
                            string_set |= set(item['strings'])
                    string_shared = set(self.stringTable[key]) & string_set
                    string_differ = set(self.stringTable[key]) | string_set - string_shared

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
                pot=np.full((len(src_list), len(des_list)), 0.1)
                #pot[-1, -1] = 0.1
                factor1 = g.factor([src_rv, des_rv], potential=pot, name = '')
                probability1 = factor1.get_potential()
            else:
                continue

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
            else:
                if len(src_list) > 1 and len(des_list) > 1:
                    src_lib = set([i[1] for i in src_list])
                    des_lib = set([i[1] for i in des_list])
                    # print test_src, test_des
                    # print len(src_list), len(des_list)
                    if src_lib == des_lib:
                        probability1[:, -1] = 0.9
                        probability1[-1, :] = 0.9
                # else:
                #     probability1[-1, -1] = 0.9


            factor1.set_potential(probability1)

        g.global_dict = global_node_dict
        iters, converged = g.lbp(global_node_dict, normalize=True, max_iters=0, progress=False)   
        print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
        results = g.get_func_prediction(normalize=True)
        
        
        # count for libraries and versions
        #counts = counting(query_1gram, results, threshold)
        version_predict = self.get_versions_through_components(query_1gram, results, global_edge_dict, threshold)
        functionpredict, correct_set = self.precision_and_recall(results, version_predict)
        print_CSV(functionpredict,  os.path.join(self.binFolder, 'out_prediction1120_Baseline.csv'))

        iters, converged = g.lbp(global_node_dict, normalize=True, max_iters=LBP_MAX_ITERS, progress=False)   
        print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
        results = g.get_func_prediction(normalize=True)
        # count for libraries and versions
        #counts = counting(query_1gram, results, threshold)
        version_predict = self.get_versions_through_components(query_1gram, results, global_edge_dict, threshold)
        functionpredict, correct_set2 = self.precision_and_recall(results, version_predict)
        print correct_set-correct_set2
        return functionpredict

    def get_versions_through_components(self, query_1gram, funcprediction, global_edge_dict, threshold):
        prediction = defaultdict(set)
        for [(name, distance), predict, sim] in query_1gram:
            if sim > threshold:
                for [lib, version, func] in predict:
                    if lib == 'libcrypto' or lib == 'libssl':
                        lib = 'openssl'
                    if (func, lib) not in funcprediction[name]:
                        continue
                    prediction[name].add(version)

        # group functions on the same call edge
        eleNodeMap = genNodeList(list(prediction.keys()))
        for (src, des) in global_edge_dict.keys():
            if src in prediction.keys() and des in prediction.keys():
                quickUnion((src, des), eleNodeMap)

        # group functions within the sliding window, and has the same library predictions
        name_size = p.load(open(self.namFile, 'r'))
        all_funcs = [i[0] for i in name_size]
        slide_window = 20
        for i, src in enumerate(all_funcs[:-slide_window]):
            for j in range(1, slide_window):
                des = all_funcs[i + j]
                if src in prediction.keys() and des in prediction.keys():
                    library_src = set([item.split('-')[0] for item in prediction[src]])
                    library_des = set([item.split('-')[0] for item in prediction[des]])
                    if library_src == library_des:
                        quickUnion((src, des), eleNodeMap)

        # get the groups
        groups = dict()
        for i in eleNodeMap.keys():
            if eleNodeMap[i].parent.num not in groups:
                groups[eleNodeMap[i].parent.num] = set()

            groups[eleNodeMap[i].parent.num].add(i)

        # get the components
        for component in groups:
            #pdb.set_trace()
            votes = Counter()
            for element in groups[component]:
                votes += Counter(prediction[element])
                if 'openssl-1.0.1f' in prediction[element] and 'openssl-1.0.1d' not in prediction[element]:
                    print element
            sorted_count = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            versions = list(filter(lambda x: x[1] >= sorted_count[0][1], sorted_count))

            for func in groups[component]:
                prediction[func] = [version[0] for version in versions]

        return prediction

    def collect_ground_truth(self):
        # TODO: in order to evaluate the precision and recall, need to know how many functions 
        # in the testing binary are library functions as ground truth
        # right now just read it from config file
        return int(self.config.get("GroundTruth", "LIBRARYFUNCTOTAL"))

    def precision_and_recall(self, name_prediction, version_prediction):
        librarynames =self.collect_ground_truth()
        count = 0
        libraries = ast.literal_eval(self.config.get("GroundTruth", "VERSIONS"))
        n = 0
        f = 0
        keylist = name_prediction.keys()
        correct_set = set()
        final_prediction = [['Query Func', 'Prediction', 'Library', 'Version','Correctness']]
        for key in keylist:
            funclist = name_prediction[key]
            label=set()
            for (func, lib) in funclist:
                label.add(func)
                if func != 'None':
                    if key == func:
                        final_prediction.append([key, func, lib, version_prediction[key], 'True'])
                    else:
                        final_prediction.append([key, func, lib, version_prediction[key], 'False'])
            if 'None' in label and len(label)==1:
                n += 1
            elif key in label:# and set(libraries) & set(version_prediction[key]) != set():
                count += 1
                correct_set.add(key)
                f += len(label)-1
            else:
                f += len(label)
        print 'correct prediction', '\t', 'wrong prediction', '\t', 'precision', '\t', 'recall:'
        print count, '\t', f, '\t', count*1.0/(count + f), '\t', count*1.0/librarynames
        print count, '\t', f, '\t', count*1.0/(len(keylist)-n), '\t', count*1.0/librarynames
        final_prediction.append(['Precision:', count*1.0/(count + f), 'Recall:', count*1.0/librarynames])
        return final_prediction, correct_set

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
            if sim > 0.99:
                ranks.append(rank + 10)
            else:
                ranks.append(rank)
        else:
            predicts.append(func)
            if sim > 0.99:
                ranks.append(rank + 10)
            else:
                ranks.append(rank)
                rank += 1
        lastsim = sim
    if lastsim < 0.99:
        ranks.append(ranks[-1])
    else:
        ranks.append(1)
    sm = softmax(ranks)
    for i in range(len(ranks)):
        ranks[i] = sm[i]
    return predicts, ranks

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def get_node_prior_belief(knn_all, threshold):
    prediction = defaultdict(set)
    sim_dict = defaultdict(float)
    for [(name, distance), predict, sim] in knn_all:
        if sim > threshold:
            for [lib, version, func] in predict:
                if lib == 'libcrypto' or lib == 'libssl':
                    lib = 'openssl'
                prediction[name].add((func, lib))
                if sim_dict[(name, func, lib)] < sim:
                    sim_dict[(name, func, lib)] = sim

    prediction_with_sim = defaultdict(list)

    for name in prediction.keys():
        for (func, lib) in prediction[name]:
            prediction_with_sim[name].append(((func, lib), sim_dict[(name, func, lib)]))

    global_dict = defaultdict(list)
    global_beliefs = defaultdict(list)
    # pdb.set_trace()
    for func in prediction_with_sim.keys():
        predicts, beliefs = getRank(prediction_with_sim[func])
        global_dict[func] = predicts
        global_beliefs[func] = beliefs
    # for name in prediction.keys():
    #     sim_list = []
    #     prediction_list = []
    #     smallest = 1.0
    #     for (func, lib) in prediction[name]:
    #         sim = sim_dict[(name, func, lib)]
    #         if sim < smallest:
    #             smallest = sim
    #         sim_list.append(sim)
    #         prediction_list.append((func, lib))
    #     sim_list.append(smallest)
    #     global_dict[name] = prediction_list
    #     global_beliefs[name] = list(softmax(sim_list))

    return global_dict, global_beliefs

def print_CSV(funcpredict, output):
    with open(output,'wb') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerows(funcpredict)
