import pydot
import pickle as p
import pdb
import numpy as np
import traceback
import os
from sslh.test_SSLH_inference import runBP
from collections import Counter


class Binary:
    def __init__(self, path, path2):
        raise NotImplementedError

    def generatefuncNameFullIDA(self, path):
        self.graph = pydot.graph_from_dot_file(path)
        print('graph loaded')
        nodeList = self.graph.get_node_list()
        func2ind = dict()
        ind2func = dict()
        #pdb.set_trace()
        for node in nodeList:
            label = node.obj_dict['attributes'].get('label', None)
            if label is not None:
                func2ind[label.strip('\"').split('\\l')[0]] = node.get_name()
                ind2func[node.get_name()] = label.strip('\"').split('\\l')[0]
        self.funcName2Ind = func2ind
        self.ind2FuncName = ind2func

    def loadCallGraph(self, path):
        self.graph = pydot.graph_from_dot_file(path)
        print('graph loaded')
        print(path)
        nodeList = self.graph.get_node_list()
        func2ind = dict()
        ind2func = dict()
        #load all the nodes in the callgraph
        for i, node in enumerate(nodeList):
            nodename = node.get_name().strip('\"')
            func2ind[nodename] = i
            ind2func[i] = nodename

        self.funcName2Ind = func2ind
        self.ind2FuncName = ind2func

    def generatefuncNameFilted(self, path):
        func2ind = dict()
        funcNameList = p.load(open(path, 'r'))
        for func in self.funcName2Ind.keys():
            if func in funcNameList:
                func2ind[func] = self.funcName2Ind[func]
            else:
                func2ind[func] = -1
        self.funcNameFilted = func2ind

    def getGraphFromPath(self):
        edgeList = self.graph.get_edge_list()
        linklistgraph = dict()
        #smallnodes = dict()
        #self.callgraphEdges = []
        for edge in edgeList:
            src = edge.get_source()
            des = edge.get_destination()
            if src in linklistgraph:
                linklistgraph[src].append((des, 1))
            else:
                linklistgraph[src] = [(des, 1)]
            # if des in linklistgraph:
            #    linklistgraph[des].append((src, 1))
            # else:
            #    linklistgraph[des] = [(src, 1)]
        print('edges loaded')
        # keylist = linklistgraph.keys()
        # findsmallnodes = -1
        # while findsmallnodes != 0:
        #     findsmallnodes = 0
        #     for src in keylist:
        #         for (des, distance) in linklistgraph[src]:
        #             if self.funcNameFilted[self.ind2FuncName[des]] == -1:
        #                 findsmallnodes += 1
        #                 linklistgraph[src].remove((des, distance))
        #                 if des in smallnodes:
        #                     for (des2, distance2) in smallnodes[des]:
        #                         if (des2, distance2 + distance) not in linklistgraph[src]:
        #                             linklistgraph[src].append((des2, distance2 + distance))

        self.callgraphEdges = linklistgraph

    def addAdjacentEdges(self, path):
        adjacentInfo = p.load(open(path,'r'))
        for idx, funcname in enumerate(adjacentInfo):
            ind1 = len(self.funcName2Ind)
            if funcname in self.funcName2Ind:
                ind1 = self.funcName2Ind[funcname]
            else:
                self.funcName2Ind[funcname] = str(ind1)
            if idx != 0:
                ind2 = self.funcName2Ind[adjacentInfo[idx - 1]]
                if str(ind1) in self.callgraphEdges:
                    self.callgraphEdges[str(ind1)].append((ind2, 1))
                else:
                    self.callgraphEdges[str(ind1)] = [(ind2, 1)]
                if str(ind2) in self.callgraphEdges:
                   self.callgraphEdges[str(ind2)].append((ind1, 1))
                else:
                   self.callgraphEdges[str(ind2)] = [(ind1, 1)]

    def loadOneBinary(self, funcnamepath, embFile):
        names = p.load(open(funcnamepath, 'r'))
        data = p.load(open(embFile, 'r'))
        self.ind2emb=dict()
        for i in range(len(names)):
            if names[i] in self.funcName2Ind:
                ind1 = self.funcName2Ind[names[i]]
                self.ind2emb[ind1]=data[i]
            # else:
            #     print names[i]

    def buildNGram(self, namPath):
        #pdb.set_trace()
        self.loadOneBinary(namPath, self.embFile)
        twoGramList = []
        linklistgraph = self.callgraphEdges
        keylist = linklistgraph.keys()
        for src in keylist:
            for (des, distance) in linklistgraph[src]:
                try:
                    #pdb.set_trace()
                    src = src.strip('\"')
                    des = des.strip('\"')
                    srcind = self.funcName2Ind[src]
                    srcemb = self.ind2emb[srcind]
                    desind = self.funcName2Ind[des]
                    desemb = self.ind2emb[desind]
                    twoGramList.append([np.concatenate((srcemb, desemb)),(src, des)])
                except:
                    #print(traceback.format_exc())
                    pass
        self.twoGramList = twoGramList

        threeGramList = []
        for src in keylist:
            for (des, distance) in linklistgraph[src]:
                if des in keylist:
                    for (des2, distance2) in linklistgraph[des]:
                        try:
                            src = src.strip('\"')
                            des = des.strip('\"')
                            des2 = des2.strip('\"')
                            srcind = self.funcName2Ind[src]
                            srcemb = self.ind2emb[srcind]
                            desind = self.funcName2Ind[des]
                            desemb = self.ind2emb[desind]
                            desind2 = self.funcName2Ind[des2]
                            desemb2 = self.ind2emb[desind2]
                            threeGramList.append([np.concatenate((srcemb, desemb, desemb2)), (src, des, des2)])
                        except:
                            #print(traceback.format_exc())
                            pass
        self.threeGramList = threeGramList
        print('n-gram loaded')
        #pdb.set_trace()


class TestBinary(Binary):
    def __init__(self, binaryName, dotPath, embFile):
        print 'init testing binary'
        #self.binaryName = binaryName
        #self.loadCallGraph(dotPath)
        #self.getGraphFromPath()
        #self.embFile = embFile

    def getRank1Neighbors(self, selectedNeighbors, funcNameList):
        print 'analyse label count'
        neighbors = dict()

        #self.neighborFunctions = selectedNeighbors
        for idx, line in enumerate(funcNameList):
            binaryname = line.rstrip().rsplit('{', 1)[0]
            funcName = line.rstrip().rsplit('{', 1)[1].split('}')[0]
            tmp = []
            for neighbor in selectedNeighbors[idx]:
                predicted_label = neighbor.rstrip().split('{')[0]
                predicted_function = neighbor.rstrip().split('{')[1].split('}')[0]
                tmp.append((predicted_label, predicted_function))
            neighbors[funcName] = tmp

        # this is for counting
        # print "predicted label"
        # count = dict()
        # for funcList in tempFuncList:
        #     for predicted_label in funcList:
        #         if predicted_label in count:
        #             count[predicted_label] += 1
        #         else:
        #             count[predicted_label] = 1
        #
        # sorted_count = sorted(count.items(), key=lambda x: x[1], reverse=True)
        # print sorted_count
        self.func2Neighbors = neighbors

    def compareSameEdges(self, library):
        edgeCount = 0
        keylist = self.callgraphEdges.keys()
        dests = set()
        srcs = set()
        for edge_src in keylist:
            for (edge_des, distance) in self.callgraphEdges[edge_src]:
                func_source = self.ind2FuncName[edge_src]
                func_destination = self.ind2FuncName[edge_des]
                # if func_source == 'ec_GFp_mont_group_clear_finish' and func_destination == 'BN_MONT_CTX_free':
                #     pdb.set_trace()
                if func_source not in self.func2Neighbors or func_destination not in self.func2Neighbors:
                    continue
                srcPredictedFunction = []
                desPredictedFunction = []
                #find whether functions in this edge have label L.libraryName or not
                for (predicted_label, predicted_function) in self.func2Neighbors[func_source]:
                    if predicted_label == library.libraryName:
                        srcPredictedFunction.append(predicted_function)
                        #break
                for (predicted_label, predicted_function) in self.func2Neighbors[func_destination]:
                    if predicted_label == library.libraryName:
                        desPredictedFunction.append(predicted_function)
                        #break
                # if func_source == 'ec_GFp_mont_group_clear_finish' and func_destination == 'BN_MONT_CTX_free':
                #     pdb.set_trace()
                #if both of them has label L.libraryName
                #if srcPredictedFunction is not [] and desPredictedFunction is not []:
                #if L also has this edge
                tempCount = 0
                for src in srcPredictedFunction:
                    if library.funcName2Ind[src] not in library.callgraphEdges:
                        continue
                    for des in desPredictedFunction:
                        if (library.funcName2Ind[des], distance) in library.callgraphEdges[library.funcName2Ind[src]]:
                            #print src, des
                            tempCount += 1
                if tempCount > 0:
                    edgeCount += 1
                    dests.add(edge_des)
                    srcs.add(edge_src)
                    #print edge_src, self.ind2FuncName[edge_src], edge_des, self.ind2FuncName[edge_des]
        # list_des = list(dests)
        # for des in list_des:
        #     if des in self.callgraphEdges:
        #         for (d, distance) in self.callgraphEdges[des]:
        #             if d not in srcs and d not in dests:
        #                 dests.remove(des)
        #                 break
        p.dump(list(srcs.union(dests)), open(library.libraryName+'.p','w'))
        print library.libraryName, edgeCount, len(srcs)+len(dests)
        return library.libraryName, edgeCount, len(srcs)+len(dests)


    def count(self, queryPath, threshold=0.999, verbose=True):
        #queryPath = 'data/versiondetect/test2/test_kNN.p'
        #query stores the query knn: each line is [query_func_name, func_name_list, similarity]
        query = p.load(open(queryPath, 'rb'))
        votes = dict()
        for i in query:
            if i[2] > threshold:
                funcList = i[1]
                #TODO: calculate IDF for each n-gram
                # if len(funcList) > 107:
                #     continue
                labels = set()
                for predicted_func in funcList:
                     binaryName = predicted_func[0]
                     version = predicted_func[1]
                     predicted_label = binaryName + '_' + version
                     labels.add(predicted_label)

                #IDF = np.log(116.0/len(labels))
                for predicted_label in labels:
                     if predicted_label in votes:
                         votes[predicted_label] += 1
                     else:
                         votes[predicted_label] = 1
                    #pdb.set_trace()
                    #try:
                    #    if type(predicted_func) is tuple:
                    #        predicted_label = predicted_func[0].split('{')[0]
                    #        labels.add(predicted_label)
                    #    elif type(predicted_func[0]) is tuple:
                    #        predicted_label = predicted_func[0][0].split('{')[0]
                    #        labels.add(predicted_label)

                    #except:
                    #    print(traceback.format_exc())
                    #    print(predicted_func)
                #pdb.set_trace()
                for predicted_label in labels:
                    if predicted_label in votes:
                        votes[predicted_label] += 1
                    else:
                        votes[predicted_label] = 1

        sorted_count = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        print(sorted_count)
        return sorted_count




    def callBP(self, queryPath_3gram, queryPath_2gram, namPath, resultsPath, threshold=0.999):
        # prepare graph [[src, des, weight],[],...]
        graph = []
        self.addAdjacentEdges(namPath)
        keylist = self.callgraphEdges.keys()
        keylist.sort()
        for edge_src in keylist:
            for (edge_des, weight) in self.callgraphEdges[edge_src]:
                graph.append([int(edge_src), int(edge_des), weight])
        #prepare labels
        query_3gram = p.load(open(queryPath_3gram, 'rb'))
        #query_2gram = p.load(open(queryPath_2gram, 'rb'))
        ind2labels = dict()
        allLabelName = set()
        for query in [query_3gram]:
            for i in query:
                if i[2] > threshold:
                    funcList = i[1]
                    labels = set()
                    for predicted_func in funcList:
                         binaryName = predicted_func[0]
                         version = predicted_func[1]
                         predicted_label = binaryName + '_' + version
                         labels.add(predicted_label)

                    # labelList = i[1]
                    functions = i[0]
                    # labels = set()
                    # for predicted_func in labelList:
                    #     try:
                    #         if type(predicted_func) is tuple:
                    #             predicted_label = predicted_func[0].split('{')[0]
                    #             labels.add(predicted_label)
                    #         elif type(predicted_func[0]) is tuple:
                    #             predicted_label = predicted_func[0][0].split('{')[0]
                    #             labels.add(predicted_label)
                    #     except:
                    #         print(traceback.format_exc())
                    #         print(predicted_func)
                    #
                    for func in functions:
                        funcname = func.split('{')[-1].split('}')[0]
                        ind = self.funcName2Ind[funcname]
                        if ind in ind2labels:
                            ind2labels[ind] = ind2labels[ind]+Counter(labels)
                        else:
                            ind2labels[ind] = Counter(labels)
                    allLabelName = allLabelName|labels
        alllabelpath = os.path.join(resultsPath, 'alllabel.p')
        allLabelList = []
        #pdb.set_trace()
        allLabelList.extend(list(allLabelName))
        p.dump(allLabelList, open(alllabelpath, 'w'))
        proirBelief = os.path.join(resultsPath, 'proirBelief.p')
        p.dump(ind2labels, open(proirBelief, 'w'))
        # call Belief Propagation
        resultsPath = os.path.join(resultsPath, 'BP_undirected.txt')
        #TODO: add weights on function labels
        runBP(graph, ind2labels, allLabelList, resultsPath, len(self.funcName2Ind))
