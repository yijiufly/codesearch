import pydot
import pickle as p
import pdb
import numpy as np
import traceback
class Binary:
    def __init__(self, path, path2):
        raise NotImplementedError

    def generatefuncNameFull(self, path):
        self.graph = pydot.graph_from_dot_file(path)
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

    def generatefuncNameFilted(self, path):
        func2ind = dict()
        funcNameList = p.load(open(path, 'r'))
        for func in self.funcName2Ind.keys():
            if func in funcNameList:
                func2ind[func] = self.funcName2Ind[func]
            else:
                func2ind[func] = -1
        self.funcNameFilted = func2ind

    def getGraphFromPath(self, path):
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
            ind1 = len(self.funcNameFull)
            if funcname in self.funcNameFull:
                ind1 = self.funcNameFull[funcname]
            else:
                self.funcNameFull[funcname] = str(ind1)
            if idx != 0:
                ind2 = self.funcNameFull[adjacentInfo[idx - 1]]
                self.callgraphEdges.append([int(ind1), int(ind2), 1])
                self.callgraphEdges.append([int(ind2), int(ind1), 1])

    def buildNGram(self, namPath):
        #pdb.set_trace()
        funcembs, namelist = loadOneQueryBinary(namPath, self.funcembFolder)
        twoGramList = []
        linklistgraph = self.callgraphEdges
        keylist = linklistgraph.keys()
        for src in keylist:
            for (des, distance) in linklistgraph[src]:
                try:
                    #pdb.set_trace()
                    srcname = self.binaryName + '{' + self.ind2FuncName[src] + '}.emb'
                    srcemb = funcembs[namelist.index(srcname)]
                    desname = self.binaryName + '{' + self.ind2FuncName[des] + '}.emb'
                    desemb = funcembs[namelist.index(desname)]
                    twoGramList.append([np.concatenate((srcemb, desemb)),(srcname, desname)])

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
                            srcname = self.binaryName + '{' + self.ind2FuncName[src] + '}.emb'
                            srcemb = funcembs[namelist.index(srcname)]
                            desname = self.binaryName + '{' + self.ind2FuncName[des] + '}.emb'
                            desemb = funcembs[namelist.index(desname)]
                            desname2 = self.binaryName + '{' + self.ind2FuncName[des2] + '}.emb'
                            desemb2 = funcembs[namelist.index(desname2)]
                            threeGramList.append([np.concatenate((srcemb, desemb, desemb2)), (srcname, desname, desname2)])
                        except:
                            #print(traceback.format_exc())
                            pass
        self.threeGramList = threeGramList


class TestBinary(Binary):
    def __init__(self, binaryName, dotPath, funcembFolder):
        print 'init testing binary'
        self.binaryName = binaryName
        self.generatefuncNameFull(dotPath)
        self.getGraphFromPath(dotPath)
        self.funcembFolder = funcembFolder

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

    ## for each function, choose similar functions according to the similarity distribution of candidiate similar functions
    ## choose the cluster of functions with the highest similarity scores
    def count(self, queryPath, threshold=0.999, verbose=True):
        #queryPath = 'data/versiondetect/test2/test_kNN.p'
        #query stores the query knn: each line is [query_func_name, func_name_list, similarity]
        query = p.load(open(queryPath, 'rb'))
        votes = dict()
        for i in query:
            if i[2] > threshold:
                funcList = i[1]
                #TODO: calculate IDF for each n-gram
                if len(funcList) > 107:
                    continue
                for predicted_func in funcList:
                    try:
                        if type(predicted_func) is tuple:
                            predicted_label = predicted_func[0].split('{')[0]
                            if predicted_label in votes:
                                votes[predicted_label] += 1
                            else:
                                votes[predicted_label] = 1
                        elif type(predicted_func[0]) is tuple:
                            predicted_label = predicted_func[0][0].split('{')[0]
                            if predicted_label in votes:
                                votes[predicted_label] += 1
                            else:
                                votes[predicted_label] = 1
                    except:
                        print(traceback.format_exc())
                        print(predicted_func)

        sorted_count = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        return sorted_count

def calculateIDF():
    pass

def loadOneQueryBinary(funcnamepath, embpath):
    names = p.load(open(funcnamepath, 'r'))
    file11 = []
    embnames = []
    for i in range(len(names)):
        embname = funcnamepath.split('/')[-1][:-8] + "{" + names[i] + "}.emb"
        PATH = embpath + '/' + embname
        f1 = open(PATH, 'rb')
        file1 = p.load(f1).tolist()
        file11.append(file1)
        embnames.append(embname)
        f1.close()
    data = np.array(file11)
    return data, embnames
