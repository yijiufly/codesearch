import pydot
import pickle as p
import pdb
import numpy as np
import traceback
import os
from collections import Counter

class Binary:
    def __init__(self, path, path2):
        raise NotImplementedError

    def loadCallGraph(self, path, name_path):
        if path[-3:] == 'dot':
            self.graph = pydot.graph_from_dot_file(path)
        else:
            self.calledge = p.load(open(path, 'r'))
        print('graph loaded')
        funcNameList = p.load(open(name_path, 'r'))
        func2ind = dict()
        ind2func = dict()
        #load all the nodes in the callgraph
        for i, (node, size) in enumerate(funcNameList):
            func2ind[node] = i
            ind2func[i] = node

        self.funcName2Ind = func2ind
        self.ind2FuncName = ind2func

    def generatefuncNameFilted(self, path, filterSize=0):
        func2ind = dict()
        funcNameList = p.load(open(path, 'r'))
        funcName2size = {i[0]:i[1] for i in funcNameList}
        for func in self.funcName2Ind.keys():
            if func in funcName2size and funcName2size[func] > filterSize:
                func2ind[func] = self.funcName2Ind[func]
            else:
                func2ind[func] = -1
        self.funcNameFilted = func2ind

    def getGraphFromPath(self):
        edgeList = self.graph.get_edge_list()
        linklistgraph = dict()
        smallnodes = dict()
        #self.callgraphEdges = []
        for edge in edgeList:
            src = edge.get_source().strip('\"')
            des = edge.get_destination().strip('\"')
            # if self.funcNameFilted[src] == -1:
            #     if src in smallnodes:
            #         smallnodes[src].append((des, 1))
            #     else:
            #         smallnodes[src] = [(des, 1)]
            # else:
            if src in linklistgraph:
                linklistgraph[src].append((des, 1))
            else:
                linklistgraph[src] = [(des, 1)]
            # if des in linklistgraph:
            #    linklistgraph[des].append((src, 1))
            # else:
            #    linklistgraph[des] = [(src, 1)]
        print('edges loaded')
        keylist = linklistgraph.keys()
        findsmallnodes = -1
        while findsmallnodes != 0:
            findsmallnodes = 0
            for src in keylist:
                for (des, distance) in linklistgraph[src]:
                    if self.funcNameFilted[des] == -1:
                        #findsmallnodes += 1
                        if des in linklistgraph:
                            continue
                        else:
                            linklistgraph[src].remove((des, distance))
                        # linklistgraph[src].remove((des, distance))
                        # if des in smallnodes:
                        #     for (des2, distance2) in smallnodes[des]:
                        #         if (des2, distance2 + distance) not in linklistgraph[src]:
                        #             linklistgraph[src].append((des2, distance2 + distance))
                        # else:
                        #     print des

        self.callgraphEdges = linklistgraph

    def getGraphFromPathfilted(self):
        if hasattr(self, 'graph'):
            edgeList = self.graph.get_edge_list()
        else:
            edgeList = self.calledge
        linklistgraph = dict()
        smallnodes = dict()
        #self.callgraphEdges = []
        for edge in edgeList:
            if type(edge) == type([]):
                src = edge[0].strip('\"')
                des = edge[1].strip('\"')
            else:
                src = edge.get_source().strip('\"')
                des = edge.get_destination().strip('\"')
            if src not in self.funcNameFilted or des not in self.funcNameFilted:
                continue
            if self.funcNameFilted[src] == -1:
                if src in smallnodes:
                    smallnodes[src].append((des, 1))
                else:
                    smallnodes[src] = [(des, 1)]
            else:
                if src in linklistgraph:
                    linklistgraph[src].append((des, 1))
                else:
                    linklistgraph[src] = [(des, 1)]
            # if des in linklistgraph:
            #    linklistgraph[des].append((src, 1))
            # else:
            #    linklistgraph[des] = [(src, 1)]
        print('edges loaded')
        keylist = linklistgraph.keys()
        findsmallnodes = -1
        while findsmallnodes != 0:
            findsmallnodes = 0
            for src in keylist:
                for (des, distance) in linklistgraph[src]:
                    if self.funcNameFilted[des] == -1:
                        findsmallnodes += 1
                        linklistgraph[src].remove((des, distance))
                        if des in smallnodes:
                            for (des2, distance2) in smallnodes[des]:
                                if (des2, distance2 + distance) not in linklistgraph[src]:
                                    linklistgraph[src].append((des2, distance2 + distance))
                        # else:
                        #     print des

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
        #self.ind2emb=dict()
        self.funcName2emb=dict()
        for i in range(len(names)):
            self.funcName2emb[names[i][0]]=data[i]


    def buildNGram(self, namPath, embFile):
        #pdb.set_trace()
        self.loadOneBinary(namPath, embFile)
        twoGramList = []
        linklistgraph = self.callgraphEdges
        keylist = linklistgraph.keys()
        for src in keylist:
            for (des, distance) in linklistgraph[src]:
                try:
                    #pdb.set_trace()
                    src = src.strip('\"')
                    des = des.strip('\"')
                    srcemb = self.funcName2emb[src]
                    desemb = self.funcName2emb[des]
                    twoGramList.append([np.concatenate((srcemb, desemb)),(src, des), distance])
                except:
                    print(traceback.format_exc())
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
                            srcemb = self.funcName2emb[src]
                            desemb = self.funcName2emb[des]
                            desemb2 = self.funcName2emb[des2]
                            threeGramList.append([np.concatenate((srcemb, desemb, desemb2)), (src, des, des2), distance + distance2])
                        except:
                            print(traceback.format_exc())
                            pass
        self.threeGramList = threeGramList
        print('n-gram loaded')
        #pdb.set_trace()

