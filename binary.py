import pydot
import pickle as p
import pdb
class Binary:
    def __init__(self, path, path2):
        raise NotImplementedError

    def generatefuncNameFull(self, path):
        graph = pydot.graph_from_dot_file(path)
        nodeList = graph.get_node_list()
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
        graph = pydot.graph_from_dot_file(path)
        edgeList = graph.get_edge_list()
        self.callgraphEdges = []
        for edge in edgeList:
            self.callgraphEdges.append([edge.get_source(), edge.get_destination(), 1])

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


class TestBinary(Binary):
    def __init__(self, path, path2):
        print 'init testing binary'
        self.generatefuncNameFull(path)
        self.getGraphFromPath(path)
        self.generatefuncNameFilted(path2)

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
        for edge in self.callgraphEdges:
            func_source = self.ind2FuncName[edge[0]]
            func_destination = self.ind2FuncName[edge[1]]
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
                for des in desPredictedFunction:
                    if [library.funcName2Ind[src], library.funcName2Ind[des], 1] in library.callgraphEdges:
                        #print src, des
                        tempCount += 1
            if tempCount > 0:
                edgeCount += 1
                #print edge[0], self.ind2FuncName[edge[0]], edge[1], self.ind2FuncName[edge[1]]
        #print edgeCount
        return library.libraryName, edgeCount
