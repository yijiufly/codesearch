import numpy as np
import pickle as p
import pdb
import operator


class Labels:
    def __init__(self, seedlabels=None, funcNameList=None, finalLabels=None, groundTruth=None, alllabels=None):
        self.seedlabels = seedlabels  # seed labeles is a 2-d list, each line is label\tweight
        self.funcNameList = funcNameList  # 1-d list
        # final labels is a N*M np array, M is the total amount of labels
        self.finalLabels = finalLabels
        self.groundTruth = groundTruth
        self.alllabels = alllabels

    def load_funcNameList(self, funcNameList):
        self.funcNameList = funcNameList

    def load_seedlabels(self, seedlabels):
        self.seedlabels = seedlabels

    def load_finalLabels(self, finalLabels):
        self.finalLabels = finalLabels

    def load_groundTruth(self, groundTruth):
        self.groundTruth = groundTruth  # 1-d list, each element is the label for a binary

    def load_alllabels(self, alllabels):
        self.alllabels = alllabels

    def generate_label_vector(self):
        b = [x[0] for x in self.seedlabels]
        self.alllabels = list(set(b))
        return self.alllabels

    def funcNameList2seedlabels(self):
        list = []
        for line in self.funcNameList:
            line = line.rstrip().split('{')[0].rsplit('_', 1)[0]
            list.append([line, 10])
        return list

    def vector2matrix(self):
        if not hasattr(self, 'alllabels'):
            raise AttributeError('Generate alllabels first!')
        label_out = np.zeros(shape=(len(self.seedlabels), len(self.alllabels)))
        for func in self.seedlabels:
            index = self.alllabels.index(func[1])
            label_out[int(func[0])][index] = int(func[2])

        return label_out

    def finalLabelDistribution(self):
        lastline = ''
        for idx, line in enumerate(self.funcNameList):
            line = line.rstrip().split('-')[1].split('.')[0].rsplit('_', 1)[0]
            if line != lastline:
                print "\n"
                print line
            # for every function find the 3 largest non zero values
            top_3_idx = np.argsort(self.finalLabels[idx])[-3:]
            print "func:" + self.alllabels[top_3_idx[0]] + ", " + self.alllabels[top_3_idx[1]] + \
                ", " + self.alllabels[top_3_idx[2]] + \
                ", " + str(self.finalLabels[idx][top_3_idx])
            lastline = line

    def compareOutWithGroundTruth(self):
        lastline = ''
        binaryidx = 0
        count = np.zeros(shape=(1, len(self.alllabels)))
        for idx, line in enumerate(self.funcNameList):
            #line = line.rstrip().split('-')[1].split('.')[0].rsplit('_',1)[0]
            line = line.rstrip().split('{')[0]
            if line != lastline and lastline != '':
                # compare with groundTruth
                print lastline
                print "\n"
                #print "real label: " + self.groundTruth[binaryidx] + "\n"
                top_3_idx = np.argsort(count)[-3:]
                print top_3_idx
                # pdb.set_trace()
                print "predicted label" + self.alllabels[top_3_idx[0]] + ", " + self.alllabels[top_3_idx[1]
                                                                                               ] + ", " + self.alllabels[top_3_idx[2]] + ", " + str(count[top_3_idx]) + "\n"
                count = np.zeros(shape=(1, len(self.alllabels)))
                binaryidx += 1
            # for every function find the 3 largest non zero values
            top_3_idx = np.argsort(self.finalLabels[idx])[-3:]
            if self.finalLabels[idx][top_3_idx[2]] > 5 and self.finalLabels[idx][top_3_idx[0]] < 1:
                count[top_3_idx[2]] += 1
            #print "func:" + self.alllabels[top_3_idx[0]] + ", " + self.alllabels[top_3_idx[1]] + ", " + self.alllabels[top_3_idx[2]] + ", " + str(self.finalLabels[idx][top_3_idx])
            print str(self.finalLabels[idx][top_3_idx])
            lastline = line

    def analyse_naive(self):
        print 'analyse naive'
        lastline = ''
        binaryidx = 0
        count = np.zeros(shape=(len(self.alllabels)))
        labelof = p.load(
            open("data/versiondetect/test1/labelAfterFirstRound.p", "r"))
        addfuncNameList = open(
            "data/versiondetect/test1/versiondetect_addfunc_list.txt", "r").readlines()
        for i in range(len(addfuncNameList)):
            labelof.append([67])
        kNNs = p.load(open("data/versiondetect/test1/test_kNN.p", "r"))

        strong_label_count = [0 for i in self.alllabels]
        for item in labelof:
            if item != []:
                strong_label_count[item[0]] += 1

        for idx, line in enumerate(self.funcNameList):
            #line = line.rstrip().split('-')[1].split('.')[0].rsplit('_',1)[0]
            line = line.rstrip().split('{')[0].rsplit('_', 1)[0]
            #print line
            if line != lastline and lastline != '':
                # compare with groundTruth
                print lastline
                print "\n"
                #print "real label: " + self.groundTruth[binaryidx] + "\n"
                print count
                top_3_idx = np.argsort(count)[-3:]
                #print top_3_idx
                # pdb.set_trace()
                print "predicted label" + self.alllabels[top_3_idx[0]] + ", " + self.alllabels[top_3_idx[1]
                                                                                               ] + ", " + self.alllabels[top_3_idx[2]] + ", " + str(count[top_3_idx]) + "\n"
                count = np.zeros(shape=(len(self.alllabels)))
                binaryidx += 1
                labelof = p.load(
                    open("data/versiondetect/test1/labelAfterFirstRound.p", "r"))
                for i in range(len(addfuncNameList)):
                    labelof.append([67])
            # for every function find the 3 largest non zero values
            for knn in kNNs:
                if knn[2] > 0.9999 and labelof[knn[1][0]] != [] and knn[0][0] - len(labelof) == idx:
                    #print labelof[knn[1][0]][0]
                    count[labelof[knn[1][0]][0]] += 1.0 / \
                        strong_label_count[labelof[knn[1][0]][0]]
                    labelof[knn[1][0]] = []
            lastline = line

        print lastline
        print "\n"
        #print "real label: " + self.groundTruth[binaryidx] + "\n"
        print count
        top_3_idx = np.argsort(count)[-3:]
        #print top_3_idx
        # pdb.set_trace()
        print "predicted label" + self.alllabels[top_3_idx[0]] + ", " + self.alllabels[top_3_idx[1]
                                                                                       ] + ", " + self.alllabels[top_3_idx[2]] + ", " + str(count[top_3_idx]) + "\n"
        count = np.zeros(shape=(len(self.alllabels)))
        binaryidx += 1

    def analyse_labelcount(self, thresholds):
        print 'analyse label count'
        print len(thresholds)
        lastbinaryname = ''
        binaryidx = 0
        tempFuncList = []

        for idx, line in enumerate(self.funcNameList):
            binaryname = line.rstrip().split('{')[0]
            if binaryname != lastbinaryname and lastbinaryname != '':
                # count label distributions
                print "\n"
                print lastbinaryname
                print "predicted label"
                count = dict()
                for funcList in tempFuncList:
                    for predicted_label in funcList:
                        if predicted_label in count:
                            count[predicted_label] += 1
                        else:
                            count[predicted_label] = 1

                sorted_count = sorted(
                    count.items(), key=lambda x: x[1], reverse=True)
                print sorted_count

                # #exclude mutual exclusive ones
                # selected = []
                # while(sorted_count != []):
                #     selected.append(sorted_count.pop(0)[0])
                #     exclusive = set()
                #     for funcList in tempFuncList:
                #         if selected[-1] in funcList:
                #             exclusive = exclusive.union(set(funcList))
                #     #pdb.set_trace()
                #     for func in exclusive:
                #         func_count = count[func]
                #         if (func, func_count) in sorted_count:
                #             sorted_count.remove((func, func_count))
                #
                #
                # print "after exclude labels"
                # print selected

                tempFuncList = []
                binaryidx += 1

            tmp = []
            for threshold in thresholds[idx]:
                predicted_label = threshold.rstrip().split('{')[
                    0].rsplit('_', 1)[0]
                tmp.append(predicted_label)
            tempFuncList.append(tmp)

            lastbinaryname = binaryname
