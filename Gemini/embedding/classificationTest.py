'''
classification on embedding
'''
import os
import pickle as p
import numpy  as np

from ssdeepTest import loadFiles, genPCA

### define path
EMBPATH = "/home/yzheng04/Downloads/embs/"

def main():

    emb    = loadFiles(EMBPATH, ".emb")

    labels = np.zeros(len(emb))     ## intialize all zero labels array
    filenameList = []
    testList = []
    for i, filenames in enumerate(emb):
        ## generate filenameList
        filenameList.append(filenames)

        ## generate labels
        if len(filenames) == 72:
            labels[i] = 1 ## set 1 for malicious

    testList = filenameList[:10000]
    # print len(testList)

        # if i == 4903:
        #     print filenames
        #     ## generate data
        #     input = EMBPATH + filenames
        #     # print input
        #     f = open(input, 'rb')
        #     ## load data
        #     data = p.load(f)
        #     ## convert to np array
        #     data = np.array(data)
        #     print data
        #     break

            # print len(filenameList)
    # PCAResults = genPCA(testList, EMBPATH)

    rmlist, results = genPCA(testList, EMBPATH)

    # print rmlist
    # print len(rmlist)

    remove_indices = rmlist
    newfilenameList = [i for j, i in enumerate(testList) if j not in remove_indices]
    ### save newnamelist
    newNamelist = "newNameList" + ".p"
    file1 = open(newNamelist ,'wb')
    p.dump(newfilenameList, file1)
    file1.close

    rmlist1, results1 = genPCA(newfilenameList, EMBPATH)

    # save results from PCA
    pcaResults = "PCAresults" + ".emb"
    file = open(pcaResults ,'wb')
    p.dump(results1, file)
    file.close
    print rmlist1


    # # load test list
    # f = open("PCAresults.emb", 'rb')
    # ## load data
    # data = p.load(f)
    # ## convert to np array
    # data = np.array(data)
    # f.close
    # # print results1.shape
    # print data.shape
    # print len(data[1])
    # print len(data[2])
    # print data.T.shape

    # if filenames[0] == 0:
    #     print data
    #     print data.shape
    #     print PCAResults
    #     print PCAResults.shape
    #     break
    # print labels
    # print labels.shape



if __name__ == '__main__':
	main()
