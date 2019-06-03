"""
Test class for SSL-H inference methods (BP, linBP_directed)

Sample code to demonstrate the use of functions
This code plots the time for BP vs directed LinBP

(C) Wolfgang Gatterbauer, 2016
"""

from __future__ import division             # allow integer division
from __future__ import print_function
import numpy as np
import datetime
import random
import os                                   # for displaying created PDF
import time
from SSLH_utils import (from_dictionary_beliefs,
                        create_parameterized_H,
                        replace_fraction_of_rows,
                        to_centering_beliefs,
                        eps_convergence_directed_linbp)
from SSLH_graphGenerator import planted_distribution_model
from SSLH_inference import linBP_symmetric, linBP_directed, beliefPropagation
from my_util_load import *
from analysis import *


# -- Determine path to data *irrespective* of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
import pickle as p
current_path = dirname(abspath(getfile(currentframe())))
figure_directory = join(current_path, 'figs')
data_directory = join(current_path, 'data')

def runBP(adjMat, priorsBelieve, allLabelName, resultsPath, size):
    propagation_echo = True
    numMaxIt = 10
    convergencePercentage_W = 0.9
    pyamg = True
    scaling = 10
    s = 0.1
    h = 2
    d = 8
    eps_max_tol=1e-04

    W = load_adjacency_matrix_versiondetect(adjMat, size)
    #W=adjMat
    n1, n2 = W.shape
    #print("Graph loaded, shape = (%d, %d)\n"%(n1, n2))
    #print(str(W))
    #f = open(priorsBelievePath, 'r')
    #X0 = p.load(f)
    #f.close()
    X0 = load_prior_belief_versiondetect(allLabelName, priorsBelieve, size)
    #X0=priorsBelieve
    k = X0.shape[1]
    #print("Prior belief loaded")
    # np.savetxt(f,X0)
    H0 = create_parameterized_H(k, h, symmetric=True)
    #H0c = to_centering_beliefs(H0)
    print("\n  Start BP: ")

    # -- Propagate with LinBP
    #X2c = to_centering_beliefs(X0, ignoreZeroRows=True)
    #print("X2c:\n" + str(X2c))
    #H2c = to_centering_beliefs(H0)
    #print("H0:\n" + str(H0))
    try:
        start = time.time()
        # -- Estimate Esp_max
        #eps_max = eps_convergence_directed_linbp(P=H0, W=W, echo=propagation_echo, pyamg=pyamg, tol=eps_max_tol)
        #print(eps_max)
        eps_max=1
        eps = s * eps_max
        F, actualIt, actualPercentageConverged = \
            beliefPropagation(X0, W, H0,
                            #echo=propagation_echo,
                            numMaxIt=numMaxIt,
                            convergencePercentage=convergencePercentage_W,
                            convergenceThreshold=0.9961947,
                            debug=2)
        time_prop = time.time() - start
    except ValueError as e:
        print (
            "ERROR: {}: d={}, h={}".format(e, d, h))

    #print("F:")
    #print(F)
    print('actual iteration: %d, actual percentage converged: %f'%(actualIt, actualPercentageConverged))
    f = open(resultsPath, 'w')
    np.savetxt(f, F)
    f.close()
    find_higheset_beliefs_versiondetect2(F, allLabelName)

def runBPforQuery(adjMatPath, priorsBelievePath, resultsPath, queryPath, queryListSize, addFuncListSize):

    h = 0.3
    d = 8
    propagation_echo = True
    convergencePercentage_W = None
    numMaxIt = 10


    W = load_adjacency_matrix_versiondetect_query(adjMatPath, queryPath)
    n1, n2 = W.shape
    print("Graph loaded, shape = (%d, %d)\n"%(n1, n2))

    #f = open(priorsBelievePath, 'r')
    #X0 = np.loadtxt(f)
    #f.close()
    f = open(priorsBelievePath, 'r')
    X0 = p.load(f)
    f.close()
    X0_list = X0.tolist()
    X0_list = [x + [0.0] for x in X0_list]
    columns = len(X0_list[0])
    add = [0 for i in range(columns)]
    add[-1] = 20
    for i in range(addFuncListSize):
        X0_list.append(add)
    query = [0 for i in range(columns)]
    for i in range(queryListSize):
        X0_list.append(query)

    X0 = np.array(X0_list)
    n3, n4 = X0.shape
    print("Prior belief loaded, shape = (%d, %d)\n"%(n3, n4))
    #X0 = load_prior_belief()
    # np.savetxt(f,X0)
    H0 = create_parameterized_H(columns, h, symmetric=True)
    print("H0:\n" + str(H0))

    print("\n  Start BP: ")

    # -- Propagate with LinBP
    X2c = to_centering_beliefs(X0, ignoreZeroRows=True)
    H2c = to_centering_beliefs(H0)
    try:
        start = time.time()
        F, actualIt, actualPercentageConverged = \
            linBP_symmetric(X2c, W, H2c,
                            echo=propagation_echo,
                            numMaxIt=numMaxIt,
                            convergencePercentage=convergencePercentage_W,
                            convergenceThreshold=0.9961947,
                            debug=2)
        time_prop = time.time() - start
    except ValueError as e:
        print (
            "ERROR: {}: d={}, h={}".format(e, d, h))

    print("F:")
    print(F)
    f = open(resultsPath, 'w')
    np.savetxt(f, F)
    f.close()
    # find_higheset_beliefs_versiondetect()


if __name__ == '__main__':
    adjMatPath = '/home/yijiufly/Downloads/codesearch/junto/examples/simple/versiondetect/candidates.txt'

    #priorsBelievePath = '/home/yijiufly/Downloads/codesearch/junto/examples/simple/versiondetect/seedlabels_versiondetect_bp.p'
    #priorsBelievePath = 'mydata/beliefAfterFirstRound.txt'
    #resultsPath = '/home/yijiufly/Downloads/codesearch/sslh/mydata/beliefForALL.txt'
    #k = 108
    size = 10
    data=[[2,1,1],[1,2,1],[2,4,1],[4,2,1],[2,3,1],[3,2,1],[2,5,1],[5,2,1],[2,0,1],[0,2,1],[6,0,1],[6,0,1],[0,7,1],[7,0,1],[0,8,1],[8,0,1],[0,9,1],[9,0,1]]
    b=np.array(data)
    sW = sparse.csr_matrix((b[ :, 2], (b[ :, 0], b[ :, 1])), shape=(size,size))
    priorsBelieve = np.full(shape=(size,6),fill_value=0.1)
    priorsBelieve[1,4]=1
    #priorsBelieve[9,4]=1
    #priorsBelieve[5,5]=1
    priorsBelieve[7,5]=1
    #priorsBelieve[0,5]=1
    allLabelName=''
    resultsPath=''
    runBP(sW, priorsBelieve, allLabelName, resultsPath, size)
