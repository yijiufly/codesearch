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
from SSLH_inference import linBP_symmetric, beliefPropagation
from my_util_load import *
from analysis import *


# -- Determine path to data *irrespective* of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
import pickle as p
current_path = dirname(abspath(getfile(currentframe())))
figure_directory = join(current_path, 'figs')
data_directory = join(current_path, 'data')

def runBP(adjMatPath, priorsBelievePath, allLabelNamePath, resultsPath):
    propagation_echo = True
    numMaxIt = 20
    convergencePercentage_W = None
    h = 0.3
    d = 8

    W = load_adjacency_matrix_versiondetect(adjMatPath)
    n1, n2 = W.shape
    print("Graph loaded, shape = (%d, %d)\n"%(n1, n2))
    #f = open(priorsBelievePath, 'r')
    #X0 = p.load(f)
    #f.close()
    X0 = load_prior_belief_versiondetect(allLabelNamePath, priorsBelievePath)
    k = X0.shape[1]
    print("Prior belief loaded")
    # np.savetxt(f,X0)
    H0 = create_parameterized_H(k, h, symmetric=True)
    #H0c = to_centering_beliefs(H0)
    print("\n  Start BP: ")

    # -- Propagate with LinBP
    X2c = to_centering_beliefs(X0, ignoreZeroRows=True)
    print("X2c:\n" + str(X2c))
    H2c = to_centering_beliefs(H0)
    print("H0:\n" + str(H0))
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
    print('actual iteration: %d, actual percentage converged: %f'%(actualIt, actualPercentageConverged))
    f = open(resultsPath, 'w')
    np.savetxt(f, F)
    f.close()
    #find_higheset_beliefs_versiondetect(F)

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
    priorsBelievePath = 'mydata/beliefAfterFirstRound.txt'
    resultsPath = '/home/yijiufly/Downloads/codesearch/sslh/mydata/beliefForALL.txt'
    k = 108
    runBP(adjMatPath, priorsBelievePath, resultsPath, k)
