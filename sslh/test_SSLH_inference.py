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
from my_util import load_adjacency_matrix, load_prior_belief, check_accurcy, find_higheset_beliefs




# -- Determine path to data *irrespective* of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
figure_directory = join(current_path, 'figs')
data_directory = join(current_path, 'data')



# -- Setup
# Since graph creation takes most time, especially for large graphs, saves graphs to a file format, then loads them later again.
# To save time, change to CREATE_DATA = ADD_DATA = CREATE_GRAPH = False after the first iteration
CHOICE = 1
CREATE_DATA = True
ADD_DATA = True
CREATE_GRAPH = False
SHOW_FIG = True
SHOW_GRAPH_TIME = False

similarityMatPath = 'mydata/funcsimilarity3.txt'
adjMatPath = 'mydata/funcsimilarity3.h5'
priorsBelievePath = 'mydata/priorBeliefs.txt'
header = ['n',
          'type',
          'time',]



# -- Default Graph parameters
distribution = 'powerlaw'
exponent = -0.3
k = 643
a = 1
err = 0
avoidNeighbors = False
convergencePercentage_W = None
f = 0.1
# propagation
pyamg = True
propagation_echo = True
scaling = 10
alpha = 0
beta = 0
gamma = 0
s = 0.1
numMaxIt = 10



# -- Main Options
if CHOICE == 1:
    n_vec = [100
             ]
    repeat_vec = [1]
    eps_max_tol = 1e-02
    h = 0.5
    d = 8

else:
    raise Warning("Incorrect choice!")

alpha0 = np.array([a, 1., 1.])
alpha0 = alpha0 / np.sum(alpha0)
H0 = create_parameterized_H(k, h, symmetric=True)
H0c = to_centering_beliefs(H0)

print(H0)
print(H0c)
RANDOMSEED = None  # For repeatability
random.seed(RANDOMSEED)  # seeds some other python random generator
np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed
print("CHOICE: {}".format(CHOICE))



# -- Create data
if CREATE_DATA or ADD_DATA:

    for i, n in enumerate(n_vec):
        print("\nn: {}".format(n))
        repeat = repeat_vec[i]


            #W, _ = load_W(join(data_directory, '{}_{}_W.csv'.format(filename, n)), skiprows=1, zeroindexing=True, n=None, doubleUndirected=False)
            #X0, _, _ = load_X(join(data_directory, '{}_{}_X.csv'.format(filename, n)), n=None, k=None, skiprows=1, zeroindexing=True)
        W = load_adjacency_matrix(adjMatPath, 0.998)
        f = open(priorsBelievePath, 'w')
        X0 = load_prior_belief()
        np.savetxt(f,X0)
        # -- Repeat loop
        for i in range(repeat):
            print("\n  repeat: {}".format(i))

            #X2, ind = replace_fraction_of_rows(X0, 1-f, avoidNeighbors=avoidNeighbors, W=W)
            #print("X2:")
            #print(X2)
            # -- Estimate Esp_max
            start = time.time()
            #eps_max = eps_convergence_directed_linbp(P=H0, W=W, echo=propagation_echo, pyamg=pyamg, tol=eps_max_tol)
            time_eps_max = time.time() - start
            #save_tuple(n, 'eps_max', time_eps_max)
            #eps = s * eps_max

            # -- Propagate with LinBP
            X2c = to_centering_beliefs(X0, ignoreZeroRows=True)
            print("X2c:")
            print(X2c)
            print("w:")
            print(W)
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
            else:
                #save_tuple(n, 'prop', time_prop)
                print("F:")
                print(F)
                f = open("mydata/belief.txt", 'w')
                np.savetxt(f,F)
                check_accurcy(F)
#             # -- Propagate with BP
#             if n < 5e5:
#                 try:
#                     start = time.time()
#                     F, actualIt, actualPercentageConverged = \
#                         beliefPropagation(X2, W, H0 ** eps,
#                                        numMaxIt=numMaxIt,
#                                        convergencePercentage=convergencePercentage_W,
#                                        convergenceThreshold=0.9961947,
#                                        debug=2)
#                     time_bp = time.time() - start
#                 except ValueError as e:
#                     print(
#                         "ERROR: {}: d={}, h={}".format(e, d, h))
#                 else:
#                     save_tuple(n, 'bp', time_bp)


# # -- Read, aggregate, and pivot data for all options
# df1 = pd.read_csv(join(data_directory, csv_filename))


# # Aggregate repetitions
# df2 = df1.groupby(['n', 'type']).agg \
#     ({'time': [np.mean, np.median, np.std, np.size],  # Multiple Aggregates
#       })
# df2.columns = ['_'.join(col).strip() for col in df2.columns.values]  # flatten the column hierarchy
# df2.reset_index(inplace=True)  # remove the index hierarchy
# df2.rename(columns={'time_size': 'count'}, inplace=True)
# print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(15)))

# # Pivot table
# df3 = pd.pivot_table(df2, index=['n'], columns=['type'], values=['time_mean', 'time_std'] )  # Pivot
# df3.columns = ['_'.join(col).strip() for col in df3.columns.values]  # flatten the column hierarchy
# df3.reset_index(inplace=True)  # remove the index hierarchy
# print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))

# # Extract values
# X = df3['n'].values     # plot x values
# X = X*d                 # calculate edges (notice that for symmetric we would have to divide by 2 as one edge woudl appear twice in symmetric adjacency matrix)
# if SHOW_GRAPH_TIME:
#     Y_graph = df3['time_mean_graph'].values
# Y_eps_max = df3['time_mean_eps_max'].values
# Y_prop = df3['time_mean_prop'].values
# Y_bp = df3['time_mean_bp'].values
