import numpy as np
import tables as tb
import math
import pickle as p

adjMatPath = 'data/funcsimilarity.h5'
priorsPath = 'data/priorBeliefs.p'
priorsPath2 = 'data/priorBeliefs.txt'
priorsPath3 = 'data/hidePriorBeliefs.txt'
outFILE = 'data/out'
threshold = 0.8
knownrate = 0.2

def load_adjacency_matrix():
	h5 = tb.open_file(adjMatPath, 'r')
	sm = h5.root.data
	am = sm[:,:]
	am[am >= threshold] = 1
	am[am < threshold] = 0
	h5.close()
	return am

#Range of initial labels  
#0: unknown label
#1: GOOD (e.g., not spam)
#2: BAD  (e.g., spam)  
def load_prior_beliefs():
	f = open(priorsPath, 'rb')
	belief = p.load(f)
	belief[belief==2]=-1
	f.close
	
	randomUnknownIndex = np.random.randint(len(belief),size=int(len(belief)*(1-knownrate)))
	belief[randomUnknownIndex] = 0
	f = open(priorsPath3, 'w')
	np.savetxt(f,belief)
	f.close
	return belief

def check_accuracy():
	f = open(priorsPath2, 'r')
	f2 = open(outFILE,'r')
	cnt=0
	for lines in f.readlines():
		line=f2.readline()
		if line != lines:
			cnt+=1
	print cnt

if __name__ == '__main__':

	print('* * * * * * * * * * * * * * * * * * * * * * * * *')
	print('*                                               *')
	print('*   FaBP(): Execution started...                *')
	print('*                                               *')

	# Initialize some constants
	max_power = 50          # maximum number of powers in power method
	epsilon = 10**(-14)		#stopping criterion for power method

	# Initialize about-half quantities
	prob_good = 0.5001 # not AI
	prob_good_h = prob_good - 0.5

	#load adjacency matrix, degree-diagonal matrix and identity matrix
	A = load_adjacency_matrix()
	no_nodes = len(A)
	degrees = A.sum(1)
	D = np.diag(degrees)
	c1 = np.trace(D)+2;
	c2 = np.trace(D**2) - 1;
	appropriate_hh_l1 = 1/(2*D.max()+1);
	appropriate_hh_l2 = math.sqrt((-c1 + math.sqrt(c1**2+4*c2))/(8*c2));
	appropriate_hh = min(appropriate_hh_l1, appropriate_hh_l2);
	h_h = 0.999*appropriate_hh;

	# Definitions of  a_h and c_h
	ah = 4*h_h**2 /(1-4*h_h**2)
	ch = 2*h_h / (1-4*h_h**2)

	inv_=load_prior_beliefs()
	priors = inv_ * prob_good_h


	# LINEAR SYSTEM: Fast Belief Propagation (FaBP)
	M = ch*A - ah*D

	# Calculate the inverse of matrix I-M using the power method.
	# (I-M)*b = phi => b = I*phi + M*phi + M*(M*phi) + M*(M*(M*phi)) + ...
	
	mat_ = np.dot(M, inv_)
	power = 1

	while mat_.max() > epsilon and power < max_power:
		print mat_.max()
		inv_ = inv_ + mat_
		mat_ = np.dot(M, mat_)
		power = power + 1

	if power == max_power:
		print('!! NO CONVERGENCE !!')


	beliefs = inv_


	# Final node labels
	
	final_labels = beliefs.copy()
	final_labels[final_labels < 0] = -1
	final_labels[final_labels > 0] = 1
	print final_labels
	# SAVE THE RESULTS
	#file = open(outFILE,'wb')
	#p.dump(final_labels, file)
	#file.close
	np.savetxt(outFILE,final_labels)

	check_accuracy()