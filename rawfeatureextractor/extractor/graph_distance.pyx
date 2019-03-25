import numpy as np
import math
import dlib

# if the feature is a constant, return their difference
# if the feature is a set, calculate Jaccard difference
def calcDiff(f1, f2):
	if type(f1) is list:
		same = [v for v in f1 if v in f2]
		smv = max(len(f1), len(f2))
		diff = smv - len(same)
		if smv == 0:
			return np.asarray([0, 0])
		else:
			jaccard = diff * 1.0 / smv
			return np.asarray([jaccard, 1])
	else:
		diff = math.fabs(f1 - f2)
		max_v = max(f1, f2)
	return np.asarray([diff, max_v])

# get value of feature for current node
def preCalc(src_id, dst_id, src_g, dst_g):
	try:
		src_vec = src_g.node[src_id]['v']
	except:
		src_vec = [[],[],0,0,0,0,0,0]

	try:
		dst_vec = dst_g.node[dst_id]['v']
	except:
		dst_vec = [[],[],0,0,0,0,0,0]

	return (src_vec, dst_vec)

# calc distance between two nodes
def calcNodeDistance(src_id, dst_id, src_g, dst_g, a):
	src, dst = preCalc(src_id, dst_id, src_g, dst_g)

	array = np.asarray([a[i]*calcDiff(src[i], dst[i]) for i in xrange(len(src))])
	sum_diff = sum(array[:,0])
	sum_max = sum(array[:,1])
	if sum_max == 0:
		return 0
	return 1 - sum_diff / sum_max

def calcGDistanceByNodes(t1, t2, a):
	g1 = t1.g
	g2 = t2.g
	matrix = []
	matrix_len = max(len(g1), len(g2))
	visited = {}

	for i in xrange(matrix_len):
		row = []
		for j in xrange(matrix_len):
			temp = calcNodeDistance(i, j, g1, g2, a)
			row.append(temp)
		matrix.append(row)

	mapping = dlib.max_cost_assignment(dlib.matrix(matrix))
	cost = 0
	for i in xrange(len(mapping)):
		cost += matrix[i][mapping[i]]
	return round(cost / len(mapping), 4)

def nodesMatch(t1, t2, a):
	g1 = t1.g
	g2 = t2.g
	matrix = []
	matrix_len = max(len(g1), len(g2))
	visited = {}

	for i in xrange(matrix_len):
		row = []
		for j in xrange(matrix_len):
			temp = calcNodeDistance(i, j, g1, g2, a)
			row.append(temp)
		matrix.append(row)

	mapping = dlib.max_cost_assignment(dlib.matrix(matrix))
	return mapping

def calDistanceBynumeric(cifs, cigs):
	diffs = []
	for i in xrange(len(cifs)):
		cif = cifs[i]
		cig = cigs[i]
		if type(cif) is list:
			smv = max(len(cif), len(cig))
			same = [v for v in cif if v in cig]
			diff = smv - len(same)
			if smv == 0:
				diffs.append(1)
			else:
				jaccard = 1 - diff * 1.0 / smv
				diffs.append(jaccard)
		else:
			max_n = max(cif, cig)
			if max_n == 0:
				diffs.append(1)
			else:
				diff = 1 - math.fabs(cif - cig)/max(cif, cig)
				diffs.append(diff)
	return [round(v,4) for v in diffs]

# calc distance between two graphs
def calcGDistance(t1, t2, a):
	distance = calcGDistanceByNodes(t1, t2, a)
	fun_f = calDistanceBynumeric(t1.fun_features, t2.fun_features)
	features = []
	features.append(distance)
	features += fun_f
	return features

def Euclidean(features):
	sum_value = 0 
	for v in features:
		sum_value += math.pow(v - 1,2)
	return math.sqrt(sum_value)

# calc distance between two graphs
def calcGDistanceValue(t1, t2, a):
	distance = calcGDistanceByNodes(t1, t2, a)
	fun_f = calDistanceBynumeric(t1.fun_features, t2.fun_features)
	features = []
	features.append(distance)
	features += fun_f
	#return 1/(1+Euclidean(features))
	return Euclidean(features)
