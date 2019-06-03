import numpy as np
import tables as tb
from scipy import sparse
import json
import pickle as p

def load_adjacency_matrix(adjMatPath, threshold):
	try:
		h5 = tb.open_file(adjMatPath, 'r')
		sm = h5.root.data
		am = sm[:,:]
		am[am >= threshold] = 1
		am[am < threshold] = 0
		h5.close()
		sW = sparse.csr_matrix(am)
		return sW
	except IOError as error:
		raise error

def load_adjacency_matrix_versiondetect(data, size):
	try:
		#f=open(adjMatPath,"r")
		#data = p.load(f)
		#size = data[-1][0] + 1
		b=np.array(data)
		sW = sparse.csr_matrix((b[ :, 2], (b[ :, 0], b[ :, 1])), shape=(size,size))
		return sW
	except IOError as error:
		raise error

def load_adjacency_matrix_kNN(adjMatPath, k):
	try:
		h5 = tb.open_file(adjMatPath, 'r')
		sm = h5.root.data
		am = sm[:,:]
		#am[am >= threshold] = 1
		#am[am < threshold] = 0
		h5.close()
		for idx, a in enumerate(am):
			ind = sorted(range(len(a)), key=lambda i: a[i])[:len(a)-k]
			am[idx][ind]=0
		sW = sparse.csr_matrix(am)
		return sW
	except IOError as error:
		raise error

def load_prior_belief():
	label_json = json.load(open('mydata/labels.txt','r'))
	try:
		f = open("mydata/subFuncNameListSorted.txt", 'r')
		lines = f.readlines()
		c = np.zeros(shape=(len(lines),643))
		f.close
		for idx, line in enumerate(lines):
			line = line.split('_')[0]
			if len(line) == 64:
				a = np.array([0])
				b = np.array(label_json[line])
				c[idx] = np.append(a, b)
			elif len(line) > 0:
				c[idx,0]=100
		return c
	except IOError as error:
		raise error

def load_prior_belief_versiondetect(allLabelName, seedlabeldict, size):
	try:
		c = np.full(shape=(size,len(allLabelName)), fill_value=0.1)
		#for i in c:
			#i[0]=0.5
		for key in seedlabeldict:
			for label in seedlabeldict[key]:
				index2 = allLabelName.index(label)
				c[int(key),index2] = 1#seedlabeldict[key][label]
				#c[int(key),0] = 0.1
		#print c[0]
		return c
	except IOError as error:
		raise error

def load_adjacency_matrix_RW(adjMatPath, threshold):
	try:
		h5 = tb.open_file(adjMatPath, 'r')
		sm = h5.root.data
		am = sm[:,:]
		am[am >= threshold] = 1
		am[am < threshold] = 0
		h5.close()
		sW = sparse.csr_matrix(am)
		value = sW.data
		Knz = sW.nonzero()
		sparserows = Knz[0]
		sparsecols = Knz[1]
		f = open("mydata/matrix_RW.txt","w")
		for i in range(len(sparserows)):
			f.write(str(sparserows[i]) + '\t' + str(sparsecols[i]) + '\t' + str(value[i]) + '\n')
		f.close
	except IOError as error:
		raise error

def load_prior_belief_RW():
	label_json = json.load(open('mydata/labels.txt','r'))
	try:
		f = open("mydata/subFuncNameListSorted.txt", 'r')
		lines = f.readlines()
		#c = np.zeros(shape=(len(lines),643))
		f.close
		f = open("mydata/labels_RW.txt","w")
		for idx, line in enumerate(lines):
			line = line.split('_')[0]
			if len(line) == 64:
				a = np.array(label_json[line])
				for b in np.flatnonzero(a):
					f.write(str(idx) + '\t' + str(b+1) + '\t' + str(a[b]) + '\n')
			elif len(line) > 0:
				f.write(str(idx) + '\t' + '0' + '\t' + '10' + '\n')
	except IOError as error:
		raise error

def load_adjacency_matrix_versiondetect_query(data, queryPath):
	try:
		#querykNN = p.load(open(queryPath, "r"))
		#data.append(querykNN)
		size = data[-1][0] + 1
		#data = np.concatenate((data, np.array(querykNN)))
		sW = sparse.csr_matrix((data[:][2],(data[:][0], data[:][1])), shape=(size, size))
		return sW
	except IOError as error:
		raise error
