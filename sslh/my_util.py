import numpy as np
import tables as tb
from scipy import sparse
import json

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

###check accuracy according to the assumption:
###every function in a benign sample is benign, and
###at least one function in a malicious sample is malicious
def check_accurcy(belief):
	try:
		f = open("mydata/subFuncNameListSorted.txt", 'r')
		progidx = []
		progisbenign = []
		lastline=''
		mistake1 = 0	#the number of malicious functions in benign samples
		mistake2 = 0	#the number of malicious binaries that has no malicious functions
		lines = f.readlines()
		#find program list
		for idx, line in enumerate(lines):
			line = line.split('_')[0]
			if line != lastline:
				if len(lastline) == 64:
					progisbenign.append(False)
					progidx.append(idx - 1)
				elif len(lastline) > 0:
					progisbenign.append(True)
					progidx.append(idx - 1)
				lastline = line
		#cal false positive and truth positive
		begin = 0
		for i in xrange(len(progidx)):
			end = progidx[i]
			if progisbenign[i]==True:
				for j in xrange(begin, end):
					ind_func = np.unravel_index(np.argmax(belief[j], axis=None), belief[j].shape)
					if ind_func[0] != 0:
						mistake1 += 1
			else:
				cnt = 0
				for j in xrange(begin, end):
					ind_func = np.unravel_index(np.argmax(belief[j], axis=None), belief[j].shape)
					if ind_func[0] != 0:
						cnt += 1
				if cnt == 0:
					mistake2 += 1
			begin = progidx[i] + 1
		print('mistake1 and mistake2')
		print(mistake1)
		print(mistake2)

	except IOError as error:
		raise error

def find_higheset_beliefs():
	f = open("mydata/belief.txt", 'r')
	belief = np.loadtxt(f)
	f.close
	label_json = json.load(open('mydata/labels.txt','r'))
	try:
		f = open("mydata/subFuncNameListSorted.txt", 'r')
		lines = f.readlines()
		lastline = ''
		ind = -1
		count = 0
		for idx, line in enumerate(lines):
			line = line.split('_')[0]

			if line != lastline:
				if len(line) == 64:
					a = np.array(label_json[line])
					ind = np.unravel_index(np.argmax(a, axis=None), a.shape)[0]
				else:
					ind = 0
				print "\n"
				print line + ": " + str(ind) + ", " + str(a[ind])
			#for every function find the 3 largest non zero values
			top_3_idx = np.argsort(belief[idx])[-3:]
			print "func:" + str(top_3_idx) + ", " + str(belief[idx][top_3_idx])
			lastline = line
			top_1_idx = top_3_idx[2]
			if ind + 1 == top_1_idx and label_json[line][ind] + 0.3 < belief[idx][top_1_idx]:
				count += 1

		print count
	except IOError as error:
		raise error

def find_higheset_beliefs_RW():
	f = open("mydata/label_prop_output", 'r')
	belief = f.readlines()
	f.close
	beliefsorted=[]
	for line in belief:
		line = line.split('\t')
		temp = line[1] + '\t' + line[3]
		list_tmp=[0]
		list_tmp[0] = int(line[0])
		list_tmp.append(temp.split('\t'))
		#print list_tmp
		beliefsorted.append(list_tmp)
	beliefsorted.sort()
	try:
		f = open("mydata/subFuncNameListSorted.txt", 'r')
		lines = f.readlines()
		lastline = ''
		for idx, line in enumerate(lines):
			line = line.split('_')[0]
			if line != lastline:
				print "\n"
				print line
			print "func:" + str(beliefsorted[idx])
			lastline = line
	except IOError as error:
		raise error

if __name__ == '__main__':
	find_higheset_beliefs()
	#load_prior_belief_RW()
	#load_adjacency_matrix_RW('mydata/funcsimilarity3.h5', 0.998)
	#find_higheset_beliefs_RW()
