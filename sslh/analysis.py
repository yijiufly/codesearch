import numpy as np
import tables as tb
from scipy import sparse
import json
import pickle as p
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

def find_higheset_beliefs_versiondetect():
	f = open('/home/yijiufly/Downloads/codesearch/data/versiondetect/test1/train_belief_addthreshold.txt', 'r')
	belief = np.loadtxt(f)
	f.close()
	beliefAfterBP = np.zeros(shape=belief.shape)
	f = open("/home/yijiufly/Downloads/codesearch/data/versiondetect/test1/alllabels.p","r")
	labelsAll = p.load(f)
	f.close()
	try:
		f = open("/home/yijiufly/Downloads/codesearch/data/versiondetect/test1/versiondetect_func_sublist.txt", 'r')
		lines = f.readlines()
		f.close()
		lastline = ''
		ind = -1
		count = 0
		label=[]
		for idx, line in enumerate(lines):
			binaryname = line.rstrip().split('{')[0].rsplit('_',1)[0]

			if binaryname != lastline:
				print "\n"
				print binaryname
			#for every function find the 3 largest non zero values
			top_3_idx = np.argsort(belief[idx])[-3:]
			print line + ": " + labelsAll[top_3_idx[0]] + ", " + labelsAll[top_3_idx[1]] + ", " + labelsAll[top_3_idx[2]] + ", " + str(belief[idx][top_3_idx])
			lastline = binaryname
			if belief[idx][top_3_idx[2]] > 9.9 and belief[idx][top_3_idx[1]] < 0 and labelsAll[top_3_idx[2]] == binaryname:
				beliefAfterBP[idx][top_3_idx[2]] = 20
				label.append([top_3_idx[2]])
			else:
				count += 1
				label.append([])
		f = open("/home/yijiufly/Downloads/codesearch/data/versiondetect/test1/beliefAfterFirstRound.p","w")
		p.dump(beliefAfterBP, f)
		f.close()
		#f = open("/home/yijiufly/Downloads/codesearch/data/versiondetect/test1/labelAfterFirstRound.p","w")
		#p.dump(label, f)
		#f.close()
		print count

	except IOError as error:
		raise error

def get_inference_distribution(resultsPath, queryList):
	f = open(resultsPath, "r")
	results = np.loadtxt(f)
	f.close()

	f = open(queryList, "r")
	qList = [line.rstrip() for line in f]
	f.close()



if __name__ == '__main__':
	#find_higheset_beliefs()
	#load_prior_belief_RW()
	#load_adjacency_matrix_RW('mydata/funcsimilarity3.h5', 0.998)
	#find_higheset_beliefs_RW()
	find_higheset_beliefs_versiondetect()
