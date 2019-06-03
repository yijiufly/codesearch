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

def find_higheset_beliefs_versiondetect(F, allLabelName):
	allLabelName = list(allLabelName)
	for i in range(len(F)):
		top_3_idx = np.argsort(F[i])[-9:]
		if F[i][top_3_idx[-1]] > 0.005:
			print F[i][top_3_idx]
			print allLabelName[top_3_idx[0]], allLabelName[top_3_idx[1]], allLabelName[top_3_idx[2]], allLabelName[top_3_idx[3]], allLabelName[top_3_idx[4]]

def find_higheset_beliefs_versiondetect2(F, allLabelName):
	#allLabelName = list(allLabelName)
	count = dict()
	for i in range(len(F)):
		top_value = max(F[i])
		top_value_round = round(top_value, 8)
		if top_value > 0.005:
			for j in range(len(F[i])):
				round_ij = round(F[i][j], 8)
				if round_ij == top_value_round:
					label = allLabelName[j]
					if label in count:
						count[label] += 1
					else:
						count[label] = 1
	sorted_count = sorted(count.items(), key=lambda x: x[1], reverse=True)
	print sorted_count[:3]

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
	f = open('/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx/nginx-{openssl-0.9.7i}{zlib-1.2.7.3}/BP_undirected.txt','r')
	belief = np.loadtxt(f)
	alllabel = p.load(open('/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx/nginx-{openssl-0.9.7i}{zlib-1.2.7.3}/alllabel.p','r'))
	find_higheset_beliefs_versiondetect2(belief, alllabel)
