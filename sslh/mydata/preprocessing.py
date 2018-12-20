####################################################################
## Do some preprocessing given the verbose file and filename file ##
####################################################################
from sets import Set
import numpy as np
import json
def countMalFamily():
	f = open('alljson.verbose','r')
	f_set = Set([])
	for line in f.readlines():
		dict_str = line.split('\t')[1].split('\n')[0]
		f_list = eval(dict_str)
		for f_class in f_list:
			f_set.add(f_class[0])
	f.close()
	#print f_set
	print len(f_set)
	f_list = list(f_set)
	return f_list

def getPriorBeliefVec():
	dict1={}
	malFamilyList = countMalFamily()
	f = open('MalwareFamilyList.txt', 'w')
	for (i, item) in enumerate(malFamilyList, start=0):
		f.write(str(i) + "," + item + "\n")
	f.close
	f = open('alljson.verbose','r')
	filename = open('filename.txt', 'r')
	for line in f.readlines():
		sha256 = filename.readline().split('\n')[0]
		f_str = line.split('\t')[1].split('\n')[0]
		f_list = eval(f_str)
		pbvec = np.zeros(len(malFamilyList))
		for f_class in f_list:
			index = malFamilyList.index(f_class[0])
			pbvec[index] = int(f_class[1])
		dict1[sha256] = pbvec.tolist()

	json.dump(dict1, open("labels.txt",'w'))
	f.close

if __name__ == '__main__':
	getPriorBeliefVec()
