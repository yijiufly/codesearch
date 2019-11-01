from srwr.srwr import SRWR
import pickle as p
from collections import defaultdict

def generateGlobalDict(queryPath_2gram, folder):
    global LBP_MAX_ITERS
    global THRES
    threshold=THRES
    # Make an empty graph
    g = fg.Graph()
    # build global dict according to candidate 2-gram
    # global dict stores the candidate version and function for each function in testing binary
    global_dict = defaultdict(list)
    if type(queryPath_2gram) == type(''):
        query_2gram = p.load(open(queryPath_2gram, 'rb'))
    else:
        query_2gram = queryPath_2gram
    hasDistance = False
    if type(query_2gram[0][0][0]) == type((1,2)):
        hasDistance = True

    # first identify the name that are very sure
    sure_dict = defaultdict(list)
    for i in query_2gram:
        if i[2] > 0.999:
            if not hasDistance:
                test_src = i[0][0]
                test_des = i[0][1]
            else:
                test_src = i[0][0][0]
                test_des = i[0][0][1]
            if test_src == test_des:
                continue

            funcList = i[1]
            # if len(funcList) > 200:
            #     continue
            for predicted_func in funcList:
                if hasDistance:
                    querydistance = i[0][1]
                    resultdistance = predicted_func[3]
                    if querydistance != resultdistance:
                        continue

                src_funcname = predicted_func[2][0]
                des_funcname = predicted_func[2][1]
                libraryname = predicted_func[0]
                # find the src function if it's already in the dict,
                # otherwise, add it to the dict
                if (src_funcname, libraryname) not in sure_dict[test_src]:
                    sure_dict[test_src].append((src_funcname, libraryname))
                    global_dict[test_src].append((src_funcname, libraryname))
                # do the same thing to des function
                if (des_funcname, libraryname) not in sure_dict[test_des]:
                    sure_dict[test_des].append((des_funcname, libraryname))
                    global_dict[test_des].append((des_funcname, libraryname))
    p.dump(sure_dict, open('sure_dict','w'))

    # add other labels, if it contradict to the very sure labels, don't add it
    for i in query_2gram:
        if i[2] > threshold:
            if not hasDistance:
                test_src = i[0][0]
                test_des = i[0][1]
            else:
                test_src = i[0][0][0]
                test_des = i[0][0][1]
            if test_src == test_des:
                continue

            funcList = i[1]
            # if len(funcList) > 200:
            #     continue
            for predicted_func in funcList:
                if hasDistance:
                    querydistance = i[0][1]
                    resultdistance = predicted_func[3]
                    if querydistance != resultdistance:
                        continue

                src_funcname = predicted_func[2][0]
                des_funcname = predicted_func[2][1]
                libraryname = predicted_func[0]

                if test_src in sure_dict and (src_funcname, libraryname) not in sure_dict[test_src]:
                    continue
                if test_des in sure_dict and (des_funcname, libraryname) not in sure_dict[test_des]:
                    continue
                # find the src function if it's already in the dict,
                # otherwise, add it to the dict
                if (src_funcname, libraryname) not in global_dict[test_src]:
                    global_dict[test_src].append((src_funcname, libraryname))
                # do the same thing to des function
                if (des_funcname, libraryname) not in global_dict[test_des]:
                    global_dict[test_des].append((des_funcname, libraryname))

def generateTSVfromDict(global_dict, query_2gram):
    namelist = []
    output_str = ''
    name_dict = defaultdict(int)
    contradictory_dict = defaultdict(list)
    threshold=THRES
    hasDistance = False
    if type(query_2gram[0][0][0]) == type((1,2)):
        hasDistance = True
    # add functions as random variables
    for key in global_dict:
    	idx_begin = len(namelist)
    	idx_end = idx_begin + len(global_dict[key])
    	#for ((predict, libraryname), score) in global_dict[key]:
    	for (predict, libraryname) in global_dict[key]:
    		name = key + '##' + predict
    		name = key + '##' + predict
    		name_dict[name] = len(namelist)
    		contradictory_dict[predict].append(len(namelist))
    		namelist.append(name)
    		

    	for i in range(idx_begin, idx_end):
    		for j in range(i + 1, idx_end):
    			output_str += str(i) + '\t' + str(j) + '\t-1\n'
    #p.dump(global_dict, open('global_dict','w'))
    for key in contradictory_dict:
    	contradicts = contradictory_dict[key]
    	if len(contradicts) > 1:
    		for i in range(len(contradicts)):
    			for j in range(i + 1, len(contradicts)):
    				output_str += str(contradicts[i]) + '\t' + str(contradicts[j]) + '\t-1\n'


    for i in query_2gram:
        if i[2] > threshold:
            if not hasDistance:
                test_src = i[0][0]
                test_des = i[0][1]
            else:
                test_src = i[0][0][0]
                test_des = i[0][0][1]
            if test_src == test_des:
                continue
            # if len(i[1]) > 200:
            #     continue
            if test_src not in global_dict or test_des not in global_dict:
                continue
          
            src_list = global_dict[test_src]
            des_list = global_dict[test_des]
            funcList = i[1]
            
            for predicted_func in funcList:
                if hasDistance:
                    querydistance = i[0][1]
                    resultdistance = predicted_func[3]
                    if querydistance != resultdistance:
                        continue

                src = predicted_func[2][0]
                des = predicted_func[2][1]
                libraryName = predicted_func[0]
                if test_src + '##' + src in namelist and test_des + '##' + des in namelist:
                	src_ind = namelist.index(test_src + '##' + src)
                	des_ind = namelist.index(test_des + '##' + des)
                	output_str += str(src_ind) + '\t' + str(des_ind) + '\t1\n'
                else:
                    continue

    with open('data/functioncallgraph.tsv', 'w') as f:
    	f.write(output_str)
   	p.dump(namelist, open('namelist', 'w'))
    #return output_str

def analyse(namelist, rd):
	tp = 0
	fn = 0
	fp = 0
	tn = 0
	positive_predict =set()
	for i, d in enumerate(rd):
		name = namelist[i].split('##')[0]
		predict = namelist[i].split('##')[1]
		if d > 0:
			if name == predict:
				tp += 1
			else:
				fp += 1
			positive_predict.add(name)
		else:
			if name == predict:
				fn += 1
			else:
				tn += 1
	print 'tp, fp, fn, tn'
	print tp, fp, fn, tn
	print len(positive_predict)

if __name__ == '__main__':
	global THRES
	THRES = 0.9
	global_dict = p.load(open('global_dict', 'r'))
	query_2gram = p.load(open('test_kNN_0924_2gram.p', 'r'))
	generateTSVfromDict(global_dict, query_2gram)
	srwr = SRWR()
	input_path = 'data/functioncallgraph.tsv'
	srwr.read_graph(input_path)
	srwr.normalize()
	seed = 2
	rd, rp, rn, residuals = srwr.query(seed)
	p.dump(rd, open('rd', 'w'))
	rd = p.load(open('rd', 'r'))
	namelist = p.load(open('namelist', 'r'))
	analyse(namelist, rd)
