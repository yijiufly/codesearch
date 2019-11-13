import binaryninja as binja
import glob
from collections import defaultdict
import mongowrapper.MongoWrapper as mdb
import os
import pdb
import numpy as np
import pickle as p
import factorgraph as fg
from simanneal import Annealer
import random
import scipy.stats
def F(beta, precision, recall):
    return (beta*beta + 1)*precision*recall / (beta*beta*precision + recall)
class FindBestParameters(Annealer):
    def __init__(self, state):
        self.mu = 10
        super(FindBestParameters, self).__init__(state)

    def move(self):
        """Swaps two cities in the route."""
        old_positive = self.state['positive']
        old_negative = self.state['negative']
        self.mu -= 0.1
        if self.mu <= 0:
            self.mu = 0.1
        a = random.uniform(max(0, old_positive - self.mu), min(10, old_positive + self.mu))
        b = random.uniform(max(0, old_negative - self.mu), min(10, old_negative + self.mu))
        self.state['positive'], self.state['negative'] = a, b

    def energy(self):
        """Calculates the length of the route."""
        print self.state['positive'], self.state['negative']
        g = BP_with_strings(knn_all, knn, string_table, mongodb, self.state['positive'], self.state['negative'])
        precision, recall = accuracy_of_functions(g, librarynames)
        return -F(1, precision, recall)


def BP_with_strings(queryPath_1gram, queryPath_2gram, string_dict, db, positiveinfluence, negativeinfluence):
    global LBP_MAX_ITERS
    global THRES
    threshold=THRES
    g = fg.Graph()
    global_node_dict = dict()
    global_edge_dict = dict()
    global_sim_dict = defaultdict(float)
    rvlabels = defaultdict(list)
    if type(queryPath_2gram) == type(''):
        query_2gram = p.load(open(queryPath_2gram, 'rb'))
        query_1gram = p.load(open(queryPath_1gram, 'rb'))
    else:
        query_2gram = queryPath_2gram
        query_1gram = queryPath_1gram
    hasDistance = False
    if type(query_2gram[0][0][0]) == type((1,2)):
        hasDistance = True

    global_node_dict, global_prior_dict = get_node_prior_belief(query_1gram)
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
#             # initialize the dictionary
            if not global_node_dict.has_key(test_src):
                global_node_dict[test_src] = []
            if not global_node_dict.has_key(test_des):
                global_node_dict[test_des] = []
            if not global_edge_dict.has_key((test_src, test_des)):
                global_edge_dict[(test_src, test_des)] = []

            funcList = i[1]
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
                if (src_funcname, libraryname) not in global_node_dict[test_src]:
                    continue
                # do the same thing to des function
                if (des_funcname, libraryname) not in global_node_dict[test_des]:
                    continue

                # add this edge to the edge dictionary
                if (src_funcname, des_funcname, libraryname) not in global_edge_dict[(test_src, test_des)]:
                    global_edge_dict[(test_src, test_des)].append((src_funcname, des_funcname, libraryname))
                    if global_sim_dict[(src_funcname, des_funcname, libraryname)] < i[2]:
                        global_sim_dict[(src_funcname, des_funcname, libraryname)] = i[2]

    string_count = defaultdict(int)
    # add functions as random variables and add factors between function and string nodes
    for key in global_node_dict:
        rvlabels[test_src].append('None')
        length = len(global_node_dict[key])
        if length > 0:
            #rv_func = g.rv(key, length + 1, prior = global_prior_dict[key])
            rv_func = g.rv(key, length + 1)
        else:
            rv_func = g.rv(key, length + 1) # None label is the last item
        for idx, (prediction, library) in enumerate(global_node_dict[key]):
            query = db.load({"name":prediction})
            if not query:
                string_list = []
            else:
                string_list = query['strings']
            string_shared = set(string_dict[key]) & set(string_list)
            string_differ = set(string_dict[key]) | set(string_list) - string_shared
#             string_shared = set()
#             string_differ = set()
            #string_differ = set(string_list) - set(string_dict[key])
            for string in string_shared:
                count = string_count[string]
                string_count[string] += 1
                rv_string = g.rv('str_' + string + str(count), 1)
                fillvalue = 1.0
                pot=np.full((length + 1, 1), fillvalue)
                pot[idx, 0] = positiveinfluence * fillvalue
                factor = g.factor([rv_func, rv_string], potential=pot, name = '')

            for string in string_differ:
                count = string_count[string]
                string_count[string] += 1
                rv_string = g.rv('str_' + string + str(count), 1)
                fillvalue = 1.0
                pot=np.full((length + 1, 1), positiveinfluence * fillvalue)
                pot[idx, 0] = fillvalue
                factor = g.factor([rv_func, rv_string], potential=pot, name = '')
        global_node_dict[key].append(('None', 'None'))
    p.dump(global_node_dict, open('global_node_dict','w'))


    # add factors
    for (test_src, test_des) in global_edge_dict.keys():
        src_rv = g.get_rvs()[test_src]
        des_rv = g.get_rvs()[test_des]
        src_list = global_node_dict[test_src]
        des_list = global_node_dict[test_des]
        # set the probability for src_rv and des_rv
        # first find if there is an existing probability, if so, add to it
        # otherwise, initialize a new probability matrix
        factor1 = None
        probability1 = None

        if test_src != test_des:
            pot=np.full((len(src_list), len(des_list)), 0.1)
            #pot[-1, -1] = 0.1
            factor1 = g.factor([src_rv, des_rv], potential=pot, name = '')
            probability1 = factor1.get_potential()

        # if there is an matching edge
        if len(global_edge_dict[(test_src, test_des)]) > 0:
            for (src_funcname, des_funcname, libraryname) in global_edge_dict[(test_src, test_des)]:
                sim = global_sim_dict[(src_funcname, des_funcname, libraryname)]
                src_ind = src_list.index((src_funcname, libraryname))
                des_ind = des_list.index((des_funcname, libraryname))
                if src_rv == factor1.get_rvs()[0]:
                    #probability1[src_ind, des_ind] = 2**(sim + 2) - 7
                    probability1[src_ind, des_ind] = positiveinfluence * 0.1
                else:
                    #probability1[des_ind, src_ind] = 2**(sim + 2) - 7
                    probability1[des_ind, src_ind] = positiveinfluence * 0.1
        else:
            if len(src_list) > 0 and len(des_list) > 0:
                probability1[:, -1] = negativeinfluence * 0.1
                probability1[-1, :] = negativeinfluence * 0.1

        factor1.set_potential(probability1)
        #factor1.name += src_funcname + '->' + des_funcname + '\n'

    iters, converged = g.lbp(global_node_dict, normalize=True, max_iters=LBP_MAX_ITERS, progress=False)
    g.global_dict = global_node_dict
    print 'LBP ran for %d iterations. Converged = %r' % (iters, converged)
#     results = g.get_func_prediction(normalize=True)
#     p.dump(results, open('prediction','w'))
#     results = g.accuracy_of_functions(normalize=True)
    return g

def getRank(prediction):
    predict = sorted(prediction, key=lambda x: x[1])
    rank = 1
    lastsim = -1
    predicts = []
    ranks = []
    for (func, sim) in predict:
        if np.isclose(sim, lastsim):
            predicts.append(func)
            ranks.append(rank)
        else:
            predicts.append(func)
            ranks.append(rank)
            rank += 1
        lastsim = sim
    sm = softmax(range(ranks[-1]+1))
    for i in range(len(ranks)):
        ranks[i] = sm[ranks[i]]
    ranks.append(sm[0])
    return predicts, ranks

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def get_node_prior_belief(knn_all):
    prediction = defaultdict(set)
    sim_dict = defaultdict(float)
    for [(name, distance), predict, sim] in knn_all:
        if sim > 0.95:
            for [lib, version, func] in predict:
                prediction[name].add((func, lib))
                if sim_dict[(name, func, lib)] < sim:
                    sim_dict[(name, func, lib)] = sim

    prediction_with_sim = defaultdict(list)

    for name in prediction.keys():
        for (func, lib) in prediction[name]:
            prediction_with_sim[name].append(((func, lib), sim_dict[(name, func, lib)]))

    global_dict = defaultdict(list)
    global_beliefs = defaultdict(list)
    for func in prediction_with_sim.keys():
        predicts, beliefs = getRank(prediction_with_sim[func])
        global_dict[func] = predicts
        global_beliefs[func] = beliefs
        ""
    return global_dict, global_beliefs

def accuracy_of_functions(g, librarynames, rvs=None, normalize=True):
        # Extract
        global_dict = g.global_dict
        tuples = g.rv_marginals(rvs, normalize)
        prediction = dict()
        count = 0
        tn = 0
        fn = 0
        wp = 0
        fp = 0
        prior = defaultdict(list)
        # Display
        for rv, marg in tuples:
            rv_label = str(rv).split(' ')[0]
            if rv_label[:4] == 'str_':
                continue
            largest = -1
            for label in marg:
                if label > largest:
                    largest = label
            names = global_dict[rv_label]

            # if np.isclose(largest, 1.0/len(names)):
            #     continue
            for i, label in enumerate(marg):
                prior[rv_label].append((names[i], label))
                if np.isclose(label, largest):
                    #pdb.set_trace()
                    if rv_label in prediction:
                        prediction[rv_label].append(names[i])
                    else:
                        prediction[rv_label]=[names[i]]

        keylist = prior.keys()
        # for key in keylist:
        #     print key
        #     print prior[key]
        #     print
        keylist = prediction.keys()
        correct_set = set()
        for key in keylist:
            funclist = prediction[key]
            label=set()
            #print key
            for (func, lib) in funclist:
                if len(func.split('{')) > 1:
                    predicted_label = func.split('{')[1].split('}')[0]
                else:
                    predicted_label = func
                label.add(predicted_label)
            if 'None' in label and len(label)==1:
                if key in librarynames:
                    fn += 1
                else:
                    tn += 1
            elif key in label:
                count += 1
                correct_set.add(key)
                wp += len(label)-1
            else:
                if key in librarynames:
                    wp += len(label)
                else:
                    fp += len(label)

        print "correct prediction, true negative, false negative, wrong prediction, false positives, total prediction, precision:"
        print count, tn, fn, wp, fp, len(keylist), count*1.0/(len(keylist)-tn-fn), count*1.0/2982
        return count*1.0/(len(keylist)-tn-fn), count*1.0/2982

def build_string_table(bv, db = None):
    addr = []
    string_dict = defaultdict(list)
    for key in bv.sections.keys():
        if 'rodata' in key:
            start = bv.sections[key].start
            end = bv.sections[key].end
            for string in bv.get_strings(start, end):
                addr.append(string.start)
    if addr == []:
        for string in bv.get_strings():
            addr.append(string.start)

    for func in bv.functions:

        string_list = []
        for block in func.low_level_il.basic_blocks:
            for instr in block:
                for oper in instr.operands:
                    if isinstance(oper, binja.LowLevelILInstruction):
                        if oper.operation == binja.LowLevelILOperation.LLIL_CONST_PTR or oper.operation == binja.LowLevelILOperation.LLIL_CONST:
                            if int(str(oper), 16) in addr:
                                string_list.append(bv.get_string_at(int(str(oper), 16)).value)

        if len(string_list) > 0:
            string_list_strip = list(set(string_list))
            my_dict = {"name": func.name, "strings": string_list_strip}
            string_dict[func.name] = string_list_strip
            if db is not None:
                db.save(my_dict)
    return string_dict

filted = 1
def loadOneBinary(funcnamepath, embFile, filted_size=0):
    names = p.load(open(funcnamepath, 'r'))
    data = p.load(open(embFile, 'r'))
    funcName2emb=dict()
    for i in range(len(data)):
        (name, size) = names[i]
        if size > filted_size:
            funcName2emb[name]=data[i]
    return funcName2emb
#dir_openssl = '/home/yijiufly/Downloads/codesearch/Gemini/testingdataset/database/'
dir_openssl = '/home/yijiufly/Downloads/codesearch/data/openssl/'
dir_zlib = '/home/yijiufly/Downloads/codesearch/data/zlib/zlib-O2/'
#folder = '101d/'
folder = 'openssl-1.0.1d/'
embfile = 'libcrypto.so_newmodel.emb'
nam = 'libcrypto.so_newmodel_withsize.nam'
sslembfile = 'libssl.so_newmodel.emb'
sslnam = 'libssl.so_newmodel_withsize.nam'
func2emb_ssl = loadOneBinary(dir_openssl+folder+sslnam, dir_openssl+folder+sslembfile, filted)
#name_filted = p.load(open(dir_openssl+'libcrypto.so.ida_filted1.nam','r'))
func2emb_openssl = loadOneBinary(dir_openssl+folder+nam, dir_openssl+folder+embfile, filted)
func2emb_zlib = loadOneBinary(dir_zlib+'zlib-1.2.11/libz.so_newmodel_withsize.nam', dir_zlib+'zlib-1.2.11/libz.so_newmodel.emb', filted)
librarynames = func2emb_openssl.keys()
librarynames.extend(func2emb_ssl.keys())
librarynames.extend(func2emb_zlib.keys())
print len(librarynames)
libraryName = 'library'
mongodb = mdb('oss', libraryName + '_stringtable')
global LBP_MAX_ITERS
global THRES
LBP_MAX_ITERS =40
THRES = 0.95
knn = p.load(open('/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx/nginx-{openssl-1.0.1d}{zlib-1.2.11}/test_kNN_1021_2gram.p', 'r'))
#binaryview = binja.BinaryViewType.get_view_of_file('/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx/nginx-{openssl-1.0.1d}{zlib-1.2.11}/nginx-{openssl-1.0.1d}{zlib-1.2.11}')
#string_table = build_string_table(binaryview)
string_table = p.load(open('/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx/nginx-{openssl-1.0.1d}{zlib-1.2.11}/strings.str', 'r'))
knn_all = p.load(open('/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx/nginx-{openssl-1.0.1d}{zlib-1.2.11}/test_kNN_1030_1gram.p'))

init_state = {'positive':5.0, 'negative':5.0}
#pdb.set_trace()
tsp = FindBestParameters(init_state)
schedule = {'tmax':25000.0, 'tmin':2.5, 'steps':100, 'updates':10}
tsp.set_schedule(schedule)
state, e = tsp.anneal()
print state, e
