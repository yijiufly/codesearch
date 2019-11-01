import os
import numpy as np
import pdb
from collections import defaultdict
#draw ROC for functions
import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import sys
import pdb
sys.path.insert(0, './embedding_w2v')
from embedding_w2v.embedding import Embedding
emb = Embedding()
def drawROC(resultMat, name_test, length, name_nginx):
    rank_index_list = []
    for s in resultMat:
        a = sorted(range(len(s)), key=lambda k: s[k][0], reverse=True)
        rank_index_list.append(a)

    for threshold in np.linspace(0.9, 1, 10):
        func_counts = len(rank_index_list)
        total_tn = []
        total_fp = []
        total_cp = []
        total_wp = []
        total_fn = []
        for func in range(func_counts):
            real_name = name_test[func]
            if resultMat[func][rank_index_list[func][0]][0] >= threshold:
                # correct prediction
                if resultMat[func][rank_index_list[func][0]][1] == real_name:
                    total_cp.append(real_name)
                elif name_test[func] in name_nginx:
                    total_fp.append(real_name)
                else:
                    total_wp.append(real_name)
            else:
                if name_test[func] in name_nginx:
                    total_tn.append(real_name)
                else:
                    total_fn.append(real_name)

        print threshold
        print 'total_tn', len(total_tn)
        print 'total_fp', len(total_fp)
        print 'total_cp', len(total_cp)
        print 'total_wp', len(total_wp)
        print 'total_fn', len(total_fn)
        print
    return rank_index_list

name = 'nginx-{openssl-1.0.1d}{zlib-1.2.11}'
#dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx/nginx-{openssl-1.0.1d}{zlib-1.2.11}/w2v/'
dir = '/home/yijiufly/Downloads/codesearch/Gemini/testingdataset/test/'
embfiles = name + '.ida_newmodel.emb'
namfiles = name + '.ida_newmodel_withsize.nam'
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
func2emb_test = loadOneBinary(dir+namfiles, dir+embfiles, filted)
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
name_test=[]
emb_test=[]
name_nginx=[]
count1 = count2 = count3 = 0
for key in func2emb_test.keys():
    name_test.append(key)
    emb_test.append(func2emb_test[key])
    if key in func2emb_openssl.keys():
        count1 += 1
    elif key in func2emb_ssl.keys():
        count2 += 1
    elif key in func2emb_zlib.keys():
        count3 += 1
    else:
        name_nginx.append(key)
print count1, count2, count3
print len(name_test)
print len(name_nginx)
name_openssl_all = ['0']
resultMat_all = [[(0, '0')] for i in range(len(name_test))]
acfgs_all = []
#for folder in os.listdir(dir_openssl)[0], os.listdir(dir_openssl)[1], os.listdir(dir_openssl)[2], os.listdir(dir_openssl)[4], os.listdir(dir_openssl)[55]:
for folder in os.listdir(dir_openssl):
    #print folder
    embfile = '/libcrypto.so_newmodel.emb'
    nam = '/libcrypto.so_newmodel_withsize.nam'
    sslembfile = '/libssl.so_newmodel.emb'
    sslnam = '/libssl.so_newmodel_withsize.nam'
    idafile = '/libcrypto.so.ida'
    sslidafile = '/libssl.so.ida'
    func2emb_ssl = loadOneBinary(dir_openssl+folder+sslnam, dir_openssl+folder+sslembfile, filted)
    name_openssl = func2emb_ssl.keys()
    emb_openssl = func2emb_ssl.values()
    #name_filted = p.load(open(dir_openssl+'libcrypto.so.ida_filted1.nam','r'))
    func2emb_openssl = loadOneBinary(dir_openssl+folder+nam, dir_openssl+folder+embfile, filted)

    name_openssl.extend(func2emb_openssl.keys())
    emb_openssl.extend(func2emb_openssl.values())

    name_openssl_all.extend(name_openssl)
    resultMat = emb.test_similarity(emb_test, emb_openssl)
    for i, s in enumerate(resultMat):
        a = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
        for j in range(10):
            resultMat_all[i].append((s[a[j]], name_openssl[a[j]]))
for folder in os.listdir(dir_zlib):
    #print folder
    embfile = '/libz.so_newmodel.emb'
    nam = '/libz.so_newmodel_withsize.nam'
    idafile = '/libz.so.ida'
    func2emb_zlib = loadOneBinary(dir_zlib+folder+nam, dir_zlib+folder+embfile, filted)
    name_zlib = func2emb_zlib.keys()
    emb_zlib = func2emb_zlib.values()
    #name_filted = p.load(open(dir_openssl+'libcrypto.so.ida_filted1.nam','r'))
    name_openssl_all.extend(name_zlib)
    resultMat = emb.test_similarity(emb_test, emb_zlib)
    for i, s in enumerate(resultMat):
        a = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
        for j in range(10):
            resultMat_all[i].append((s[a[j]], name_zlib[a[j]]))
    #resultMat_all = np.concatenate((resultMat_all, resultMat), axis = 1)
rank = drawROC(resultMat_all, name_test, len(name_openssl_all), name_nginx)
