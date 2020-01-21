import sys
sys.path.insert(0, './Gemini/embedding')
from embedding import Embedding
emb = Embedding()
import numpy as np
import matplotlib.pyplot as plt
import pickle as p
import pdb

def cdf(sim, plot=True):
    for x in sim:
        num_bins = 100
        counts, bin_edges = np.histogram (x, bins=num_bins, normed=True)
        cdf = np.cumsum (counts)
        
        plt.plot (bin_edges[1:], cdf/cdf[-1], )
    plt.axis([-1, 1, 0, 1])
    plt.title('CDF: similarity distribution between a nginx function and functions in DB')
    plt.xlabel('similarity')
    plt.ylabel('cumulative distribution ')
    plt.legend(['with inline', 'w/o inline', 'gcn'])
    plt.show()

def loadOneBinary(funcnamepath, embFile, filted_size=1):
    names = p.load(open(funcnamepath, 'r'))
    data = p.load(open(embFile, 'r'))
    funcName2emb=dict()
    for i in range(len(data)):
        if type(names[i]) == type((1,2)):
            (name, size) = names[i]
            if size > filted_size:
                funcName2emb[name]=data[i]
        else:
            funcName2emb[names[i]]=data[i]
    return funcName2emb

def testPR(func2emb_lib, func2emb_bin, threshold):
    nam_lib = func2emb_lib.keys()
    emb_test = func2emb_bin.values()
    nam_test = func2emb_bin.keys()
    emb_lib = func2emb_lib.values()
    resultMat = cal_similarity(func2emb_lib, func2emb_bin)
    cp = []
    fp = []
    n = []
    if type(resultMat) == type((1, 2)):
        rank_index_list_origin = []
        rank_index_list_struc = []
        (mat_origin, mat_struc) = resultMat
        for s, t in zip(mat_origin, mat_struc):
            a = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
            b = sorted(range(len(t)), key=lambda k: t[k], reverse=True)
            rank_index_list_origin.append(a)
            rank_index_list_struc.append(b)
        for i in range(len(emb_test)):
            if mat_origin[i][rank_index_list_origin[i][0]] > threshold:
                import pdb; pdb.set_trace()
                if emb_test[i][1].all() == 0 and emb_lib[rank_index_list_origin[i][0]][1].all() == 0:
                    if nam_lib[rank_index_list_origin[i][0]] == nam_test[i]:
                        cp.append(nam_test[i])
                    else:
                        fp.append(nam_test[i])
                elif emb_test[i][1].all() == 0 or emb_lib[rank_index_list_origin[i][0]][1].all() == 0:
                    n.append(nam_test[i])
                elif mat_struc[i][rank_index_list_origin[i][0]] > 0.9:
                    if nam_lib[rank_index_list_origin[i][0]] == nam_test[i]:
                        cp.append(nam_test[i])
                    else:
                        fp.append(nam_test[i])
            else:
                n.append(nam_test[i])

    else:
        rank_index_list = []
        for s in resultMat:
            a = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
            rank_index_list.append(a)
        
        for i in range(len(emb_test)):
            if resultMat[i][rank_index_list[i][0]] > threshold:
                #import pdb; pdb.set_trace()
                if nam_lib[rank_index_list[i][0]] == nam_test[i]:
                    cp.append((nam_test[i], resultMat[i][rank_index_list[i][0]]))
                else:
                    fp.append((nam_test[i], resultMat[i][rank_index_list[i][0]]))
            else:
                n.append((nam_test[i], resultMat[i][rank_index_list[i][0]]))
    print(threshold, len(cp), len(fp), len(n))
    #return len(cp), len(fp), len(n)
    return cp, fp, n

def cal_similarity(func2emb_lib, func2emb_bin):
    emb_lib = func2emb_lib.values()
    nam_lib = func2emb_lib.keys()
    emb_test = func2emb_bin.values()
    nam_test = func2emb_bin.keys()
    if type(emb_lib[0]) == type((1,2)):
        emb_lib_origin = [np.hstack((i[0], i[1])) for i in emb_lib]
        emb_lib_struc = [i[1] for i in emb_lib]
        emb_test_origin = [np.hstack((i[0], i[1])) for i in emb_test]
        emb_test_struc = [i[1] for i in emb_test]
        resultMat1 = emb.test_similarity(emb_test_origin, emb_lib_origin)
        #resultMat2 = emb.test_similarity(emb_test_struc, emb_lib_struc)
        return resultMat1#, resultMat2
    else:
        resultMat = emb.test_similarity(emb_test, emb_lib)
        return resultMat

def F(beta, precision, recall):
    return (beta*beta + 1)*precision*recall / (beta*beta*precision + recall)

if __name__ == '__main__':
    # inline data
    name = 'inline'
    dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx/nginx-{openssl-1.0.1d}{zlib-1.2.11}/'
    embfiles = name + '.so_newmodel.emb'
    namfiles = name + '.so_newmodel_withsize.nam'
    filted = 1
    func2emb_test = loadOneBinary(dir+namfiles, dir+embfiles, filted)
    dir_openssl = '/home/yijiufly/Downloads/codesearch/data/openssl/openssl-1.0.1d/'
    emb1 = 'inline.so_newmodel.emb'
    nam1 = 'inline.so_newmodel_withsize.nam'
    func2emb_openssl = loadOneBinary(dir_openssl+nam1, dir_openssl+emb1, filted)


    # w/o inline data
    name = 'nginx-{openssl-1.0.1d}{zlib-1.2.11}'
    dir = '/home/yijiufly/Downloads/codesearch/data/versiondetect/test3/nginx/nginx-{openssl-1.0.1d}{zlib-1.2.11}/'
    #dir = '/home/yijiufly/Downloads/codesearch/data/mupdf/MUPDF-all/mupdf-1.15.0/'
    embfiles = name + '.ida_newmodel.emb'
    namfiles = name + '.ida_newmodel_withsize.nam'
    filted = 1
    func2emb_test2 = loadOneBinary(dir+namfiles, dir+embfiles, filted)
    dir_openssl = '/home/yijiufly/Downloads/codesearch/data/openssl/openssl-1.0.1d/'
    #dir_freetype2 = '/home/yijiufly/Downloads/codesearch/data/mupdf/freetype2-test/VER-2-9-1/'
    emb2 = 'libcrypto.so_newmodel.emb'
    nam2 = 'libcrypto.so_newmodel_withsize.nam'
    #name_filted = p.load(open(dir_openssl+'libcrypto.so.ida_filted1.nam','r'))
    func2emb_openssl2 = loadOneBinary(dir_openssl+nam2, dir_openssl+emb2, filted)

    # graph embedding
    dir_ge = '/home/yijiufly/Downloads/gae/gae/data/'
    func2emb_test3 = p.load(open(dir_ge + 'test_dic.pkl', 'rb'))
    func2emb_openssl3 = p.load(open(dir_ge + 'train_dic.pkl', 'rb'))
    [func2emb_test3.pop(key) for key in set(func2emb_test3.keys()) - set(func2emb_test.keys())]
    [func2emb_openssl3.pop(key) for key in set(func2emb_openssl3.keys()) - set(func2emb_openssl.keys())]

    # func2emb_test3 = loadOneBinary(dir+namfiles, dir_ge+'test_dic.pkl', filted)
    # func2emb_openssl3 = loadOneBinary(dir_ge+'openssl-1.0.1d/openssl-1.0.1d.nam', dir_ge+'train_dic.pkl', filted)
    # p.dump(func2emb_test3, open(dir + 'test_dic.pkl', 'wb'), protocol=2)
    # p.dump(func2emb_test3, open(dir + 'train_dic.pkl', 'wb'), protocol=2)

    # resultMat = cal_similarity(func2emb_openssl, func2emb_test)
    # resultMat2 = cal_similarity(func2emb_openssl2, func2emb_test2)
    # resultMat3 = cal_similarity(func2emb_openssl3, func2emb_test3)
    # sim = [[], [], []]
    # for item in resultMat:
    #     sim[0].extend(item)
    # for item in resultMat2:
    #     sim[1].extend(item)
    # for item in resultMat3:
    #     sim[2].extend(item)
    # cdf(np.array(sim))

    positive_total = [len(set(func2emb_openssl.keys()) & set(func2emb_test.keys()))]
    positive_total.append(len(set(func2emb_openssl2.keys()) & set(func2emb_test2.keys())))
    positive_total.append(len(set(func2emb_openssl3.keys()) & set(func2emb_test3.keys())))
    print(positive_total)
    print(positive_total[2])
    x = np.linspace(0.7, 1.0, 10)
    y = [[], [], []]
    openssl = [func2emb_openssl, func2emb_openssl2, func2emb_openssl3]
    test = [func2emb_test, func2emb_test2, func2emb_test3]
    for threshold in x:
        for i in range(len(openssl)):
            cp, fp, n = testPR(openssl[i], test[i], threshold)
            cp = len(cp)
            fp = len(fp)
            n = len(n)
            precision = cp * 1.0 / (cp + fp)
            recall = cp * 1.0 / positive_total[i]
            
            y[i].append(F(1, precision, recall))

    plt.plot(x, y[0])
    plt.plot(x, y[1])
    plt.plot(x, y[2])
    plt.legend(['with inline', 'w/o inline', 'gcn'])
    plt.xlabel('threshold')
    plt.ylabel('F1 score')
    plt.show()