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

def loadOneBinary(funcnamepath, embFile, filted_size=0):
    names = p.load(open(funcnamepath, 'r'))
    data = p.load(open(embFile, 'r'))
    funcName2emb=dict()
    for i in range(len(data)):
        (name, size) = names[i]
        if size > filted_size:
            funcName2emb[name]=data[i]
    return funcName2emb

def testPR(func2emb_lib, func2emb_bin, threshold):
    nam_lib = func2emb_lib.keys()
    emb_test = func2emb_bin.values()
    nam_test = func2emb_bin.keys()
    resultMat = cal_similarity(func2emb_lib, func2emb_bin)
    rank_index_list = []
    length = len(resultMat[0])
    for s in resultMat:
        a = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
        rank_index_list.append(a)
    cp = []
    fp = []
    n = []
    for i in range(len(emb_test)):
        if resultMat[i][rank_index_list[i][0]] > threshold:
            if nam_lib[rank_index_list[i][0]] == nam_test[i]:
                cp.append((nam_test[i], resultMat[i][rank_index_list[i][0]]))
            else:
                fp.append((nam_test[i], resultMat[i][rank_index_list[i][0]]))
        else:
            n.append((nam_test[i], resultMat[i][rank_index_list[i][0]]))
    print(threshold, len(cp), len(fp), len(n))
    return len(cp), len(fp), len(n)

def cal_similarity(func2emb_lib, func2emb_bin):
    emb_lib = func2emb_lib.values()
    nam_lib = func2emb_lib.keys()
    emb_test = func2emb_bin.values()
    nam_test = func2emb_bin.keys()
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
    filted = 0
    func2emb_test = loadOneBinary(dir+namfiles, dir+embfiles, 0)
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
    filted = 0
    func2emb_test2 = loadOneBinary(dir+namfiles, dir+embfiles, filted)
    dir_openssl = '/home/yijiufly/Downloads/codesearch/data/openssl/openssl-1.0.1d/'
    #dir_freetype2 = '/home/yijiufly/Downloads/codesearch/data/mupdf/freetype2-test/VER-2-9-1/'
    emb2 = 'libcrypto.so_newmodel.emb'
    nam2 = 'libcrypto.so_newmodel_withsize.nam'
    #name_filted = p.load(open(dir_openssl+'libcrypto.so.ida_filted1.nam','r'))
    func2emb_openssl2 = loadOneBinary(dir_openssl+nam2, dir_openssl+emb2, filted)

    # graph embedding
    dir = '/home/yijiufly/Downloads/codesearch/test/'
    func2emb_test3 = p.load(open(dir + 'test_dic.pkl', 'rb'))
    func2emb_openssl3 = p.load(open(dir + 'train_dic.pkl', 'rb'))
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

    x = np.linspace(0.7, 1.0, 30)
    y = [[], [], []]
    openssl = [func2emb_openssl, func2emb_openssl2, func2emb_openssl3]
    test = [func2emb_test, func2emb_test2, func2emb_test3]
    for threshold in x:
        for i in range(len(openssl)):
            cp, fp, n = testPR(openssl[i], test[i], threshold)
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