import sys
sys.path.insert(0, "faiss/python")
import faiss                   # make faiss available
import pickle as p
import os
import numpy as np
import random
import pdb
import time
from Quick_Find import *
from Quick_Union import *
from library import Library
def indexing(index, isFunc = True):
    #path_lib = '/home/yijiufly/Downloads/codesearch/data/zlib/zlib-O2'
    path_lib = 'example_lib/zlib'
    vectors = []
    metadatas = []
    for folder in os.listdir(path_lib):
        if isFunc:
            vector, metadata = build1gram(path_lib, folder, 'libz', index)
        else:
            vector, metadata = build2gram(path_lib, folder, 'libz', index)
        vectors.extend(vector)
        metadatas.extend(metadata)

    #path_lib = '/home/yijiufly/Downloads/codesearch/data/openssl'
    path_lib = 'example_lib/openssl'
    for folder in os.listdir(path_lib):
        if isFunc:
            vector, metadata = build1gram(path_lib, folder,'libcrypto', index)
        else:
            vector, metadata = build2gram(path_lib, folder,'libcrypto', index)
        vectors.extend(vector)
        metadatas.extend(metadata)
        if isFunc:
            vector, metadata = build1gram(path_lib, folder,'libssl', index)
        else:
            vector, metadata = build2gram(path_lib, folder,'libssl', index)
        vectors.extend(vector)
        metadatas.extend(metadata)
    return vectors, metadatas

def loadOneBinary(funcnamepath, embFile):
    names = p.load(open(funcnamepath, 'r'))
    data = p.load(open(embFile, 'r'))
    
    funcName2emb=dict()
    for i in range(len(names)):
        funcName2emb[names[i][0]]=data[i]

    return funcName2emb

def build1gram(pathLib, folder, lib, index):
    filterSize = 1
    dotfile = lib+'.so_bn.dot'
    dotPath = os.path.join(pathLib, folder, dotfile)

    libraryName = folder + '_' + dotfile.rsplit('.',1)[0]
    embPath = os.path.join(pathLib, folder, lib+'.so_newmodel.emb')
    namPath = os.path.join(pathLib, folder, lib+'.so_newmodel_withsize.nam')
    funcName2emb = loadOneBinary(namPath, embPath)
    metadata = []
    vectors = []
    
    for func in funcName2emb.keys():
        vectors.append(funcName2emb[func])
        metadata.append([lib, folder, func])
    # normalize the vectors
    norm_vectors = vectors / np.linalg.norm(vectors, axis=1)[:,None]
    index.add(np.array(norm_vectors))                  # add vectors to the index
    print index.ntotal
    print('load ' + libraryName)
    return vectors, metadata

def build2gram(pathLib, folder, lib, index):
    filterSize = 1
    dotfile = lib+'.so_bn.dot'
    dotPath = os.path.join(pathLib, folder, dotfile)
    libraryName = folder + '_' + dotfile.rsplit('.',1)[0]
    embPath = os.path.join(pathLib, folder, lib+'.so_newmodel.emb')
    namPath = os.path.join(pathLib, folder, lib+'.so_newmodel_withsize.nam')
    libitem = Library(libraryName, dotPath, embPath, namPath, filterSize)
    libitem.buildNGram(namPath, embPath)
    print(len(libitem.twoGramList))
    metadata = []
    vectors = []
    for [twogram, name, distance] in libitem.twoGramList:
        vectors.append(twogram)
        metadata.append([lib, folder, name, distance])
    norm_vectors = vectors / np.linalg.norm(vectors, axis=1)[:,None]
    index.add(np.array(norm_vectors))                  # add vectors to the index
    print index.ntotal
    print('load ' + libitem.libraryName)
    return norm_vectors, metadata

def grouping(distance, index, lower, upper, eleNodeMap):
    for i in range(len(index)):
        for j, dis in enumerate(distance[i]):
            if dis > 0.9999999:
                quickUnion((lower + i, index[i][j]), eleNodeMap)
            else:
                break

def build2gramDB():
    start_time = time.time()
    index = faiss.IndexFlatIP(128)   # build the index, d=size of vectors 
    vectors, metadatas = indexing(index, isFunc = False)

    print("--- %s seconds ---" % (time.time() - start_time))
    faiss.write_index(index,"2gram_cosine.index")
    
    eleList = [i for i in range(len(vectors))]
    eleNodeMap = genNodeList(eleList)
    for i in range(len(vectors)/5000):
        start_time = time.time()
        lower = i * 5000
        upper = min((i + 1) * 5000, len(vectors))
        D, I = index.search(np.array(vectors[lower:upper]), 120)
        print("--- %s seconds ---" % (time.time() - start_time))
        
        grouping(D, I, lower, upper, eleNodeMap)
    #pdb.set_trace()

    groups = dict()
    for i in eleNodeMap.keys():
        if eleNodeMap[i].parent.num not in groups:
            groups[eleNodeMap[i].parent.num] = set()

        groups[eleNodeMap[i].parent.num].add(i)

    new_vector = []
    new_metadata = []
    for key in groups.keys():
        new_vector.append(vectors[key])
        group_metadata = []
        for idx in groups[key]:
            group_metadata.append(metadatas[idx])
        new_metadata.append(group_metadata)

    index2 = faiss.IndexFlatIP(128)
    index2.add(np.array(new_vector))
    faiss.write_index(index2,"2gram_aftergrouping_cosine.index")
    p.dump(new_metadata, open('2grammetadata_aftergrouping_cosine.p', 'w'))

def build1gramDB():
    start_time = time.time()
    index = faiss.IndexFlatIP(64)   # build the index, d=size of vectors 
    vectors, metadatas = indexing(index, isFunc = True)

    print("--- %s seconds ---" % (time.time() - start_time))
    faiss.write_index(index,"funcemb.index")
    
    eleList = [i for i in range(len(vectors))]
    eleNodeMap = genNodeList(eleList)
    for i in range(len(vectors)/5000):
        start_time = time.time()
        lower = i * 5000
        upper = min((i + 1) * 5000, len(vectors))
        D, I = index.search(np.array(vectors[lower:upper]), 120)
        print("--- %s seconds ---" % (time.time() - start_time))
        grouping(D, I, lower, upper, eleNodeMap)


    groups = dict()
    for i in eleNodeMap.keys():
        if eleNodeMap[i].parent.num not in groups:
            groups[eleNodeMap[i].parent.num] = set()

        groups[eleNodeMap[i].parent.num].add(i)

    new_vector = []
    new_metadata = []
    for key in groups.keys():
        new_vector.append(vectors[key])
        group_metadata = []
        for idx in groups[key]:
            group_metadata.append(metadatas[idx])
        new_metadata.append(group_metadata)

    index2 = faiss.IndexFlatIP(64)
    index2.add(np.array(new_vector))
    faiss.write_index(index2, "funcemb_aftergrouping_cosine.index")
    p.dump(new_metadata, open('funcmetadata_aftergrouping_cosine.p', 'w'))


if __name__ == '__main__':
    index = faiss.read_index("funcemb_aftergrouping.index")
    vectors = index.reconstruct_n(0, index.ntotal)
    metadatas = p.load(open('funcmetadata_aftergrouping.p', 'r'))
    assert len(vectors) == len(metadatas)
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1)[:,None]
    index_cosine = faiss.IndexFlatIP(64)
    index_cosine.add(vectors_norm)
    
    eleList = [i for i in range(len(vectors_norm))]
    eleNodeMap = genNodeList(eleList)
    for i in range(len(vectors_norm)/5000):
        start_time = time.time()
        lower = i * 5000
        upper = min((i + 1) * 5000, len(vectors_norm))
        D, I = index_cosine.search(np.array(vectors_norm[lower:upper]), 120)
        print("--- %s seconds ---" % (time.time() - start_time))
        
        grouping(D, I, lower, upper, eleNodeMap)
    #pdb.set_trace()

    groups = dict()
    for i in eleNodeMap.keys():
        if eleNodeMap[i].parent.num not in groups:
            groups[eleNodeMap[i].parent.num] = set()

        groups[eleNodeMap[i].parent.num].add(i)

    new_vector = []
    new_metadata = []
    for key in groups.keys():
        new_vector.append(vectors_norm[key])
        group_metadata = []
        for idx in groups[key]:
            group_metadata.extend(metadatas[idx])
        new_metadata.append(group_metadata)

    index2 = faiss.IndexFlatIP(64)
    index2.add(np.array(new_vector))
    faiss.write_index(index2,"funcemb_aftergrouping_cosine.index")
    p.dump(new_metadata, open('funcmetadata_aftergrouping_cosine.p', 'w'))

