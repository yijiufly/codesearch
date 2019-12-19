import sys
import numpy as np
import pickle as p
sys.path.insert(0, "faiss/python")
import faiss
from testbinary import TestBinary
import os
def loadFile(PATH):
    if os.path.exists(PATH):
        return PATH
    else:
        raise Exception
folder = 'mupdf-1.15.0'
binName = 'mutool'
dir = '/home/yijiufly/Downloads/codesearch/data/mupdf/MUPDF-all/'
try:
    dotfile = loadFile(os.path.join(dir, folder, binName + '_bn.dot'))
    namfile = loadFile(os.path.join(dir, folder, binName + '.so_newmodel_withsize.nam'))
    embfile = loadFile(os.path.join(dir, folder, binName + '.so_newmodel.emb'))
    strfile = loadFile(os.path.join(dir, folder, binName + '.str'))
except Exception:
    print 'Missing generate embeddings, callgraphs, string files'

binFolder = os.path.join(dir, folder)

testbin = TestBinary(binName, binFolder, dotfile, embfile, namfile, strfile, 1)
index = faiss.read_index("funcemb_aftergrouping_cosine2.index")
meta = p.load(open('funcmetadata_aftergrouping_cosine2.p', 'r'))
testbin.loadBinary()

query = testbin.funcName2emb.keys()
query_emb = testbin.funcName2emb.values()
query_norm = query_emb / np.linalg.norm(query_emb, axis=1)[:,None]
D, I = index.search(np.array(query_norm), 1)
correct = 0
positive = 1061.0
threshold = 0.95
predictednodes = 0.0
for ind, i in enumerate(I):
    if D[ind][0] < threshold:
        continue
    predictednodes += 1
    result = set()
    for [libname, version, funcname] in meta[i[0]]:
        result.add(funcname)
    if query[ind] in result:
        correct += 1
print correct/predictednodes, correct/positive