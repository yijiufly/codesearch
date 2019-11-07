from sklearn.preprocessing import normalize
from library import Library
import traceback
import time
import json
import os
import numpy as np
import sys
sys.path.append('./SPTAG/Release')
import SPTAG
import pickle as p
def loadFiles(PATH, ext=None):  # use .ida or .emb for ida file and embedding file
    filenames = []
    filenames = [f for f in os.listdir(PATH) if f.endswith(ext)]
    return filenames


def build3gram(libdir):
    for folder in os.listdir(libdir):
        print(folder)
        start_time = time.time()
        try:
            dotfiles = loadFiles(os.path.join(libdir, folder), ext='.dot')
            dotfiles.sort()
            namfiles = loadFiles(os.path.join(libdir, folder), ext='.nam')
            namfiles.sort()
            embfiles = loadFiles(os.path.join(libdir, folder), ext='.emb')
            embfiles.sort()
        except Exception:
            print traceback.format_exc()
            continue
        for i in range(len(dotfiles)):
            dotfile = dotfiles[i]
            if dotfile == 'libcrypto.so.dot':
                continue
            namfile = namfiles[i]
            embfile = embfiles[i]
            print dotfile, namfile, embfile
            libraryName = folder + '_' + dotfile.split('.')[0]
            dotPath = os.path.join(libdir, folder, dotfile)
            namPath = os.path.join(libdir, folder, namfile)
            embPath = os.path.join(libdir, folder, embfile)
            library = Library(libraryName, dotPath, embPath)
            library.buildNGram(namPath)

            metadata = ''
            x = []
            for item in library.threeGramList:
                x.append(item[0])
                m = {'libraryname': libraryName,
                    'version': folder, 'funcs': item[1]}
                metadata += json.dumps(m) + '\n'
            testAddWithMetaData('3gramdb2', normalize(
                np.array(x)), metadata, '3gramdb2')
            print('load ' + libraryName)
            print("--- %s seconds ---\n" % (time.time() - start_time))


def queryForOneBinary3Gram(threeGramList, outpath):
    start_time = time.time()
    x = []
    for item in threeGramList:
        x.append(item[0])
    print len(x)
    results = testSearchWithMetaData('3gramdb', normalize(np.array(x[:100000])), 200)
    print("--- %s seconds ---\n" % (time.time() - start_time))
    p.dump(results, open(outpath, "w"))


def testBuildWithMetaData(algo, distmethod, x, s, out):
   i = SPTAG.AnnIndex(algo, 'Float', x.shape[1])
   i.SetBuildParam("NumberOfThreads", '8')
   i.SetBuildParam("DistCalcMethod", distmethod)
   if i.BuildWithMetaData(x.tobytes(), s, x.shape[0]):
       i.Save(out)


def testSearchWithMetaData(index, q, k):
    j = SPTAG.AnnIndex.Load(index)
    j.SetSearchParam("MaxCheck", '1024')
    results = []
    for t in range(q.shape[0]):
        result = j.SearchWithMetaData(q[t].tobytes(), k)
        results.append(result)
        # print (result[0]) # ids
        # print (result[1]) # distances
        # print (result[2]) # metadata
    return results

def testAddWithMetaData(index, x, s, out):
    if index != None:
        i = SPTAG.AnnIndex.Load(index)
    # else:
    #     i = SPTAG.AnnIndex(algo, 'Float', x.shape[1])
    #     i.SetBuildParam("NumberOfThreads", '4')
    #     i.SetBuildParam("DistCalcMethod", distmethod)
    if i.AddWithMetaData(x.tobytes(), s, x.shape[0]):
        i.Save(out)

def testDelete(index, x, out):
   i = SPTAG.AnnIndex.Load(index)
   ret = i.Delete(x.tobytes(), x.shape[0])
   print (ret)
   i.Save(out)

def Test(algo, distmethod):
   x = np.ones((n, 10), dtype=np.float32) * np.reshape(np.arange(n, dtype=np.float32), (n, 1))
   q = np.ones((r, 10), dtype=np.float32) * np.reshape(np.arange(r, dtype=np.float32), (r, 1)) * 2
   print x
   print q
   m = ''
   for i in range(n):
       m += str(i)+'\t'+str(i)+'\t'+str(i)+'\n'

   print ("Build.............................")
   # testBuild(algo, distmethod, x, 'testindices')
   # testSearch('testindices', q, k)
   # print ("Add.............................")
   # testAdd('testindices', x, 'testindices', algo, distmethod)
   # testSearch('testindices', q, k)
   # print ("Delete.............................")
   # testDelete('testindices', q, 'testindices')
   # testSearch('testindices', q, k)
   testBuildWithMetaData(algo, distmethod, x, m,'testindices')
   # print ("AddWithMetaData.............................")
   # testAddWithMetaData(None, x, m, 'testindices', algo, distmethod)
   print ("Search.............................")
   testSearchWithMetaData('testindices', q, k)
   print ("Delete.............................")
   testDelete('testindices', q, 'testindices')
   # testSearchWithMetaData('testindices', q, k)

if __name__ == '__main__':
    # x = np.ones((1, 192), dtype=np.float32) * np.reshape(np.arange(1, dtype=np.float32), (1, 1))
    # s='testbuild\n'
    # testBuildWithMetaData('BKT', 'cosine', normalize(x), s, '3gramdb2')
    build3gram('/home/yijiufly/Downloads/codesearch/data/openssl')
