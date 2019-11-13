from binary import Binary
import os
from redis import Redis
from db import db
import numpy as np
import pickle as p
import os
import ConfigParser
import ast
import sys
sys.path.insert(0, './Gemini/embedding')
from embedding import Embedding
class Library(Binary):
    def __init__(self, libraryName, dotPath, embFile, namFile, filterSize=0):
        self.libraryName = libraryName
        self.loadCallGraph(dotPath, namFile)
        self.generatefuncNameFilted(namFile, filterSize)
        self.getGraphFromPathfilted()
        self.embFile = embFile


def build2gram(pathLib, folder, lib):
    # get configuration
    config = ConfigParser.RawConfigParser()
    config.read('config')

    redisObject = Redis(host='localhost', port=6379, db=int(config.get("TwoGramRedis", "DATABASE")))
    hashMap = db()
    hashMap.loadHashMap(ast.literal_eval(config.get("TwoGramRedis", "CONFIG")), redisObject, int(config.get("TwoGramRedis", "DIM")), int(config.get("TwoGramRedis", "PROJECTIONS")))
    filterSize = 1
    dotfile = lib+'.so_bn.dot'
    dotPath = os.path.join(pathLib, folder, dotfile)
    #libraryName = folder.split('-')[1] + '_' + dotfile.rsplit('.',1)[0]
    libraryName = folder + '_' + dotfile.rsplit('.',1)[0]
    embPath = os.path.join(pathLib, folder, lib+'.so_newmodel.emb')
    namPath = os.path.join(pathLib, folder, lib+'.so_newmodel_withsize.nam')
    libitem = Library(libraryName, dotPath, embPath, namPath, filterSize)
    libitem.buildNGram(namPath, embPath)
    print(len(libitem.twoGramList))
    for [twogram, name, distance] in libitem.twoGramList:
        hashMap.indexing([twogram], [[lib, folder, name, distance]])

    print('load ' + libitem.libraryName)

def build1gram(pathLib, folder, lib):
    # get configuration
    config = ConfigParser.RawConfigParser()
    config.read('config')
    redisObject = Redis(host='localhost', port=6379, db=config.get("FuncRedis", "DATABASE"))
    hashMap = db()
    hashMap.loadHashMap(ast.literal_eval(config.get("FuncRedis", "CONFIG")), redisObject, int(config.get("FuncRedis", "DIM")), int(config.get("FuncRedis", "PROJECTIONS")))
    filterSize = 1
    dotfile = lib+'.so_bn.dot'
    dotPath = os.path.join(pathLib, folder, dotfile)
    #libraryName = folder.split('-')[1] + '_' + dotfile.rsplit('.',1)[0]
    libraryName = folder + '_' + dotfile.rsplit('.',1)[0]
    embPath = os.path.join(pathLib, folder, lib+'.so_newmodel.emb')
    namPath = os.path.join(pathLib, folder, lib+'.so_newmodel_withsize.nam')
    libitem = Library(libraryName, dotPath, embPath, namPath, filterSize)
    libitem.loadOneBinary(namPath, libitem.embFile)
    for func in libitem.funcNameFilted.keys():
        if libitem.funcNameFilted[func] != -1:
            hashMap.indexing([libitem.funcName2emb[func]], [[lib, folder, func]])

    print('load ' + libitem.libraryName)


def grouping():
    emb = Embedding()
    # get configuration
    config = ConfigParser.RawConfigParser()
    config.read('config')

    redisObject = Redis(host='localhost', port=6379, db=int(config.get("TwoGramRedis", "DATABASE")))
    hashMap = db()
    hashMap.loadHashMap(ast.literal_eval(config.get("TwoGramRedis", "CONFIG")), redisObject, int(config.get("TwoGramRedis", "DIM")), int(config.get("TwoGramRedis", "PROJECTIONS")))
    hashMap.grouping(emb)

    redisObject = Redis(host='localhost', port=6379, db=config.get("FuncRedis", "DATABASE"))
    hashMap2 = db()
    hashMap2.loadHashMap(ast.literal_eval(config.get("FuncRedis", "CONFIG")), redisObject, int(config.get("FuncRedis", "DIM")), int(config.get("FuncRedis", "PROJECTIONS")))
    hashMap2.grouping(emb)