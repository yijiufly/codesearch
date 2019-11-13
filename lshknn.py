from db import db
import numpy as np
import pickle as p
import os
from library import Library
import pymongo
from redis import Redis
import traceback
from multiprocessing import Pool
import time
from testbinary import TestBinary
import pdb
import multiprocessing


def loadFiles(PATH, ext=None):  # use .ida or .emb for ida file and embedding file
    filenames = []
    filenames = [f for f in os.listdir(PATH) if f.endswith(ext)]
    return filenames


def getHashMap(configname, storage_object, dim=64, projections=20):
    db_instance = db()
    db_instance.loadHashMap(configname, storage_object, dim, projections)

    return db_instance


def cleanAllBuckets(db_instance):
    db_instance.engine.clean_all_buckets()


def addToHashMap(db_instance, data, funcList):
    db_instance.indexing(data, funcList)

    return db_instance


def doSearch(db_instance, query):
    return db_instance.querying(query)


if __name__ == '__main__':
    pass