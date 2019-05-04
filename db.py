from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections, RandomDiscretizedProjections, RandomBinaryProjectionTree
import numpy as np
from redis import Redis
import pymongo
from nearpy.storage import RedisStorage, MongoStorage
from nearpy.filters import NearestFilter, UniqueFilter, RankNFilter
import pickle as p
import time
import pdb
import sys
sys.path.insert(0, './embedding_w2v')
from embedding_w2v.embedding import Embedding
from Quick_Find import *
from Quick_Union import *


class db:

    def __init__(self):
        self.engine = None

    def loadHashMap(self, configname, storage_object, dim=64):
        # Create redis storage adapter
        #redis_object = Redis(host='localhost', port=6379, db=0)
        #self.redis_storage = RedisStorage(redis_object)
        if isinstance(storage_object, Redis):
            self.storage = RedisStorage(storage_object)
            print("using redis")
        elif isinstance(storage_object, pymongo.collection.Collection):
            self.storage = MongoStorage(storage_object)
            print("using mongodb")
        else:
            print("Please select a storage method")
            return

        # Get hash config from redis
        config = self.storage.load_hash_configuration(str(configname))

        if config is None:
            # Config is not existing, create hash from scratch, with 20 projections
            self.lshash = RandomBinaryProjectionTree(configname, 50, 1000)
            print('new configuration!')
        else:
            # Config is existing, create hash with None parameters
            self.lshash = RandomBinaryProjectionTree(None, None, None)
            # Apply configuration loaded from redis
            self.lshash.apply_config(config)

        # Create engine for feature space of 64 dimensions and use our hash.
        # This will set the dimension of the lshash only the first time, not when
        # using the configuration loaded from redis. Use redis storage to store
        # buckets.
        N = 1
        rank1 = RankNFilter(N)
        self.engine = Engine(
            dim, lshashes=[self.lshash], storage=self.storage, vector_filters=[rank1])
        # Finally store hash configuration in redis for later use
        self.storage.store_hash_configuration(self.lshash)

    def _countTotalNum(self, verbose=False):
        count = 0
        keys = self.engine.storage.get_all_bucket_keys(self.lshash.hash_name)
        print('key num: ' + str(len(keys)))
        for key in keys:
                bucket_content = self.engine.storage.get_bucket(
                    self.lshash.hash_name,
                    key,
                )
                #print bucket_content
                #return
                if verbose:
                    print('bucket size: ' + str(len(bucket_content)))
                count += len(bucket_content)

        print('total size: ' + str(count))
        return count

        # Do some stuff like indexing or querying with the engine...
    def indexing(self, vectors, names):
        #count = self._countTotalNum(verbose=False)
        for idx, vec in enumerate(vectors):
            #print type(vec)
            self.engine.store_vector(vec, names[idx])



    def querying(self, query):
        start_time = time.time()
        N_query = []
        #vector_filters = [NearestFilter(5)]
        for [q, name] in query:
            #print self.engine.candidate_count(q)
            N = self.engine.neighbours(q)
            for item in N:
                N_query.append([name, item[1], 1 - item[2]])#the list has the format: [(query_idx, query_func_name), (result_idx, result_func_name), similarity(which is 1-distance)]
            #
            # if idx%10000 == 9999:
            #     print('has done: ' + str(idx))
                #f = open('data/versiondetect/train_kNN_' + str(idx) + '.p', 'w')
                #p.dump(N_query, f)
                #f.close()
                #N_query = []
        print('all has done, time: %s' % (time.time() - start_time))
        #f = open('data/versiondetect/train_kNN_end.p', 'w')
        #p.dump(N_query, f)
        #f.close()
        #N_query = []
        return N_query

    def _format_redis_key(self, hash_name, bucket_key):
        return '{}{}'.format(self._format_hash_prefix(hash_name), bucket_key)

    def _format_hash_prefix(self, hash_name):
        return "nearpy_{}_".format(hash_name)

    def grouping(self):
        emb = Embedding()
        keys = self.engine.storage.get_all_bucket_keys(self.lshash.hash_name)
        for key in keys:
            #print key
            bucket_content = self.engine.storage.get_bucket(self.lshash.hash_name, key,)
            redis_key = self._format_redis_key(self.lshash.hash_name, key)
            vectors=[]
            for content in bucket_content:
                vectors.append(content[0])
            sim_matrix = emb.test_similarity(vectors, vectors)
            eleList = [i for i in range(len(sim_matrix))]
            eleNodeMap = genNodeList(eleList)
            for i in range(len(sim_matrix)):
                for j in range(i, len(sim_matrix)):
                    if sim_matrix[i][j] > 0.999999:
                        quickUnion((i,j), eleNodeMap)
            result = [i.num for i in eleNodeMap.values()]
            #print result
            groups=dict()
            for i in range(len(result)):
                if i == result[i]:
                    groups[i] = [vectors[i], bucket_content[i][1]]
                else:
                    groups[result[i]].append(bucket_content[i][1])
            self.engine.storage.redis_object.delete(redis_key)
            for element in groups:
                #print groups[element][0]
                #print groups[element][1:]
                self.engine.storage.store_vector(self.lshash.hash_name, key, groups[element][0], groups[element][1:])
                #self.engine.store_vector(groups[element][0], groups[element][1:])
            print "group one bucket"
