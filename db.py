from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections, RandomDiscretizedProjections, RandomBinaryProjectionTree, RandomBinaryProjectionTreeNode
import numpy as np
from redis import Redis
import pymongo
from nearpy.storage import RedisStorage, MongoStorage
from nearpy.filters import NearestFilter, UniqueFilter, RankNFilter, DistanceThresholdFilter
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

    def loadHashMap(self, confignames, storage_object, dim=64, projections=20):
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

        self.lshash = []
        for configname in confignames:
            # Get hash config from redis
            config = self.storage.load_hash_configuration(str(configname))

            if config is None:
                # Config is not existing, create hash from scratch, with 20 projections
                self.lshash.append(RandomBinaryProjections(configname, projections))
                print('new configuration!')
            else:
                # Config is existing, create hash with None parameters
                #lshash = RandomDiscretizedProjections(None, None, None)
                lshash = RandomBinaryProjections(None, None)
                # Apply configuration loaded from redis
                #config['minimum_result_size']=500
                lshash.apply_config(config)
                self.lshash.append(lshash)

        # Create engine for feature space of 64 dimensions and use our hash.
        # This will set the dimension of the lshash only the first time, not when
        # using the configuration loaded from redis. Use redis storage to store
        # buckets.
        distancethreshold = 0.1
        distancefilter = DistanceThresholdFilter(distancethreshold)
        N = 10
        rankfilter = RankNFilter(N)
        self.engine = Engine(
            dim, lshashes=self.lshash, storage=self.storage, vector_filters=[distancefilter, rankfilter])
        # Finally store hash configuration in redis for later use
        for lshash in self.lshash:
            self.storage.store_hash_configuration(lshash)

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
        #pdb.set_trace()
        start_time = time.time()
        N_query = []
        #vector_filters = [NearestFilter(5)]
        for [q, name, distance] in query:
            #print self.engine.candidate_count(q)
            N = self.engine.neighbours(q)
            if len(N) == 0:
                N_query.append([(name, distance), [], 1])
            for item in N:
                N_query.append([(name, distance), item[1], 1 - item[2]])#the list has the format: [(query_idx, query_func_name), (result_idx, result_func_name), similarity(which is 1-distance)]

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

    def recalculate_size(self, tree_node, depth):
        if depth == 50:
            bucket_content = self.engine.storage.get_bucket(self.lshash.hash_name, tree_node.bucket_key,)
            tree_node.vector_count = len(bucket_content)
            return
        count = 0
        #pdb.set_trace()
        for node_key in tree_node.childs:
            nodes = tree_node.childs[node_key]
            self.recalculate_size(nodes, depth+1)
            count += nodes.vector_count
        tree_node.vector_count = count

    def ungrouping(self):
        keys = self.engine.storage.get_all_bucket_keys(self.lshash.hash_name)
        for key in keys:
            redis_key = self._format_redis_key(self.lshash.hash_name, key)
            bucket_content = self.engine.storage.get_bucket(self.lshash.hash_name, key,)
            self.engine.storage.redis_object.delete(redis_key)
            for group in bucket_content:
                vector = group[0]
                ngramList = group[1]
                for ngram in ngramList:
                    self.engine.storage.store_vector(self.lshash.hash_name, key, vector, ngram)
            print "ungroup one bucket"

    def grouping(self):
        emb = Embedding()
        for lshash in self.lshash:
            lshashname = lshash.hash_name
            keys = self.engine.storage.get_all_bucket_keys(lshashname)
            for key in keys:
                #print key
                bucket_content = self.engine.storage.get_bucket(lshashname, key,)
                redis_key = self._format_redis_key(lshashname, key)
                eleList = [i for i in range(len(bucket_content))]
                eleNodeMap = genNodeList(eleList)
                for i in range(0, len(bucket_content), 5000):
                    vectors = [content[0] for content in bucket_content[i: min(i + 5000, len(bucket_content))]]
                    if i == 0:
                        sim_matrix = emb.test_similarity(vectors, vectors)
                        for idx in range(len(sim_matrix)):
                            for j in range(idx, len(sim_matrix)):
                                if sim_matrix[idx][j] > 0.99999999:
                                    quickUnion((idx, j), eleNodeMap)
                    else:
                        root_nodes = set()
                        for j in range(i):
                            root_nodes.add(eleNodeMap[j].parent.num)
                        root_nodes_idx = list(root_nodes)
                        root_emb = [bucket_content[j][0] for j in root_nodes_idx]
                        sim_matrix = emb.test_similarity(vectors, root_emb)
                        for idx in range(len(sim_matrix)):
                            for j in range(idx, len(sim_matrix[0])):
                                if sim_matrix[idx][j] > 0.99999999:
                                    quickUnion((idx + i, root_nodes_idx[j]), eleNodeMap)

                result = [i.parent.num for i in eleNodeMap.values()]
                #print result
                groups=dict()
                for i in range(len(result)):
                    if result[i] not in groups:
                        groups[result[i]] = [bucket_content[i][0]]
                    groups[result[i]].append(bucket_content[i][1])
                    #groups[result[i]].extend(bucket_content[i][1])
                self.engine.storage.redis_object.delete(redis_key)
                for element in groups:
                    #print groups[element][0]
                    #print groups[element][1:]
                    self.engine.storage.store_vector(lshashname, key, groups[element][0], groups[element][1:])
                    #self.engine.store_vector(groups[element][0], groups[element][1:])
                print "group one bucket " + str(len(self.engine.storage.get_bucket(lshashname, key,)))
        ##when the lshash is tree
        # print("before grouping: " + str(self.lshash.tree_root.vector_count))
        # self.recalculate_size(self.lshash.tree_root, 0)
        # print("after grouping:" + str(self.lshash.tree_root.vector_count))

    def buildTree(self):
        keys = self.engine.storage.get_all_bucket_keys(self.lshash.hash_name)
        for key in keys:
            bucket_content = self.engine.storage.get_bucket(self.lshash.hash_name, key,)
            for content in bucket_content:
                self.lshash.tree_root.insert_entry_for_bucket(key, 0)
