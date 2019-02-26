from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections, RandomDiscretizedProjections
import numpy as np
from redis import Redis
from nearpy.storage import RedisStorage
from nearpy.filters import NearestFilter, UniqueFilter
import pickle as p


class db:

    def __init__(self):
        self.engine = None

    def loadHashMap(self, configname):
        # Create redis storage adapter
        redis_object = Redis(host='localhost', port=6379, db=0)
        self.redis_storage = RedisStorage(redis_object)

        # Get hash config from redis
        config = self.redis_storage.load_hash_configuration(configname)

        if config is None:
            # Config is not existing, create hash from scratch, with 10 projections
            self.lshash = RandomBinaryProjections(configname, 50)
            print 'new configuration!'
        else:
            # Config is existing, create hash with None parameters
            self.lshash = RandomBinaryProjections(None, None)
            # Apply configuration loaded from redis
            self.lshash.apply_config(config)

        # Create engine for feature space of 64 dimensions and use our hash.
        # This will set the dimension of the lshash only the first time, not when
        # using the configuration loaded from redis. Use redis storage to store
        # buckets.
        N = 100
        nearest = NearestFilter(N)
        self.engine = Engine(
            64, lshashes=[self.lshash], storage=self.redis_storage, vector_filters=[nearest])

    def _countTotalNum(self, verbose=False):
        count = 0
        keys = self.engine.storage.get_all_bucket_keys(self.lshash.hash_name)
        print('key num: ' + str(len(keys)))
        for key in keys:
                bucket_content = self.engine.storage.get_bucket(
                    self.lshash.hash_name,
                    key,
                )
                if verbose:
                    print('bucket size: ' + str(len(bucket_content)))
                count += len(bucket_content)

        print('total size: ' + str(count))
        return count

        # Do some stuff like indexing or querying with the engine...
    def indexing(self, vectors, names):
        count = self._countTotalNum()
        for idx, vec in enumerate(vectors):
            self.engine.store_vector(vec, (count + idx , names[idx]))
        # Finally store hash configuration in redis for later use
        self.redis_storage.store_hash_configuration(self.lshash)

    def querying(self, query_vectors, names):
        N_query = []
        count = self._countTotalNum()
        for idx, q in enumerate(query_vectors):
            #print self.engine.candidate_count(q)
            N = self.engine.neighbours(q)
            for item in N:
                N_query.append([(idx + count, names[idx]), item[1], 1 - item[2]])#the list has the format: [(query_idx, query_func_name), (result_idx, result_func_name), similarity(which is 1-distance)]

            if idx%10000 == 9999:
                print('has done: ' + str(idx))
                #f = open('data/versiondetect/train_kNN_' + str(idx) + '.p', 'w')
                #p.dump(N_query, f)
                #f.close()
                #N_query = []
        print('all has done: ')
        #f = open('data/versiondetect/train_kNN_end.p', 'w')
        #p.dump(N_query, f)
        #f.close()
        #N_query = []
        return N_query
