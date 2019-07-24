from redis import Redis
from lshknn import getHashMap, doSearch
import pickle as p
from Quick_Find import *
from Quick_Union import *
from embedding_w2v.embedding import Embedding
emb = Embedding()
redis_object = Redis(host='localhost', port=6379, db=1)
hashMap = getHashMap("filtedleaf-2gram", redis_object, dim=128)
keys = hashMap.engine.storage.get_all_bucket_keys(hashMap.lshash.hash_name)
vectors=[]
for key in keys:
    bucket_content = hashMap.engine.storage.get_bucket(hashMap.lshash.hash_name, key,)
    if key == '1111101011000111101100001110011011000000000110110010001':
#         for item in bucket_content:
#             gramlist = item[1]
#             if gramlist[0][2]==('ec_GFp_simple_dbl', 'BN_mod_lshift1_quick'):
#                 print key
#                 print item
#                 print
#             vectors.append(item[0])
# print len(vectors)
# sim_matrix = emb.test_similarity(vectors, vectors)
# print sim_matrix
# eleList = [i for i in range(len(sim_matrix))]
# eleNodeMap = genNodeList(eleList)
# for i in range(len(sim_matrix)):
#     for j in range(i, len(sim_matrix)):
#         if sim_matrix[i][j] > 0.99999999:
#             quickUnion((i,j), eleNodeMap)
# result = [i.parent.num for i in eleNodeMap.values()]
# print result
        vectors=[]
        for content in bucket_content:
            vectors.append(content[0])
        sim_matrix = emb.test_similarity(vectors, vectors)
        eleList = [i for i in range(len(sim_matrix))]
        eleNodeMap = genNodeList(eleList)
        for i in range(len(sim_matrix)):
            for j in range(i, len(sim_matrix)):
                print sim_matrix[i][j]
                if sim_matrix[i][j] > 0.99999999:
                    quickUnion((i,j), eleNodeMap)
        result = [i.parent.num for i in eleNodeMap.values()]
        print result
