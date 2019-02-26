```
codesearch/
          __init__.py
          sslh/
                    __init__.py
                    SSLH_inference.py
                    test_SSLH_inference.py
                    my_util_load.py

          junto/
          data/
                    ssl/
                    versiondetect/
          fabp/
          lshknn/
          db.py                                           db class: build hashmap, index all the train data, query for test data
          lshknn.py                                       call db, input: the original data and query data we want to put in the hash table, output: the kNN file
          labels.py                                       Labels class
          main.py                                         call BP algorithm
```
