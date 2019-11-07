codesearch/
          __init__.py
          data/
                    versiondetect/
          db.py                                           db class: build hashmap, index all the train data, query for test data
          lshknn.py                                       call db, input: the original data and query data we want to put in the hash table, output: the kNN file, build LSH database
          binary.py                                       Binary class and TestBinary class
          main.py                                         call BP algorithm
          preprocess.py                                   build callgraph using IDA Pro. convert .gdl callgraph to .dot file

## To generate embeddings for a library:
```
structure
lib/
      version1/
                  lib.so
      version2/
              ......
  ......
```
1. generate IDA files
  run python3 Gemini/gemini_feature_extraction_ST.py
2. generate function embedding
  run scripts/generateEMB.sh
3. generate callgraph, .dot file
  run python scripts/gencallgraph.py


## To build LSH database
1. install nearpy library
   cd Nearpy
   sudo python setup.py install

2. python lshknn.py

## To do query for Test Binaries
  run python main.py
