## To generate embeddings for a library:
```
library folder structure
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

python main.py --mode=Indexing --path=None --name=None

## To build Faiss index
python main.py --mode=IndexingFaiss --path=None --name=None
(examples in /example_lib)

## To do query for Test Binaries
An example in /example, can try it out even without the LSH database (query result already included)
  1. first, unzip test_kNN_1112_2gram.p.zip, the original file is too large
  2. run the command:
      python main.py --mode=Searching --path=example --name=nginx-{openssl-1.0.1d}{zlib-1.2.11}

Output:
print out precision and recall
detailed prediction in a csv file under the example folder