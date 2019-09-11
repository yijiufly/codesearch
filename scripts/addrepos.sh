#!/bin/sh
filepath='/data/repo/opensslversions'
repos=`cat $filepath`
for r in $repos
do
  python detector.py index -d https://github.com/openssl/openssl/$repo
done
