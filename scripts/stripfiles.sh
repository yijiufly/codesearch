#!/bin/bash


home=$PWD
#dir=../data/zlib/zlib-O2
dir=../data/versiondetect/test3/nginx
#dir=../data/nginx/nginx-openssl-zlib
cd $dir
#rm *.i64 *.asm
for FILE in *
do
  strip $FILE/$FILE -o $FILE/$FILE.strip
done
