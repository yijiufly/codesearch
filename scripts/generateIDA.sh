#!/bin/bash


home=$PWD
dir=../data/zlib/zlib

cd $dir
rm *.i64 *.asm
for FILE in *
do
  cd $home
	cd ..
	#python preprocess.py --path data/nginx/nginx-openssl-zlib --name $FILE --out data/versiondetect/test2/idafiles/
  python preprocess.py --path data/zlib/zlib/ --name $FILE --out data/zlib/idafiles
done
