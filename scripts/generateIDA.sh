#!/bin/bash


home=$PWD
#dir=../data/zlib/zlib-O2
dir=../data/openssl
#dir=../data/nginx/nginx-openssl-zlib
cd $dir
#rm *.i64 *.asm
for FILE in *
do
  cd $home
	cd ..
	#python preprocess.py --path data/nginx/nginx-openssl-zlib --name $FILE --out data/versiondetect/test3/nginx/
  #python preprocess.py --path data/zlib/zlib-O2/ --name $FILE --out data/zlib/idafilesO2
  #python preprocess.py --path data/openssl/$FILE --name libssl.so --out data/openssl/$FILE
  python preprocess.py --path data/openssl/$FILE --name libcrypto.so --out data/openssl/$FILE
done
