#!/bin/sh


zlib=$3

for zlib in '1.2.11' '1.2.10' '1.2.9' '1.2.8' '1.2.7.3' '1.2.7.2' '1.2.7.1' '1.2.7'
do
  wget https://www.zlib.net/fossils/zlib-$zlib.tar.gz && tar xzvf zlib-$zlib.tar.gz
  #echo $zlib
  rm -rf *.tar.gz
  cd zlib-$zlib
  ./configure --prefix=/usr/local/zlib
  make
  cd ..
  cp zlib-$zlib/libz.so.$zlib zlib
done
