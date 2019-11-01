#!/bin/sh


# zlib_home=/home/yijiufly/Downloads/codesearch/data/zlib
# cd $zlib_home
# for zlib in '1.2.11' '1.2.10' '1.2.9' '1.2.8' '1.2.7.3' '1.2.7.2' '1.2.7.1'
# do
#   wget https://www.zlib.net/fossils/zlib-$zlib.tar.gz && tar xzvf zlib-$zlib.tar.gz
#   #echo $zlib
#   rm -rf *.tar.gz
#   cd zlib-$zlib
#   CFLAGS="-O2 -fomit-frame-pointer -pipe " CC="cc" \
# 		./configure
#   make libz.a
#   cd ..
#   mkdir zlib-O2/zlib-$zlib
#   cp zlib-$zlib/libz.a zlib-O2/zlib-$zlib
#   mkdir zlib-O2/zlib-$zlib/objfiles
#   cd zlib-O2/zlib-$zlib/objfiles
#   ar -x ../libz.a
#   cd $zlib_home
# done


openssl_home=/home/yijiufly/Downloads/codesearch/data/
cd $openssl_home
for openssl in 'openssl-1.0.1q'  'openssl-1.0.2m'  'openssl-1.1.0h'  'openssl-1.0.1p'  'openssl-0.9.8t'  'openssl-0.9.8r'  'openssl-0.9.7b'  'openssl-0.9.7m'  'openssl-1.0.1f'  'openssl-1.0.0t'  'openssl-1.0.1n'  'openssl-1.0.2a'  'openssl-1.0.1k'  'openssl-1.1.0d'  'openssl-1.0.1g'  'openssl-1.0.0a'  'openssl-1.0.1b'  'openssl-0.9.8h'  'openssl-1.0.0r'  'openssl-1.0.0e'  'openssl-1.0.0m'  'openssl-1.0.1j'  'openssl-1.0.2d'  'openssl-0.9.8l'  'openssl-1.1.0c'  'openssl-0.9.7j'  'openssl-0.9.7d'  'openssl-0.9.8m'  'openssl-1.0.0p'  'openssl-0.9.8a'  'openssl-1.0.1c'  'openssl-1.0.1h'  'openssl-1.0.2l'  'openssl-1.1.0g'  'openssl-1.0.1s'  'openssl-1.0.2q'  'openssl-1.0.2i'  'openssl-1.0.1m'  'openssl-1.0.1i'  'openssl-1.0.2j'  'openssl-1.0.1t'  'openssl-0.9.8c'  'openssl-1.0.0'  'openssl-1.0.0n'  'openssl-0.9.8d'  'openssl-0.9.7i'  'openssl-1.0.1d'  'openssl-0.9.7l'  'openssl-1.0.2n'  'openssl-1.0.2b'  'openssl-1.0.1a'  'openssl-1.1.0e'  'openssl-1.0.1e'  'openssl-1.0.0f'  'openssl-0.9.8b'  'openssl-1.0.0b'  'openssl-1.1.0b'  'openssl-0.9.7g'  'openssl-1.0.0s'  'openssl-1.0.0q'  'openssl-0.9.8f'  'openssl-0.9.7h'  'openssl-0.9.7a'  'openssl-0.9.8g'  'openssl-0.9.7k'  'openssl-0.9.8w'  'openssl-1.1.0f'  'openssl-1.0.1o'  'openssl-1.0.0k'  'openssl-0.9.8'  'openssl-0.9.8p'  'openssl-1.0.1'  'openssl-1.0.0l'  'openssl-1.0.0d'  'openssl-0.9.7e'  'openssl-1.1.0a'  'openssl-1.0.2k'  'openssl-1.0.2h'  'openssl-0.9.7c'  'openssl-0.9.7f'  'openssl-1.0.2c'  'openssl-0.9.8u'  'openssl-0.9.8i'  'openssl-0.9.8n'  'openssl-0.9.8j'  'openssl-1.0.0h'  'openssl-1.0.2'  'openssl-1.1.0'  'openssl-0.9.8k'  'openssl-0.9.7'  'openssl-1.0.0o'  'openssl-1.0.2o'  'openssl-1.0.1r'  'openssl-0.9.8e'  'openssl-0.9.8y'  'openssl-0.9.8s'  'openssl-1.0.0g'  'openssl-1.0.2f'  'openssl-0.9.8v'  'openssl-0.9.8q'  'openssl-1.0.2g'  'openssl-1.0.2e'  'openssl-0.9.8x'  'openssl-1.0.0j'  'openssl-1.0.0c'  'openssl-0.9.8o'  'openssl-1.0.0i'  'openssl-1.0.1l'
do
  wget https://www.openssl.org/source/$openssl.tar.gz && tar xzvf $openssl.tar.gz
  rm -rf *.tar.gz
  cd $openssl
  ./config no-shared
  make
  cd ..
  cp $openssl/libcrypto.a openssl/$openssl
  mkdir openssl/$openssl/objfiles-libcrypto
  cd openssl/$openssl/objfiles-libcrypto
  ar -x ../libcrypto.a
  cd $openssl_home
  cp $openssl/libssl.a openssl/$openssl
  mkdir openssl/$openssl/objfiles-libssl
  cd openssl/$openssl/objfiles-libssl
  ar -x ../libssl.a
  cd $openssl_home
done
