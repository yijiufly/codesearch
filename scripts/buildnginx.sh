#!/bin/sh

# nginx=$1
# pcre=$2
# zlib=$3
# openssl=$4
cd /home/yijiufly/Downloads/codesearch/data/nginx-temp
nginx='nginx-1.10.3'
pcre='pcre-8.40'
zlib='zlib-1.2.7'
openssl='openssl-0.9.7e'

#wget https://nginx.org/download/$nginx.tar.gz && tar zxvf $nginx.tar.gz
wget https://ftp.pcre.org/pub/pcre/$pcre.tar.gz && tar xzvf $pcre.tar.gz
wget https://www.zlib.net/fossils/$zlib.tar.gz && tar xzvf $zlib.tar.gz
wget https://www.openssl.org/source/$openssl.tar.gz && tar xzvf $openssl.tar.gz

#rm -rf *.tar.gz
cd $nginx
rm -r objs
./configure --prefix=/home/yijiufly/Downloads/codesearch/data/nginx/nginx-1.10.3/install --user=www-data --group=www-data --build=Ubuntu --http-client-body-temp-path=/var/lib/nginx/body --http-fastcgi-temp-path=/var/lib/nginx/fastcgi --http-proxy-temp-path=/var/lib/nginx/proxy --http-scgi-temp-path=/var/lib/nginx/scgi --http-uwsgi-temp-path=/var/lib/nginx/uwsgi --with-openssl=../$openssl --with-openssl-opt=enable-ec_nistp_64_gcc_128 --with-openssl-opt=no-nextprotoneg --with-openssl-opt=no-weak-ssl-ciphers --with-openssl-opt=no-ssl3 --with-pcre=../$pcre --with-pcre-jit --with-zlib=../$zlib --with-file-aio --with-threads --with-http_addition_module --with-http_auth_request_module --with-http_dav_module --with-http_flv_module --with-http_gunzip_module --with-http_gzip_static_module --with-http_mp4_module --with-http_random_index_module --with-http_realip_module --with-http_slice_module --with-http_ssl_module --with-http_sub_module --with-http_stub_status_module --with-http_v2_module --with-http_secure_link_module --with-mail --with-mail_ssl_module --with-stream --with-stream_ssl_module --with-debug --with-cc-opt='-Wl,--exclude-libs=ALL -fvisibility=hidden -O3 -fPIE -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2'
make
make install
# mv objs/nginx ../nginx-{$openssl}{$zlib}
#
# cd ..
# strip -s nginx-{$openssl}{$zlib} -o strip/nginx-{$openssl}{$zlib}.strip
# rm -r $openssl/ $pcre/ $zlib/
