#!/bin/sh
home=/home/yijiufly/Downloads/codesearch/data/zlib/zlib-O2
cd $home
files=$(ls $PWD)
for file in $files
do
	if test -d $file
	then
		cd $file
		mkdir emb_sharedlib
		subfiles=$(ls *.ida*)
		for subfile in $subfiles
		do
			mv $subfile emb_sharedlib/$subfile
		done
		cd ..
	fi
done
