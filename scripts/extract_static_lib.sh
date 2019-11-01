#!/usr/bin/env bash

lipo -extract_family x86_64 $1 -o $1_thin
mkdir $1_objs
cd $1_objs
ar -x ../$1_thin
