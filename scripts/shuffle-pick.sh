#!/bin/bash

# Parameters
path=$1;
count=$2
dest_dir=$3

mkdir -p $dest_dir
mkdir -p $dest_dir/images
mkdir -p $dest_dir/masks

IFS=$'\n'
for i in $(find $path/images/*.jpg -type f -printf "%f\n" | shuf | tail -$count); do
    cp $path/images/$i $dest_dir/images/$i
    cp $path/masks/$i $dest_dir/masks/$i
done