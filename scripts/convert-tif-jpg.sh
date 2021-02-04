#!/bin/bash

# Convert all *.tif file into jpegs.
IFS=$'\n'
for i in $(find -f ./*.tif); do 
    echo $i
    convert $i $i.jpg
done
