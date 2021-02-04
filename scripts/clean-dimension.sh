#!/bin/bash

# Clean all *.tif files which are not fit into dimension of 448x448.
IFS=$'\n'
for i in $(find -f ./*.tif); do
    dim=$(identify -format "%wx%h" $i) 
    if [ $dim != "448x448" ]; then
	rm $i
    fi
done
