#!/bin/bash

cols=$1

count=0
files=""
prev_row=""
for i in $(find ./*.jpg -type f | sort -V); do
    ((count=count+1))
    files="$files $i"

    n=$(($count%$cols))
    if [[ $n -eq 0 ]]; then
        echo $files
        curr_row="temp-row-$count.jpg"
        convert +append $files $curr_row
        if [[ "$prev_row" != "" ]]; then
            echo "$prev_row"
            convert -append $prev_row $curr_row $curr_row
            rm $prev_row
        fi
        prev_row=$curr_row
        files=""
    fi
done
