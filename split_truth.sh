#!/bin/bash

input=$1
mkdir truth/
for dirname in $(sed -n '2,$p' $input | cut -f3 -d, | sort -u); do
	dataname=truth/$(echo "$dirname" | tr / _).csv
	echo $dataname
	head -n1 "$input" > "$dataname"
	grep "$dirname" "$input" >> "$dataname"
done
