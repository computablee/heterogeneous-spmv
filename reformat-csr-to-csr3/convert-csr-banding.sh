#!/bin/bash

export PROJECT=~

rm -rf $PROJECT/matrices/csr-bandk
mkdir $PROJECT/matrices/csr-bandk

for filename in $PROJECT/matrices/norm/*.mtx.csr;
do
		export filefolder="$(dirname "$filename")"
		export basename="$(basename -- $filename)"
		echo "reformatting $filename, outputting to $filefolder/../csr-bandk/$basename.bandk"
		./reformat $filename $filefolder/../csr-bandk/$basename.bandk
done