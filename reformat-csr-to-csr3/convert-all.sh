#!/bin/bash

rm -rf $PROJECT/matrices/csr3 
mkdir $PROJECT/matrices/csr3
rm -rf $PROJECt/matrices/csr2
mkdir $PROJECT/matrices/csr2

for filename in $PROJECT/matrices/norm/*.mtx.csr;
do
		export filefolder="$(dirname "$filename")"
		export basename="$(basename -- $filename)"
		echo "reformatting $filename, outputting to $filefolder/../csr3/$basename.rcm.csr3"
		./reformat-auto $filename $filefolder/../csr3/$basename.rcm.csr3
		echo "reformatting $filename, outputting to $filefolder/../csr2/$basename.rcm.csr2"
		./reformat $filename $filefolder/../csr2/$basename.rcm.csr2 96
done