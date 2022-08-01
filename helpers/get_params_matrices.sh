#!/bin/bash

rm -f ./params.txt
touch ./params.txt

for filename in ~/matrices/norm/*;
do
    echo -e "$filename" >> ./params.txt
    octave csr3params.m $filename >> ./params.txt
    echo -e "\n" >> ./params.txt
done

for filename in ~/matrices/norm/normnew/*;
do
    echo -e "$filename" >> ./params.txt
    octave csr3params.m $filename >> ./params.txt
    echo -e "\n" >> ./params.txt
done