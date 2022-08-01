#!/bin/bash

rm -f ./overhead_csr2.txt
touch ./overhead_csr2.txt

for filename in $PROJECT/matrices/csr2/*;
do
    echo -e "$filename" >> ./overhead_csr2.txt
    octave calculate_overhead.m $filename csr2 >> ./overhead_csr2.txt
    echo -e "\n" >> ./overhead_csr2.txt
done
