#!/bin/bash

# Replace this line with one or more shell commands
# You may write to intermediate text files on disk if necessary

for file in `ls test_*.txt`; do
    THE=`cat $file | grep -iw "the" -c`
    A=`cat $file | egrep -iw "a" -c`
    AN=`cat $file | egrep -iw "an" -c`
    echo $file,$THE,$A,$AN
done
