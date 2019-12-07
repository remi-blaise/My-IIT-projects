#!/bin/bash

# Replace this line with a sequence of shell commands connected with Unix pipes ("|")
cat /dev/stdin | grep -o "[[:alpha:]]*" | sort | uniq > /dev/stdout
