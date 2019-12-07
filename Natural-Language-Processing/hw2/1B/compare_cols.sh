#!/bin/bash
cut -d , -f 3,5 | egrep -ic "^[[:space:]]*([^[:space:]]+)([[:space:]][^,]+)?,([^,]+[[:space:]])?\1([[:space:]][^,]+)?$"
