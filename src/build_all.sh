#!/bin/bash
if [[ $# -eq 1 ]]; then
    if [[ `grep -c "int main" $1` -ge 1 ]] ; then
        echo "compiling $1"
        g++ -std=c++11 -ggdb `pkg-config --cflags opencv` -o `basename $1 .cpp` $1 `pkg-config --libs opencv`;
    else
        echo "not compiling $1, no main function found"
    fi
else
    for i in *.cpp; do
        if [[ `grep -c "int main" $i` -ge 1 ]] ; then
            echo "compiling $i"
            g++ -std=c++11 -ggdb `pkg-config --cflags opencv` -o `basename $i .cpp` $i `pkg-config --libs opencv`;
        else
            echo "not compiling $i, no main function found"
        fi
    done
fi