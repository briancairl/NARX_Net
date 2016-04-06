# !/bin/bash
#
# Need to make sure to define DFANN_BFS when compiling with flann
#
# Need to make sure you have the "-lm" flag for math functions
#
gcc -o narx_test.o -DFLOATFANN=1 -DFANN_BFS -I../include -I../thirdparty/fann/include ../thirdparty/fann/src/*.c ../src/*.c example.c -lm