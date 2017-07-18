#! /bin/sh
#
# make_main.sh
# Copyright (C) 2017 zq <zq@solitude>
#
# Distributed under terms of the MIT license.
#


rm main.out
gcc -o main.out main.c header.h lib.c -lm
./main.out
