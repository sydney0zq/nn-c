#! /bin/sh
#
# make_main.sh
# Copyright (C) 2017 zq <zq@solitude>
#
# Distributed under terms of the MIT license.
#


gcc -o main.out main.c header.h read_data.c lib.c calculate_loss.c -lm
