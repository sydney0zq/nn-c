/*
 * header.h
 * Copyright (C) 2017 zq <zq@solitude>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef HEADER_H
#define HEADER_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

struct data_box{
    double xc;      //The coordinate x and y, must be double type
    double yc;
    int label;
};

#define TRAIN_NUM 200
#define LEARNING_RATE 0.01  //Learning rate for gradient descent
#define REGULARIATION_LAMBDA 0.01   //Regularization strength
#define ITER_TIMES 20000

//DEBUG DOMAIN/////////////////////////
#define DEBUG_PRINT_INIT_VALUE 1
#define DEBUG_TRANS_DISPLAY 0
///////////////////////////////////////

/* The size of W1 is layer[0]*layer[1]
 * The size of b1 is layer[1]
 * The size of W2 is layer[1]*layer[2]
 * The size of b2 is layer[2]
 */
extern double* z1;
extern double* z2;
extern double* a1;
extern double* W1;
extern double* b1;
extern double* W2;
extern double* b2;
extern double* exp_scores;
extern double* probs;

/* Function declarations */
void read_data(char* path, struct data_box* ptr_train_data);
void debug_print_init_value(double* W1, double* b1, double* W2, double* b2);
double gaussrand();

#endif /* !HEADER_H */
