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
#define ITER_TIMES 100000

//DEBUG DOMAIN/////////////////////////
#define DEBUG_MAIN 0
#define DEBUG_PRINT_INIT_VALUE 0
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
void read_data(char* path, struct data_box* ptr_train_data, double* X);
void debug_print_init_value(double* W1, double* b1, double* W2, double* b2);
double gaussrand();
double* transpose(double* matrix, int row, int col);
double* matrix_multi(double* Ma, double* Mb, int row_a, int col_a, int row_b, int col_b);
void matrix_add_vector(double* matrix, double* vec, int row, int col);
void matrix_single_op(double* matrix, int row, int col, char* type);
double* matrix_sum(double* matrix, int row, int col);
void matrix_add(double* Ma, double* Mb, int row, int col);
double* elemwise_multi(double* Ma, double* Mb, int row_a, int col_a, int row_b, int col_b);
double* matrix_single_const(double* matrix, double num, int row, int col, char* type);
double calculate_loss(struct data_box* ptr_train_data, double* X, double* W1, double* b1, double* W2, double* b2);
void print_matrix(double* m, int row, int col);

#endif /* !HEADER_H */
