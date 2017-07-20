#ifndef HEADER_H
#define HEADER_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

extern int layer[3];

#define TRAIN_NUM 50000
#define VALIDATION_NUM  10000
#define TEST_NUM  10000
#define CLASS_NUM 10

#define BATCH_SIZE 50
#define LEARNING_RATE 0.09  //Learning rate for gradient descent
#define REGULARIATION_LAMBDA 0.01   //Regularization strength
#define EPOCHES 10000
#define INTERVAL_CHECK 200
#define TOERLATE_THRES 0.01

//DEBUG DOMAIN/////////////////////////
#define DEBUG_MAIN 0
#define DEBUG_PRINT_DATASET_VALUE 0
#define DEBUG_MAIN_PROCESS 0
///////////////////////////////////////

/* Function declarations */
void read_data(double* X_train, int* y_train, double* X_valid, int* y_valid);
void init_model_params(double* W1, double* b1, double* W2, double* b2);
void gen_batch(double* X_train, int* y_train, double* X_batch, int* y_batch);
double gaussrand();
void print_matrix(double* m, int row, int col);
void softmax(double* matrix, int row, int col);

void matrix_single_op(double* matrix, int row, int col, char* type);
double* transpose(double* matrix, int row, int col);
double* matrix_multi(double* Ma, double* Mb, int row_a, int col_a, int row_b, int col_b);
void matrix_add_vector(double* matrix, double* vec, int row, int col);
void matrix_single_const(double* matrix, double num, int row, int col, char* type);
int max_index(double* vec, int size);

double* matrix_sum(double* matrix, int row, int col);
void matrix_copy(double* Ma, double* Mb, int row, int col);
void matrix_add(double* Ma, double* Mb, int row, int col);
double* elemwise_multi(double* Ma, double* Mb, int row_a, int col_a, int row_b, int col_b);
double calculate_loss(double* X_batch, int* y_batch, double* W1, double* b1, double* W2, double* b2);
void read_model(double* W1, double* b1, double* W2, double* b2);
void save_model(double* W1, double* b1, double* W2, double* b2);
void read_test_data(double* X_test, int* y_test);
double test_accuracy(double* X_test, int* y_test, double* W1, double* b1, double* W2, double* b2);

#endif /* !HEADER_H */
