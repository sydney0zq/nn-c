/*
 * main.c
 * Copyright (C) 2017 zq <zq@solitude>
 *
 * Distributed under terms of the MIT license.
 */

#include "header.h"

/* The size of W1 is layer[0]*layer[1]
 * The size of b1 is layer[1]
 * The size of W2 is layer[1]*layer[2]
 * The size of b2 is layer[2]
 * The size of z1 is TRAIN_NUM * layer[1]
 * The size of z2 is TRAIN_NUM * layer[2]
 * The size of exp_scores is TRAIN_NUM * layer[2]  == exp(z2)
 * The size of probs is TRAIN_NUM * layer[2]
 * The size of delta3 is same with probs
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
extern int layer[3];
int layer[3] = {2, 3, 2};


int main(char argc, char **argv){
    struct data_box train_data[TRAIN_NUM];
    struct data_box* ptr_train_data;
    int row, col, i, j, iter;
    double* X  = (double *)malloc(TRAIN_NUM*layer[0]*sizeof(double));

    ptr_train_data = train_data;
    read_data("./dataset.txt", ptr_train_data, X);

    // Build model: Learns paramters for the NN and returns the model
    double* W1 = (double *)malloc(layer[0]*layer[1]*sizeof(double));
    double* b1 = (double *)malloc(layer[1]*sizeof(double));
    double* W2 = (double *)malloc(layer[1]*layer[2]*sizeof(double));
    double* b2 = (double *)malloc(layer[2]*sizeof(double));

    double* z1;
    double* a1;
    double* z2;
    double* exp_scores;
    double* probs = (double *)malloc(TRAIN_NUM * layer[2] * sizeof(double));
    double* delta3 = (double *)malloc(TRAIN_NUM * layer[2] * sizeof(double));

    // Initialize the paramters
    for (row = 0; row < layer[0]; row++)
        for (col = 0; col < layer[1]; col++)
            *(W1 + row*layer[1] + col) = gaussrand()/sqrt(layer[0]);
    for (i = 0; i < layer[1]; i++)
        *(b1 + i) = 0;
    for (row = 0; row < layer[1]; row++)
        for (col = 0; col < layer[2]; col++)
            *(W2 + row*layer[2] + col) = gaussrand()/sqrt(layer[1]);
    for (i = 0; i < layer[2]; i++)
        *(b2 + i) = 0;

    if (DEBUG_PRINT_INIT_VALUE) debug_print_init_value(W1, b1, W2, b2);
    
    // Gradient descent for each batch
    for (iter = 0; iter < ITER_TIMES; iter++){
        // Forward propagation
	    // z1 = X.dot(W1) + b1 and a1 = np.tanh(z1)
        z1 = matrix_multi(X, W1, TRAIN_NUM, layer[0], layer[0], layer[1]);  // TODO free
        matrix_add_vector(z1, b1, TRAIN_NUM, layer[1]);
        matrix_single_op(z1, TRAIN_NUM, layer[1], "tanh");
        a1 = z1;
        
        // z2 = a1.dot(W2) + b2 and exp_scores = np.exp(z2)
        z2 = matrix_multi(a1, W2, TRAIN_NUM, layer[1], layer[1], layer[2]); // TODO free
        matrix_add_vector(z2, b2, TRAIN_NUM, layer[2]);
        matrix_single_op(z2, TRAIN_NUM, layer[2], "exp");
        exp_scores = z2;
        
        // probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        // Caculate the probability
        double tmp;
        for (row = 0; row < TRAIN_NUM; row++){
            tmp = 0;
            // Sum the whole row value
            for (col = 0; col < layer[2]; col++){
                tmp += *(exp_scores + row*layer[2] + col);
            }
            for (col = 0; col < layer[2]; col++){
                *(probs + row*layer[2] + col) = *(exp_scores + row*layer[2] + col) / tmp;
            }
        }

        // Backpropagation
        // delta3 = probs
        for (row = 0; row < TRAIN_NUM; row++){
            for (col = 0; col < layer[2]; col++){
                *(delta3 + row*layer[2] + col) = *(probs + row*layer[2] + col);
            }
        }
        
        double* dW2; double* db2; double* dW1; double* db1; double* delta2;
        double* left;
        double* right;

        // delta3[range(num_examples), y] -= 1
        for (row = 0; row < TRAIN_NUM; row++){
            *(delta3 + row*layer[2] + (ptr_train_data+row*3+2)->label) -= 1;
        }
        // dW2 = (a1.T).dot(delta3)
        a1 = transpose(a1, TRAIN_NUM, layer[1]);
        dW2 = matrix_multi(a1, delta3, layer[1], TRAIN_NUM, TRAIN_NUM, layer[2]);   //TODO free
        // db2 = np.sum(delta3, axis=0, keepdims=True)
        db2 = matrix_sum(delta3, TRAIN_NUM, layer[2]);  //TODO free

        // delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        W2 = transpose(W2, layer[1], layer[2]);
        left = matrix_multi(delta3, W2, TRAIN_NUM, layer[2], layer[2], layer[1]);
        matrix_single_op(a1, TRAIN_NUM, layer[1], "pow2");
        for (i = 0; i < TRAIN_NUM; i++){
            for (j = 0; j < layer[1]; j++){
                *(a1 + i*layer[1] + j) = 1 - 1*(*(a1 + i*layer[1] + j));
            }
        }
        right = a1;

        delta2 = elemwise_multi(left, right, TRAIN_NUM, layer[1], TRAIN_NUM, layer[1]);

        // dW1 = np.dot(X.T, delta2)
        X = transpose(X, TRAIN_NUM, layer[0]); 
        dW1 = matrix_multi(X, delta2, layer[0], TRAIN_NUM, TRAIN_NUM, layer[1]);
        // db1 = np.sum(delta2, axis=0)
        db1 = matrix_sum(delta2, TRAIN_NUM, layer[1]);
        X = transpose(X, layer[0], TRAIN_NUM); 

        // Add regularization terms (b1 and b2 don't have regularization terms)
        double * reg_matrix;
        // dW2 += reg_lambda * W2
        // dW1 += reg_lambda * W1
        W2 = transpose(W2, layer[2], layer[1]);
        reg_matrix = matrix_single_const(W2, REGULARIATION_LAMBDA, layer[1], layer[2], "multi");
        matrix_add(dW2, reg_matrix, layer[1], layer[2]);
        free(reg_matrix);

        reg_matrix = matrix_single_const(W1, REGULARIATION_LAMBDA, layer[0], layer[1], "multi");
        matrix_add(dW1, reg_matrix, layer[0], layer[1]);
        free(reg_matrix);

        // Gradient descent parameter update
        // W1 += -epsilon * dW1
        // b1 += -epsilon * db1
        // W2 += -epsilon * dW2
        // b2 += -epsilon * db2
        reg_matrix = matrix_single_const(dW1, (-1) * LEARNING_RATE, layer[0], layer[1], "multi");
        matrix_add(W1, reg_matrix, layer[0], layer[1]);
        free(reg_matrix);

        reg_matrix = matrix_single_const(db1, (-1) * LEARNING_RATE, 1, layer[1], "multi");
        matrix_add(b1, reg_matrix, 1, layer[1]);
        free(reg_matrix);
        
        reg_matrix = matrix_single_const(dW2, (-1) * LEARNING_RATE, layer[1], layer[2], "multi");
        matrix_add(W2, reg_matrix, layer[1], layer[2]);
        free(reg_matrix);

        reg_matrix = matrix_single_const(db2, (-1) * LEARNING_RATE, 1, layer[2], "multi");
        matrix_add(b2, reg_matrix, 1, layer[2]);
        free(reg_matrix);

        if (i % 100 == 0){
            printf("Loss after iteration %d: %f", 
                    i, calculate_loss(ptr_train_data, X, W1, b1, W2, b2));
        }
    }// End of gradient descent

    
    
    //free(W1, b1, W2, b2);

    return 0;
}
