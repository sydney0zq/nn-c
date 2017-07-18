/*
 * utils.c
 * Copyright (C) 2017 zq <zq@solitude>
 *
 * Distributed under terms of the MIT license.
 */


/* The size of W1 is layer[0]*layer[1]
 * The size of b1 is layer[1]
 * The size of W2 is layer[1]*layer[2]
 * The size of b2 is layer[2]
 */
#include "header.h"

double calculate_loss(double* ptr_train_data, double* X, double* W1, double* b1, double* W2, double* b2){
    // Forward propagation to calculate our prediction
    double z1;
    
    double* probs = (double *)malloc(TRAIN_NUM * layer[2] * sizeof(double));
    int row, col, i;

    // z1 = X.dot(W1) + b1 and a1 = np.tanh(z1)
    z1 = matrix_multi(X, W1, TRAIN_NUM, layer[0], layer[0], layer[1]);  // TODO free
    matrix_add_vector(z1, b1, TRAIN_NUM, layer[1]);
    matrix_single_op(z1, TRAIN_NUM, layer[1], "tanh");

    // z2 = a1.dot(W2) + b2 and exp_scores = np.exp(z2)
    z2 = matrix_multi(a1, W2, TRAIN_NUM, layer[1], layer[1], layer[2]); // TODO free
    matrix_add_vector(z2, b2, TRAIN_NUM, layer[2]);
    matrix_single_op(z2, TRAIN_NUM, layer[2], "exp");

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

    // Calculating the loss
    double loss=0;
    for (i = 0; i < TRAIN_NUM; i++){
        loss += ( -log(*(probs+i*layer[2]+(ptr_train_data+i)->label)) );
    }
    // Add regulatization term to loss (optional)
    // data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return (1.0*loss) / TRAIN_NUM
}




















