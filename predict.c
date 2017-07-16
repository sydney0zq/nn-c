/*
 * predict.c
 * Copyright (C) 2017 zq <zq@solitude>
 *
 * Distributed under terms of the MIT license.
 */

/* The size of W1 is layer[0]*layer[1]
 * The size of b1 is layer[1]
 * The size of W2 is layer[1]*layer[2]
 * The size of b2 is layer[2]
 */

/* Helper function to predict an output
 */
int predict(double* W1, double* b1, double* W2, double* b2, struct data_box* x){
    int 
    // Forward propagation

    // z1 = x.dot(W1) + b1
    for (col = 0; col < layer[1]; col++){
        *(z1 + row*layer[1] + col) = 
        (ptr_train_data+row)->xc * W1[0][col] + 
        (ptr_train_data+row)->yc * W1[1][col];
        *(z1 + row*layer[1] + col) += b1[col];
    }

}
