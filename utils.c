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
double calculate_loss(double* ptr_train_data, double* W1, double* b1, double* W2, double* b2){
    // Forward propagation to calculate our prediction
    double* z1 = (double *)malloc(layer[1] * TRAIN_NUM * sizeof(double));
    double* z2 = (double *)malloc(layer[2] * TRAIN_NUM * sizeof(double));
    double* a1 = (double *)malloc(layer[1] * TRAIN_NUM * sizeof(double));
    double* prob = (double *)malloc(layer[2] * sizeof(double));
    int row, col, i;

    // z1 = X.dot(W1) + b1
    for (row = 0; row < TRAIN_NUM; row++){
        for (col = 0; col < layer[1]; col++){
            *(z1 + row*layer[1] + col) = 
                (ptr_train_data+row)->xc * W1[0][col] + 
                (ptr_train_data+row)->yc * W1[1][col];
            *(z1 + row*layer[1] + col) += b1[col];
        }
    }

    // a1 = np.tanh(z1)
    for (row = 0; row < TRAIN_NUM; row++){
        for (col = 0; col < layer[1]; col++){
            *(a1 + row*layer[1] + col) = tanh(z1 + row*layer[1] + col);
        }
    }

    // z2 = a1.dot(W2) + b2 and exp_scores = np.exp(z2)
    // z2's size: TRAIN_NUM * layer[2]
    double tmp;
    for (row = 0; row < TRAIN_NUM; row++){
        for (col = 0; col < layer[2]; col++){
            tmp = 0;
            for (i = 0; i < layer[1]; i++){
                tmp += a1[row][i] * W1[i][col];
            }
            *(z2 + row*layer[2] + col) += b2[col];
            *(z2 + row*layer[2] + col) = exp(*(z2 + row*layer[2] + col));
        }
    }

    // Caculate the probability
    for (row = 0; row < TRAIN_NUM; row++){
        tmp = 0;
        // Sum the whole row value
        for (col = 0; col < layer[2]; col++){
            tmp += *(z2 + row*layer[2] + col);
        }
        for (col = 0; col < layer[2]; col++){
            *(z2 + row*layer[2] + col) /= tmp;
        }
    }

    // Calculating the loss
    double loss=0;
    for (row = 0; row < TRAIN_NUM; row++){
        loss += log(*(z2 + row*layer[2]));

    }
    

}

