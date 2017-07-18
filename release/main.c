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
    int row, col, i, iter;
    double* X  = (double *)malloc(TRAIN_NUM*layer[0]*sizeof(double);)

    ptr_train_data = train_data;
    read_data("./dataset.txt", ptr_train_data, X);

    // Build model: Learns paramters for the NN and returns the model
    double* W1 = (double *)malloc(layer[0]*layer[1]*sizeof(double));
    double* b1 = (double *)malloc(layer[1]*sizeof(double));
    double* W2 = (double *)malloc(layer[1]*layer[2]*sizeof(double));
    double* b2 = (double *)malloc(layer[2]*sizeof(double));

    double* z1, a1, z2, exp_scores;
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
        matrix_single_op(z1, TRAIN_NUM, layer[1], "tanh")
        a1 = z1;
        
        // z2 = a1.dot(W2) + b2 and exp_scores = np.exp(z2)
        z2 = matrix_multi(a1, W2, TRAIN_NUM, layer[1], layer[1], layer[2]); // TODO free
        matrix_add_vector(z2, b2, TRAIN_NUM, layer[2]);
        matrix_single_op(z2, TRAIN_NUM, layer[2], "exp");
        exp_scores = z2;
        
        // probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        // Caculate the probability
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
        
        double* dW2, db2, dW1, db1;
        double* tmp;

        // delta3[range(num_examples), y] -= 1
        for (row = 0; row < TRAIN_NUM; row++){
            *(delta3 + row*layer[2] + (ptr_train_data+row*3+2)->label) -= 1;
        }
        // dW2 = (a1.T).dot(delta3)
        a1 = transpose(a1);
        dW2 = matrix_multi(a1, delta3, layer[1], TRAIN_NUM, TRAIN_NUM, layer[2]);   //TODO free
        // db2 = np.sum(delta3, axis=0, keepdims=True)
        db2 = matrix_sum(delta3, TRAIN_NUM, layer[2]);  //TODO free

        // delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = matrix_multi()










    }// End of gradient descent

    
    
    //free(W1, b1, W2, b2);

    return 0;
}



/*
 * Read data from a file which is generated from python script
 * gen_train_data.py
 */
void read_data(char* path, struct data_box* ptr_train_data, double* X){
    FILE* fp;
    struct data_box* base_ptr = ptr_train_data;
    int i;

    if ((fp = fopen(path, "r")) != NULL){
        for (i = 0; i < TRAIN_NUM; i++, ptr_train_data++){
            fscanf(fp, "%lf %lf %d", 
                &(ptr_train_data->xc), &(ptr_train_data->yc), &(ptr_train_data->label));
            *(X + i*layer[0]) = ptr_train_data->xc;
            *(X + i*layer[0] + 1) = ptr_train_data->yc;
        }
    }else{
        printf("Read File Error, Now exiting...");
        exit(1);
    }
    fclose(fp);
    ptr_train_data = base_ptr;
    //for (i = 0; i < TRAIN_NUM; i++, ptr_train_data++){
    //    printf("%f, %f, %d\n", ptr_train_data->xc, ptr_train_data->yc, ptr_train_data->label);   
    //}
}

void debug_print_init_value(double* W1, double* b1, double* W2, double* b2){
    int row, col;
    printf("Now print init value of W1, b1, W2, b2...\n");
    // Print W1
    printf("**********W1**********\n");
    for (row = 0; row < layer[0]; row++){
        for (col = 0; col < layer[1]; col++){
            printf("%f\t", *(W1 + row * layer[0] + col));
        }
        printf("\n");
    }
    printf("**********************\n");
    // Print b1
    printf("\n**********b1**********\n");
    for (col = 0; col < layer[1]; col++)
        printf("%f\t", *(b1 + col));
    printf("\n**********************\n");
    // Print W2
    printf("\n**********W2**********\n");
    for (row = 0; row < layer[1]; row++){
        for (col = 0; col < layer[2]; col++){
            printf("%f\t", *(W2 + row * layer[2] + col));
        }
        printf("\n");
    }
    printf("**********************\n");
    // Print b2
    printf("\n**********b2**********\n");
    for (col = 0; col < layer[2]; col++){
        printf("%f\t", *(b2 + col));
    }
    printf("\n*********************\n");
}

double gaussrand()
{
  double x = (double)random() / RAND_MAX,
         y = (double)random() / RAND_MAX,
         z = sqrt(-2 * log(x)) * cos(2 * M_PI * y);
  return z;
}

void * dot_matrix_multiply(){


}
