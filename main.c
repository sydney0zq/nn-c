/*
 * main.c
 * Copyright (C) 2017 zq <zq@solitude>
 *
 * Distributed under terms of the MIT license.
 */

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

extern int layer[3];
int layer[3] = {2, 3, 2};

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
double gaussrand();

void debug_print_init_value(double* W1, double* b1, double* W2, double* b2);

int main(char argc, char **argv){
    struct data_box train_data[TRAIN_NUM];
    struct data_box* ptr_train_data;
    int row, col, i, iter;

    ptr_train_data = train_data;
    read_data("./dataset.txt", ptr_train_data);

    // Build model: Learns paramters for the NN and returns the model
    double* W1 = (double *)malloc(layer[0]*layer[1]*sizeof(double));
    double* b1 = (double *)malloc(layer[1]*sizeof(double));
    double* W2 = (double *)malloc(layer[1]*layer[2]*sizeof(double));
    double* b2 = (double *)malloc(layer[2]*sizeof(double));

    double* z1 = (double *)malloc(TRAIN_NUM * layer[1] * sizeof(double));
    double* a1 = (double *)malloc(TRAIN_NUM * layer[1] * sizeof(double));
    double* z2 = (double *)malloc(TRAIN_NUM * layer[2] * sizeof(double));
    double* exp_scores = (double *)malloc(TRAIN_NUM * layer[2] * sizeof(double));
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
        for (row = 0; row < TRAIN_NUM; row++){
            for (col = 0; col < layer[1]; col++){
                // Not portable code
                *(z1 + row*layer[1] + col) = 
                    ((ptr_train_data+row)->xc) * (*(W1 + col)) + 
                    ((ptr_train_data+row)->yc) * (*(W1 + layer[1] + col));
                *(z1 + row*layer[1] + col) += b1[col];
                *(a1 + row*layer[1] + col) = tanh(*(z1 + row*layer[1] + col));
            }
        }
        
        // z2 = a1.dot(W2) + b2 and exp_scores = np.exp(z2)
        double tmp;
        for (row = 0; row < TRAIN_NUM; row++){
            for (col = 0; col < layer[2]; col++){
                tmp = 0;
                for (i = 0; i < layer[1]; i++){
                    tmp += (*(a1 + row*layer[1] + i)) * (*(W2 + i*layer[1] + col));
                }
                *(z2 + row*layer[2] + col) += b2[col];
                *(exp_scores + row*layer[2] + col) = exp(*(z2 + row*layer[2] + col));
            }
        }        
        
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
        
        double dW2, db2, dW1, db1, W1, b1, W2, b2;
        // delta3[range(num_examples), y] -= 1
        for (row = 0; row < TRAIN_NUM; row++){
            *(delta3 + row*layer[2] + *(ptr_train_data+row*3+2)) -= 1;

        }









    }// End of gradient descent

    
    
    //free(W1, b1, W2, b2);

    return 0;
}



/*
 * Read data from a file which is generated from python script
 * gen_train_data.py
 */
void read_data(char* path, struct data_box* ptr_train_data){
    FILE* fp;
    struct data_box* base_ptr = ptr_train_data;
    int i;

    if ((fp = fopen(path, "r")) != NULL){
        for (i = 0; i < TRAIN_NUM; i++, ptr_train_data++){
            fscanf(fp, "%lf %lf %d", 
                &(ptr_train_data->xc), &(ptr_train_data->yc), &(ptr_train_data->label));
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

void transpose(int row, int col, double* matrix){
    int i, j;
    double* matrix_res = (double *)malloc(col * row * sizeof(double));
    int row_res = col;
    int col_res = row;
    // Displaying the matrix[][] 
    if (DEBUG_TRANS_DISPLAY){
        printf("\n*****Matrix To Be Transposed*****\n");
        for(i = 0; i < row; ++i)
            for(j = 0; j < col; ++j){
                printf("%f  ", *(matrix + i*col + j));
                if (j == col-1)
                    printf("\n");
            }
    }
    
    // Finding the transpose of matrix
    for(i = 0; i < row; ++i)
        for(j = 0; j < col; ++j)
            *(matrix_res + j*col_res + i) = *(matrix + i*col + j);
    
    // Displaying the transpose of matrix
    if (DEBUG_TRANS_DISPLAY){
        printf("\n*****Transposed Matrix*****\n");
        for(i = 0; i < row_res; ++i)
            for(j = 0; j < col_res; ++j){
                printf("%f  ", *(matrix_res + i*col_res + j));
                if(j == col_res-1)
                    printf("\n");
            }
    }
    free(matrix);
    matrix = matrix_res;
}


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

