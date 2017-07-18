/*
 * main.c
 * Distributed under terms of the GPLv3 license.
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
/* The input is 28*28 image, I need to flat it to 784 and normalize it 
 */
int layer[3] = {784, 100, 10};

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
    double* probs = (double *)malloc(TRAIN_NUM*layer[2]*sizeof(double));
    double* delta3 = (double *)malloc(TRAIN_NUM*layer[2]*sizeof(double));

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
        if (DEBUG_MAIN){ printf("I am on inter%d\n", iter); }
        // Forward propagation
	    // z1 = X.dot(W1) + b1 and a1 = np.tanh(z1)
        z1 = matrix_multi(X, W1, TRAIN_NUM, layer[0], layer[0], layer[1]);  // TODO free
        matrix_add_vector(z1, b1, TRAIN_NUM, layer[1]);
        matrix_single_op(z1, TRAIN_NUM, layer[1], "tanh");
        a1 = z1;
#if (DEBUG_MAIN)
printf("\nPrint z1 = X.dot(W1) + b1 and a1 = np.tanh(z1) ==> a1\n");
print_matrix(a1, TRAIN_NUM, layer[1]);
#endif     
        // z2 = a1.dot(W2) + b2 and exp_scores = np.exp(z2)
        z2 = matrix_multi(a1, W2, TRAIN_NUM, layer[1], layer[1], layer[2]); // TODO free
        matrix_add_vector(z2, b2, TRAIN_NUM, layer[2]);
        matrix_single_op(z2, TRAIN_NUM, layer[2], "exp");
        exp_scores = z2;
#if (DEBUG_MAIN)
printf("\nPrint z2 = a1.dot(W2) + b2 and exp_scores = np.exp(z2) ==> exp_scores\n");
print_matrix(exp_scores, TRAIN_NUM, layer[2]);
#endif     
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
#if (DEBUG_MAIN)
printf("\nPrint probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) ==> probs\n");
print_matrix(probs, TRAIN_NUM, layer[2]);
#endif     


        /* Backpropagation */
        // delta3 = probs
        for (row = 0; row < TRAIN_NUM; row++){
            for (col = 0; col < layer[2]; col++){
                *(delta3 + row*layer[2] + col) = *(probs + row*layer[2] + col);
            }
        }
        
        double* dW2; double* db2; double* dW1; double* db1; double* delta2; double* left; double* right;

        // delta3[range(num_examples), y] -= 1
        for (row = 0; row < TRAIN_NUM; row++){
            *(delta3 + row*layer[2] + (ptr_train_data+row)->label) -= 1;
        }
        // dW2 = (a1.T).dot(delta3)
        a1 = transpose(a1, TRAIN_NUM, layer[1]);
        dW2 = matrix_multi(a1, delta3, layer[1], TRAIN_NUM, TRAIN_NUM, layer[2]);   //TODO free
#if (DEBUG_MAIN)
printf("\nPrint dW2 = (a1.T).dot(delta3) ==> dW2\n");
print_matrix(dW2, layer[1], layer[2]);
#endif     
        // db2 = np.sum(delta3, axis=0, keepdims=True)
        db2 = matrix_sum(delta3, TRAIN_NUM, layer[2]);  //TODO free

#if (DEBUG_MAIN)
printf("\nPrint delta3[range(num_examples), y] -= 1 ==> delta3\n");
print_matrix(delta3, TRAIN_NUM, layer[2]);
#endif     

#if (DEBUG_MAIN)
printf("\nPrint db2 = np.sum(delta3, axis=0, keepdims=True) ==> db2\n");
print_matrix(db2, 1, layer[2]);
#endif     

        // delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        W2 = transpose(W2, layer[1], layer[2]);
        left = matrix_multi(delta3, W2, TRAIN_NUM, layer[2], layer[2], layer[1]);
#if (DEBUG_MAIN)
printf("\nPrint delta3.dot(W2.T) ==> left\n");
print_matrix(left, TRAIN_NUM, layer[1]);
#endif     

        a1 = transpose(a1, layer[1], TRAIN_NUM);
        matrix_single_op(a1, TRAIN_NUM, layer[1], "pow2");
        double* reg0;
        double* reg1;
        a1 = matrix_single_const(a1, -1, TRAIN_NUM, layer[1], "multi");
        //free(a1);
        right = matrix_single_const(a1, 1, TRAIN_NUM, layer[1], "add");
        //free(reg0);
#if (DEBUG_MAIN)
printf("\nPrint (1 - np.power(a1, 2)) ==> right");
print_matrix(right, TRAIN_NUM, layer[1]);
#endif     

        delta2 = elemwise_multi(left, right, TRAIN_NUM, layer[1], TRAIN_NUM, layer[1]);
#if (DEBUG_MAIN)
printf("\nPrint delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2)) ==> delta2");
print_matrix(delta2, TRAIN_NUM, layer[1]);
#endif     

        // dW1 = np.dot(X.T, delta2)
        X = transpose(X, TRAIN_NUM, layer[0]); 
        dW1 = matrix_multi(X, delta2, layer[0], TRAIN_NUM, TRAIN_NUM, layer[1]);
#if (DEBUG_MAIN)
printf("\nPrint dW1 = np.dot(X.T, delta2) ==> dW1");
print_matrix(dW1, layer[0], layer[1]);
#endif     
        // db1 = np.sum(delta2, axis=0)
        db1 = matrix_sum(delta2, TRAIN_NUM, layer[1]);
        X = transpose(X, layer[0], TRAIN_NUM); 
#if (DEBUG_MAIN)
printf("\nPrint db1 = np.sum(delta2, axis=0) ==> db1");
print_matrix(db1, 1, layer[1]);
#endif     

        // Add regularization terms (b1 and b2 don't have regularization terms)
        double * reg_matrix;
        // dW2 += reg_lambda * W2
        // dW1 += reg_lambda * W1
        W2 = transpose(W2, layer[2], layer[1]);
        reg_matrix = matrix_single_const(W2, REGULARIATION_LAMBDA, layer[1], layer[2], "multi");
        matrix_add(dW2, reg_matrix, layer[1], layer[2]);
        free(reg_matrix);
#if (DEBUG_MAIN)
printf("\nPrint dW2 += reg_lambda * W2 ==> dW2");
print_matrix(dW2, layer[1], layer[2]);
#endif     

        reg_matrix = matrix_single_const(W1, REGULARIATION_LAMBDA, layer[0], layer[1], "multi");
        matrix_add(dW1, reg_matrix, layer[0], layer[1]);
        free(reg_matrix);
#if (DEBUG_MAIN)
printf("\nPrint dW1 += reg_lambda * W1 ==> dW1");
print_matrix(dW1, layer[0], layer[1]);
#endif     

        // Gradient descent parameter update
        // W1 += -epsilon * dW1
        // b1 += -epsilon * db1
        // W2 += -epsilon * dW2
        // b2 += -epsilon * db2
        reg_matrix = matrix_single_const(dW1, (-1) * LEARNING_RATE, layer[0], layer[1], "multi");
        matrix_add(W1, reg_matrix, layer[0], layer[1]);
        free(reg_matrix);
#if (DEBUG_MAIN)
printf("\nPrint W1 += -epsilon * dW1 ==> W1");
print_matrix(W1, layer[0], layer[1]);
#endif  

        reg_matrix = matrix_single_const(db1, (-1) * LEARNING_RATE, 1, layer[1], "multi");
        matrix_add(b1, reg_matrix, 1, layer[1]);
        free(reg_matrix);
#if (DEBUG_MAIN)
printf("\nPrint b1 += -epsilon * db1 ==> b1\n");
print_matrix(b1, 1, layer[1]);
#endif     
        
        reg_matrix = matrix_single_const(dW2, (-1) * LEARNING_RATE, layer[1], layer[2], "multi");
        matrix_add(W2, reg_matrix, layer[1], layer[2]);
        free(reg_matrix);
#if (DEBUG_MAIN)
printf("\nPrint W2 += -epsilon * dW2 ==> W2\n");
print_matrix(W2, layer[1], layer[2]);
#endif     

        reg_matrix = matrix_single_const(db2, (-1) * LEARNING_RATE, 1, layer[2], "multi");
        matrix_add(b2, reg_matrix, 1, layer[2]);
        free(reg_matrix);
#if (DEBUG_MAIN)
printf("\nPrint b2 += -epsilon * db2 ==> b2\n");
print_matrix(b2, 1, layer[2]);
#endif     

    if (iter % 10 == 0)
        printf("Loss after iteration %d: %lf\n", 
                iter, calculate_loss(ptr_train_data, X, W1, b1, W2, b2));
    }// End of gradient descent

    
    
    //free(W1, b1, W2, b2);

    return 0;
}
