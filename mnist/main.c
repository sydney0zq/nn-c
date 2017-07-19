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
    double* z1;
    double* a1;
    double* z2;
    double* exp_scores;
    double* probs = (double *)malloc(TRAIN_NUM*layer[2]*sizeof(double));
    double* delta3 = (double *)malloc(TRAIN_NUM*layer[2]*sizeof(double));
    // Gradient descent for each batch
    for (iter = 0; iter < ITER_TIMES; iter++){

        // delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        W2 = transpose(W2, layer[1], layer[2]);
        left = matrix_multi(delta3, W2, TRAIN_NUM, layer[2], layer[2], layer[1]);

        a1 = transpose(a1, layer[1], TRAIN_NUM);
        matrix_single_op(a1, TRAIN_NUM, layer[1], "pow2");
        double* reg0;
        double* reg1;
        a1 = matrix_single_const(a1, -1, TRAIN_NUM, layer[1], "multi");
        //free(a1);
        right = matrix_single_const(a1, 1, TRAIN_NUM, layer[1], "add");
        //free(reg0);

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

    if (iter % 10 == 0)
        printf("Loss after iteration %d: %lf\n", 
                iter, calculate_loss(ptr_train_data, X, W1, b1, W2, b2));
    }// End of gradient descent

    
    
    //free(W1, b1, W2, b2);

    return 0;
}
