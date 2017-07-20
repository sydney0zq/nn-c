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
extern int layer[3];
/* The input is 28*28 image, I need to flat it to 784
 */
int layer[3] = {784, 100, 10};

int main(char argc, char **argv){
    int row, col, i, j, iter;
    //srand((unsigned)time(NULL));

    /* Load data */
    double* X_train = (double *)malloc(TRAIN_NUM * layer[0] * sizeof(double));
    int*    y_train = (int *)malloc(TRAIN_NUM * CLASS_NUM * sizeof(int));
    double* X_valid = (double *)malloc(VALIDATION_NUM * layer[0] * sizeof(double));
    int*    y_valid = (int *)malloc(VALIDATION_NUM * CLASS_NUM * sizeof(int));
    double* X_test  = (double *)malloc(TEST_NUM * layer[0] * sizeof(double));
    int*    y_test  = (int *)malloc(TEST_NUM * CLASS_NUM * sizeof(int));

    double* X_batch = (double *)malloc(BATCH_SIZE * layer[0] * sizeof(double));
    int*    y_batch = (int *)malloc(BATCH_SIZE * CLASS_NUM * sizeof(int));
#if DEBUG_MAIN_PROCESS
        printf("\nStart read data....\n");
#endif
    read_data(X_train, y_train, X_valid, y_valid, X_test, y_test);

    /* Init model */
    double* W1      = (double *)malloc(layer[0] * layer[1] * sizeof(double));
    double* b1      = (double *)malloc(layer[1] * sizeof(double));
    double* W2      = (double *)malloc(layer[1] * layer[2] * sizeof(double));
    double* b2      = (double *)malloc(layer[2] * sizeof(double));
#if DEBUG_MAIN_PROCESS
        printf("\nStart init model params....\n");
#endif
    init_model_params(W1, b1, W2, b2);
    

/********************START MACHINE LEARNING********************/
    for (iter = 0; iter < EPOCHES; iter++){
        gen_batch(X_train, y_train, X_batch, y_batch);
        double* z1; double* a1; double* probs; double* z2; double* exp_scores;


#if DEBUG_MAIN_PROCESS
        printf("\nStart forward propagation....\n");
#endif
        /* Forward propagation */
        // z1 = X_batch.dot(W1) + b1 and a1 = np.tanh(z1)
        // z1 size: BATCH_SIZE * layer[1]; same with a1
        z1 = matrix_multi(X_batch, W1, BATCH_SIZE, layer[0], layer[0], layer[1]);  // TODO free -
        matrix_add_vector(z1, b1, BATCH_SIZE, layer[1]);
        matrix_single_op(z1, BATCH_SIZE, layer[1], "tanh");
        a1 = z1;
        
        // z2 = a1.dot(W2) + b2 and exp_scores = np.exp(z2)
        // z2 size: BATCH_SIZE * layer[2]; same with exp_scores, probs, delta3
        z2 = matrix_multi(a1, W2, BATCH_SIZE, layer[1], layer[1], layer[2]); // TODO free -
        matrix_add_vector(z2, b2, BATCH_SIZE, layer[2]);
        matrix_single_op(z2, BATCH_SIZE, layer[2], "exp");
        exp_scores = z2;

        // probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
        // Caculate the probability
        softmax(exp_scores, BATCH_SIZE, layer[2]);
        probs = exp_scores;

#if DEBUG_MAIN_PROCESS
        printf("\nStart backward propagation....\n");
#endif
        /* Backpropagation */
        double* delta3; double* dW2; double* dW1; double* left; double* right;
        double* db2;    double* db1; double* delta2;
        // delta3 = probs
        // softmax back gradient
        // http://okye062gb.bkt.clouddn.com/2017-07-19-142324.jpg   Softmax BP Algorithm
        delta3 = probs;
        for (i = 0; i < BATCH_SIZE; i++){
            int hit;
            for (j = 0; j < CLASS_NUM; j++){
                if (*(y_batch + i*CLASS_NUM + j) == 1){
                    *(delta3 + i*layer[2] + j) -= 1;
                    break;  // Break can only break out the most nearby for
                }
            }
        }
#if DEBUG_MAIN_PROCESS
        printf("\nSoftmax back gradient....\n");
#endif

        // dW2 = (a1.T).dot(delta3)
        a1 = transpose(a1, BATCH_SIZE, layer[1]);
        dW2 = matrix_multi(a1, delta3, layer[1], BATCH_SIZE, BATCH_SIZE, layer[2]);   //TODO free
#if DEBUG_MAIN_PROCESS
        printf("dW2 = (a1.T).dot(delta3)....\n");
#endif

        // db2 = np.sum(delta3, axis=0, keepdims=True)
        db2 = matrix_sum(delta3, BATCH_SIZE, layer[2]);  //TODO free
#if DEBUG_MAIN_PROCESS
        printf("db2 = np.sum(delta3, axis=0, keepdims=True)....\n");
#endif

        // delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        W2 = transpose(W2, layer[1], layer[2]);
        left = matrix_multi(delta3, W2, BATCH_SIZE, layer[2], layer[2], layer[1]); //TODO free -
        a1 = transpose(a1, layer[1], BATCH_SIZE);
        matrix_single_op(a1, BATCH_SIZE, layer[1], "pow2");
        matrix_single_const(a1, -1, BATCH_SIZE, layer[1], "multi");
        matrix_single_const(a1, 1, BATCH_SIZE, layer[1], "add");
        right = a1;
        delta2 = elemwise_multi(left, right, BATCH_SIZE, layer[1], BATCH_SIZE, layer[1]); //TODO free
        free(delta3); free(right);    //free z2(delta3/probs), free z1(a1)
        free(left);   //free left
#if DEBUG_MAIN_PROCESS
        printf("delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))....\n");
#endif

        // dW1 = np.dot(X.T, delta2)
        X_batch = transpose(X_batch, BATCH_SIZE, layer[0]); 
        dW1 = matrix_multi(X_batch, delta2, layer[0], BATCH_SIZE, BATCH_SIZE, layer[1]);    //TODO free
#if DEBUG_MAIN_PROCESS
        printf("dW1 = np.dot(X.T, delta2)....\n");
#endif
        // db1 = np.sum(delta2, axis=0)
        db1 = matrix_sum(delta2, BATCH_SIZE, layer[1]);  //TODO free
#if DEBUG_MAIN_PROCESS
        printf("db1 = np.sum(delta2, axis=0)....\n");
#endif

        // Add regularization terms (b1 and b2 don't have regularization terms)
        double* reg_matrix_W2  = (double *)malloc(layer[1] * layer[2] * sizeof(double)); //TODO free -
        // dW2 += REGULARIATION_LAMBDA * W2
        // dW1 += REGULARIATION_LAMBDA * W1
        W2 = transpose(W2, layer[2], layer[1]);
        matrix_copy(reg_matrix_W2, W2, layer[1], layer[2]);
        matrix_single_const(reg_matrix_W2, REGULARIATION_LAMBDA, layer[1], layer[2], "multi");
        matrix_add(dW2, reg_matrix_W2, layer[1], layer[2]);
        free(reg_matrix_W2); reg_matrix_W2 = NULL;

        double* reg_matrix_W1  = (double *)malloc(layer[0] * layer[1] * sizeof(double)); //TODO free -
        matrix_copy(reg_matrix_W1, W1, layer[0], layer[1]);
        matrix_single_const(reg_matrix_W1, REGULARIATION_LAMBDA, layer[0], layer[1], "multi");
        matrix_add(dW1, reg_matrix_W1, layer[0], layer[1]);
        free(reg_matrix_W1); reg_matrix_W1 = NULL;
#if DEBUG_MAIN_PROCESS
        printf("Add regularization terms....\n");
#endif

#if DEBUG_MAIN_PROCESS
        printf("Start SGD....\n");
#endif
        /* Gradient descent parameter update */
        // W1 += -LEARNING_RATE * dW1
        // b1 += -LEARNING_RATE * db1
        // W2 += -LEARNING_RATE * dW2
        // b2 += -LEARNING_RATE * db2
        matrix_single_const(dW1, (-1) * LEARNING_RATE, layer[0], layer[1], "multi");
        matrix_add(W1, dW1, layer[0], layer[1]);
#if DEBUG_MAIN_PROCESS
        printf("W1 += -LEARNING_RATE * dW1....\n");
#endif
        matrix_single_const(db1, (-1) * LEARNING_RATE, 1, layer[1], "multi");
        matrix_add(b1, db1, 1, layer[1]);
#if DEBUG_MAIN_PROCESS
        printf("b1 += -LEARNING_RATE * db1....\n");
#endif
        matrix_single_const(dW2, (-1) * LEARNING_RATE, layer[1], layer[2], "multi");
        matrix_add(W2, dW2, layer[1], layer[2]);
#if DEBUG_MAIN_PROCESS
        printf("W2 += -LEARNING_RATE * dW2....\n");
#endif
        matrix_single_const(db2, (-1) * LEARNING_RATE, 1, layer[2], "multi");
        matrix_add(b2, db2, 1, layer[2]);
        
        printf("Loss after iteration %d %lf\n", 
                iter, calculate_loss(X_batch, W1, b1, W2, b2));
    }


    return 0;
}
