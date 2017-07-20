#include "header.h"
extern int layer[3];

double calculate_loss(double* X_batch, int* y_batch, double* W1, double* b1, double* W2, double* b2){

    // Forward propagation to calculate our prediction
    double* z1;
    double* z2;
    double* a1;
    double* exp_scores; double* probs;
    
    int i, j;

    // z1 = X.dot(W1) + b1 and a1 = np.tanh(z1)
    z1 = matrix_multi(X_batch, W1, BATCH_SIZE, layer[0], layer[0], layer[1]);  // TODO free -
    matrix_add_vector(z1, b1, BATCH_SIZE, layer[1]);
    matrix_single_op(z1, BATCH_SIZE, layer[1], "tanh");
    a1 = z1;

    // z2 = a1.dot(W2) + b2 and exp_scores = np.exp(z2)
    z2 = matrix_multi(a1, W2, BATCH_SIZE, layer[1], layer[1], layer[2]); // TODO free -
    matrix_add_vector(z2, b2, BATCH_SIZE, layer[2]);
    matrix_single_op(z2, BATCH_SIZE, layer[2], "exp");
    exp_scores = z2;

    // Caculate the probability
    softmax(exp_scores, BATCH_SIZE, layer[2]);
    probs = exp_scores;

    // Calculating the loss
    double loss=0;
    for (i = 0; i < BATCH_SIZE; i++){
        for (j = 0; j < CLASS_NUM; j++){
            if (*(y_batch + i*CLASS_NUM + j) == 1){ 
                loss += ( -log(*(probs + i*CLASS_NUM + j)) );
                //printf("%d\t", *(y_batch+i*CLASS_NUM+j));
                //printf("%f\t", *(probs + i*CLASS_NUM + j));
                break;
            }
        }
    }
    free(probs); //free probs -> exp_scores -> z2
    free(a1);    //free a1 -> z1
    // Add regulatization term to loss (optional)
    // data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return (1.0*loss) / BATCH_SIZE;
}
