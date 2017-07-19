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
    int row, col, i, j, iter;

    /* Load data */
    double* X_train = (double *)malloc(TRAIN_NUM * layer[0] * sizeof(double));
    int*    y_train = (int *)malloc(TRAIN_NUM * CLASS_NUM * sizeof(int));
    double* X_valid = (double *)malloc(VALIDATION_NUM * layer[0] * sizeof(double));
    int*    y_valid = (int *)malloc(VALIDATION_NUM * CLASS_NUM * sizeof(int));
    double* X_test  = (double *)malloc(TEST_NUM * layer[0] * sizeof(double));
    int*    y_test  = (int *)malloc(TEST_NUM * CLASS_NUM * sizeof(int));
    read_data(X_train, y_train, X_valid, y_valid, X_test, y_test);

    /* Init model */
    double* W1      = (double *)malloc(layer[0] * layer[1] * sizeof(double));
    double* b1      = (double *)malloc(layer[1] * sizeof(double));
    double* W2      = (double *)malloc(layer[1] * layer[2] * sizeof(double));
    double* b2      = (double *)malloc(layer[2] * sizeof(double));
    init_model_params(W1, b1, W2, b2);

    



























    return 0;
}
