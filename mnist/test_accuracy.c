#include "header.h"

#define DEBUG_ACCURACY 0

int main(char argc, char** argv){
    
    double* W1      = (double *)malloc(layer[0] * layer[1] * sizeof(double));
    double* b1      = (double *)malloc(layer[1] * sizeof(double));
    double* W2      = (double *)malloc(layer[1] * layer[2] * sizeof(double));
    double* b2      = (double *)malloc(layer[2] * sizeof(double));
    double* X_test  = (double *)malloc(TEST_NUM * layer[0] * sizeof(double));
    int*    y_test  = (int *)malloc(TEST_NUM * CLASS_NUM * sizeof(int));
    
    double accuracy;

#if DEBUG_ACCURACY
printf("start read model...\n");
printf("layer[0]: %d, layer[1]: %d, layer[2]: %d\n", layer[0], layer[1], layer[2]);
#endif
    read_model(W1, b1, W2, b2);
#if DEBUG_ACCURACY
printf("read model over...\n");
#endif
    read_test_data(X_test, y_test);
    accuracy = test_accuracy(X_test, y_test, W1, b1, W2, b2);
#if DEBUG_ACCURACY
printf("TEST accuracy over...\n");
#endif
    printf("Accuracy: %lf\n", accuracy);

    return 0;
}
