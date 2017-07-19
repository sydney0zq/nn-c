#include "header.h"
extern int layer[3];
void read_data(double* X_train, int* y_train, double* X_valid, int* y_valid, double* X_test, int* y_test){
    FILE* fp;
    int i, j;

    // Read training data
    if ((fp = fopen("./data/dataset_train.txt", "r")) != NULL){
        for (i = 0; i < TRAIN_NUM; i++)
            for (j = 0; j < layer[0]; j++)
                fscanf(fp, "%lf", X_train + i*layer[0] + j);
        fclose(fp);
    }else{
        printf("Read File Error, Now exiting...");
        exit(1);
    }
    if ((fp = fopen("./data/dataset_train_label.txt", "r")) != NULL){
        for (i = 0; i < TRAIN_NUM; i++)
            for (j = 0; j < CLASS_NUM; j++)
                fscanf(fp, "%d", y_train + i*CLASS_NUM + j);
        fclose(fp);
    }else{
        printf("Read File Error, Now exiting...");
        exit(1);
    }

    // Read validation data
    if ((fp = fopen("./data/dataset_vali.txt", "r")) != NULL){
        for (i = 0; i < VALIDATION_NUM; i++)
            for (j = 0; j < layer[0]; j++)
                fscanf(fp, "%lf", X_valid + i*layer[0] + j);
        fclose(fp);
    }else{
        printf("Read File Error, Now exiting...");
        exit(1);
    }
    if ((fp = fopen("./data/dataset_vali_label.txt", "r")) != NULL){
        for (i = 0; i < VALIDATION_NUM; i++)
            for (j = 0; j < layer[0]; j++)
                fscanf(fp, "%d", X_valid + i*layer[0] + j);
        fclose(fp);
    }else{
        printf("Read File Error, Now exiting...");
        exit(1);
    }

    // Read test data
    if ((fp = fopen("./data/dataset_test.txt", "r")) != NULL){
        for (i = 0; i < TEST_NUM; i++)
            for (j = 0; j < layer[0]; j++)
                fscanf(fp, "%lf", X_test + i*layer[0] + j);
        fclose(fp);
    }else{
        printf("Read File Error, Now exiting...");
        exit(1);
    }
    if ((fp = fopen("./data/dataset_test_label.txt", "r")) != NULL){
        for (i = 0; i < TEST_NUM; i++)
            for (j = 0; j < CLASS_NUM; j++)
                fscanf(fp, "%d", X_test + i*CLASS_NUM + j);
        fclose(fp);
    }else{
        printf("Read File Error, Now exiting...");
        exit(1);
    }
    
#if DEBUG_PRINT_DATASET_VALUE
    // Print the first line
    printf("****Print first line of training data*****\n");
    for (j = 0; j < layer[0]; j++)
        printf("%f\t", *(X_train + j));
    printf("\n");
    for (j = 0; j < layer[0]; j++)
        printf("%d\t", *(y_train + j));
    printf("\n");
    
#endif
}
