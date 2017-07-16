/*
 * main.c
 * Copyright (C) 2017 zq <zq@solitude>
 *
 * Distributed under terms of the MIT license.
 */

#include <stdio.h>
#include <stdlib.h>

struct data_box{
    double xc;      //The coordinate x and y, must be double type
    double yc;
    int label;
};

#define TRAIN_NUM 200
#define LEARNING_RATE 0.01  //Learning rate for gradient descent
#define REGULARIATION_LAMBDA 0.01   //Regularization strength
extern int layer[3] = [2, 3, 2];





void read_data(char* path, struct data_box* ptr_train_data);


int main(char argc, char **argv){
    struct data_box train_data[TRAIN_NUM];
    struct data_box* ptr_train_data;

    ptr_train_data = train_data;
    read_data("./dataset.txt", ptr_train_data);

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
    for (i = 0; i < TRAIN_NUM; i++, ptr_train_data++){
        printf("%f, %f, %d\n", ptr_train_data->xc, ptr_train_data->yc, ptr_train_data->label);   
    }
}

