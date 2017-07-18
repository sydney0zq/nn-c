// Copyright © 2017-07-18 Sydney <theodoruszq@gmail.com>

#include "header.h"


#define DEBUG_MATRIX_MULTI 1
double* matrix_multi(double* Ma, double* Mb, int row_a, int col_a, int row_b, int col_b);

// You should NOTICE FREE matrix
double* matrix_multi(double* Ma, double* Mb, int row_a, int col_a, int row_b, int col_b){
    if (col_a != row_b){
        printf("Error when multiply matrix...");
        exit(1);
    }

    int i, j, k;

    double* matrix_res = (double *)malloc(row_a * col_b * sizeof(double));
    double tmp;
    for (i = 0; i < row_a; i++){
        for (j = 0; j < col_b; j++){
            tmp = 0;
            for (k = 0; k < col_a; k++){
                tmp += ((*(Ma + i*col_a + k)) * (*(Mb + k*col_b + j)));
            }
            *(matrix_res + i*col_b + j) = tmp;
        }
    }

    if (DEBUG_MATRIX_MULTI){
        printf("\n*****Print Ma*****\n");
        for (i = 0; i < row_a; i++){
            for (j = 0; j < col_a; j++)
                printf("%lf\t", *(Ma + i*col_a + j));
            printf("\n");
        }
        printf("\n*****Print Mb*****\n");
        for (i = 0; i < row_b; i++){
            for (j = 0; j < col_b; j++)
                printf("%lf\t", *(Mb + i*col_b + j));
            printf("\n");
        }
        printf("\n*****Print Result******\n");
        for (i = 0; i < row_a; i++){
            for (j = 0; j < col_b; j++)
                printf("%lf\t", *(matrix_res + i*col_b + j));
            printf("\n");
        }
    }
    return matrix_res;
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


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

