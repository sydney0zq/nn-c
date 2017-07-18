// Copyright © 2017-07-18 Sydney <theodoruszq@gmail.com>

#include <stdio.h>
#include <stdlib.h>

#define DEBUG_MATRIX_MULTI 1

double* matrix_multi(double* Ma, double* Mb, int row_a, int col_a, int row_b, int col_b);
int main(char argc, char ** argv){
    int row_a = 3, col_a = 2;
    int row_b = 2, col_b = 2;
    int i, j;
    double* Ma = (double *)malloc(row_a * col_a * sizeof(double));
    double* Mb = (double *)malloc(row_b * col_b * sizeof(double));
    double* res;
    //init Ma
    printf("Now input Ma..\n");
    for (i = 0; i < row_a; i++){
        for (j = 0; j < col_a; j++)
            scanf("%lf", Ma+i*col_a+j);
    }
    //init Mb
    printf("Now input Mb..\n");
    for (i = 0; i < row_b; i++){
        for (j = 0; j < col_b; j++)
            scanf("%lf", Mb+i*col_b+j);
    }
    res = matrix_multi(Ma, Mb, row_a, col_a, row_b, col_b);

    return 0;
    for (i = 0; i < row_a; i++){
        for (j = 0; j < col_b; j++)
            printf("%lf\t", *(res + i*col_b +j));
        printf("\n");
    }

}

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
