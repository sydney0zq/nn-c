#include <stdio.h>
#include <stdlib.h>

#define DEBUG_MATRIX_SUM 1

double* matrix_sum(double* matrix, int row, int col);

int main(char argc, char ** argv){
    int row = 2, col = 2;
    int i, j;
    double num = 5;
    double* Ma = (double *)malloc(row * col * sizeof(double));
    //init Ma
    printf("Now input Ma..\n");
    for (i = 0; i < row; i++){
        for (j = 0; j < col; j++)
            scanf("%lf", Ma+i*col+j);
    }
    matrix_sum(Ma, row, col);
    return 0;
}

double* matrix_sum(double* matrix, int row, int col){
    int i, j;
    double* vec_res = (double *)malloc(col*sizeof(double));

    for (i = 0; i < col; i++){
        *(vec_res + i) = 0;
        for (j = 0; j < row; j++){
            *(vec_res + i) += *(matrix + i + j*col);
        }
    }

    if (DEBUG_MATRIX_SUM){
        printf("\n*****Print Matrix******\n");
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++)
                printf("%lf\t", *(matrix + i*col + j));
            printf("\n");
        }
        printf("\n*****Print Result******\n");
        for (j = 0; j < col; j++)
            printf("%lf\t", *(vec_res + j));
    }

    return vec_res;
}

