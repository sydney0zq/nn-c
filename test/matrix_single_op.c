#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DEBUG_MATRIX_SINGLE_OP 1

void matrix_single_op(double* matrix, int row, int col, char* type);
int main(char argc, char ** argv){
    int row = 1, col = 2;
    int i, j;
    double num = 5;
    double* Ma = (double *)malloc(row * col * sizeof(double));
    //init Ma
    printf("Now input Ma..\n");
    for (i = 0; i < row; i++){
        for (j = 0; j < col; j++)
            scanf("%lf", Ma+i*col+j);
    }
    matrix_single_op(Ma, row, col, "exp");
    matrix_single_op(Ma, row, col, "tanh");
    return 0;
}

void matrix_single_op(double* matrix, int row, int col, char* type){
    int i, j;
    if (type == "exp"){
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++){
                *(matrix + i*col + j) = exp(*(matrix + i*col + j));
            }
        }
    }else if (type == "tanh"){
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++){
                *(matrix + i*col + j) = tanh(*(matrix + i*col + j));
            }
        }
    }

    if (DEBUG_MATRIX_SINGLE_OP){
        printf("\n*****Print Result******\n");
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++)
                printf("%lf\t", *(matrix + i*col + j));
            printf("\n");
        }
    }
}
