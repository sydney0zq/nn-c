#include <stdio.h>
#include <stdlib.h>

#define DEBUG_MATRIX_SINGLE_CONST 1

void matrix_single_const(double* matrix, double num, int row, int col, char* type);

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
    //matrix_single_const(Ma, 4, row, col, "add");
    matrix_single_const(Ma, 4, row, col, "multi");
    return 0;
}

void matrix_single_const(double* matrix, double num, int row, int col, char* type){
    int i, j;
    if (type == "add"){
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++){
                *(matrix + i*col + j) += num;
            }
        }
    }else if (type == "multi"){
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++){
                *(matrix + i*col + j) *= num;
            }
        }
    }

    if (DEBUG_MATRIX_SINGLE_CONST){
        printf("\n*****Print Result******\n");
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++){
                printf("%f\t", *(matrix + i*col + j));
            }
        }
    }

}

