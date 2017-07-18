#include <stdio.h>
#include <stdlib.h>

#define DEBUG_MATRIX_ADD 1

void matrix_add(double* matrix, double num, int row, int col);
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
    matrix_add(Ma, num, row, col);
    return 0;
}

void matrix_add(double* matrix, double num, int row, int col){
    int i, j;
    for (i = 0; i < row; i++){
        for (j = 0; j < col; j++){
            *(matrix + i*col + j) += num;
        }
    }

    if (DEBUG_MATRIX_ADD){
        printf("\n*****Print Result******\n");
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++)
                printf("%lf\t", *(matrix + i*col + j));
            printf("\n");
        }
    }
}
