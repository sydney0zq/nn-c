#include <stdio.h>
#include <stdlib.h>

#define DEBUG_MATRIX_ADD 1

void matrix_add(double* Ma, double* Mb, int row, int col);
int main(char argc, char ** argv){
    int row = 2, col = 2;
    int i, j;
    double* Ma = (double *)malloc(row * col * sizeof(double));
    double* Mb = (double *)malloc(row * col * sizeof(double));
    //init Ma
    printf("Now input Ma..\n");
    for (i = 0; i < row; i++){
        for (j = 0; j < col; j++)
            scanf("%lf", Ma+i*col+j);
    }
    printf("Now input Mb..\n");
    for (i = 0; i < row; i++){
        for (j = 0; j < col; j++)
            scanf("%lf", Mb+i*col+j);
    }
    matrix_add(Ma, Mb, row, col);
    return 0;
}

void matrix_add(double* Ma, double* Mb, int row, int col){
    int i, j;
    for (i = 0; i < row; i++){
        for (j = 0; j < col; j++){
            *(Ma + i*col + j) += *(Mb + i*col + j);
        }
    }

    if (DEBUG_MATRIX_ADD){
        printf("\n*****Print Result******\n");
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++)
                printf("%lf\t", *(Ma + i*col + j));
            printf("\n");
        }
    }
}
