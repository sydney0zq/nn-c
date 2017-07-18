#include <stdio.h>
#include <stdlib.h>

#define DEBUG_MATRIX_ADD_VECTOR 1

void matrix_add_vector(double* matrix, double* vec, int row, int col);

int main(char argc, char ** argv){
    int row = 2, col = 2;
    int i, j;
    double* Ma = (double *)malloc(row * col * sizeof(double));
    double* vec = (double *)malloc(col * sizeof(double));
    //init Ma
    printf("Now input Ma..\n");
    for (i = 0; i < row; i++){
        for (j = 0; j < col; j++)
            scanf("%lf", Ma+i*col+j);
    }
    printf("Now input vector..\n");
    for (i = 0; i < col; i++){
        scanf("%lf", vec + i);
    }
    matrix_add_vector(Ma, vec, row, col);
    return 0;
}

void matrix_add_vector(double* matrix, double* vec, int row, int col){
    int i, j;

    // Print origin matrix
    if (DEBUG_MATRIX_ADD_VECTOR){
        printf("\n*****Print Matrix******\n");
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++)
                printf("%lf\t", *(matrix + i*col + j));
            printf("\n");
        }
        printf("\n*****Print Vector******\n");
        for (i = 0; i < col; i++){
            printf("%lf\t", *(vec + i));
        }
    }

    for (i = 0; i < row; i++){
        for (j = 0; j < col; j++){
            *(matrix + i*col + j) += *(vec + j);
        }
    }

    if (DEBUG_MATRIX_ADD_VECTOR){
        printf("\n*****Print Result******\n");
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++)
                printf("%lf\t", *(matrix + i*col + j));
            printf("\n");
        }
    }
}
