
#include <stdio.h>
#include <stdlib.h>

int main(char argc, char ** argv){

    double tmp, loss;
    int i,j, row = 2, col = 2;
    double* m = (double *)malloc(row*col*sizeof(double));

    for (i = 0; i < row; i++){
        for (j = 0; j < col; j++)
            scanf("%lf", m + i*col + j);
    }

    print_matrix(m, row, col);
    for (i = 0; i < row; i++){
        tmp = 0;
        for (j = 0; j < col; j++){
            tmp += *(m + i*col + j);
        }
        for (j = 0; j < col; j++){
            *(m + i*col + j) = *(m + i*col + j) / tmp;
        }
    }
    loss = 0;
    for (i = 0; i < row; i++){
        loss += ( -log(*(probs+i*layer[2])) );
    }
    printf("\n********************\n");
    print_matrix(m, row, col);
    

    return 0;
}


void print_matrix(double* m, int row, int col){
    int i, j;
    for (i = 0; i < row; i++){
        for (j = 0; j < col; j++){
            printf("%f\t", *(m+i*col+j));
        }
        printf("\n");
    }
}
