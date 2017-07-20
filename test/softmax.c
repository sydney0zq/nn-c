#include <stdio.h>

int main(){
    int row = 2, col = 2;
    double* m = (double*) malloc(row*col*sizeof(double));
    
    int i,j;
    for (i = 0; i < row; i++){
        for (j = 0; j < col; j++){
            scanf("%lf", m+i*col+j);
        }
    }
    print_matrix(m, row, col);
    
    softmax(m, row, col);
    print_matrix(m, row, col);
}



void print_matrix(double* m, int row, int col){
    int i, j;
    printf("\n**********\n");
    for (i = 0; i < row; i++){
        for (j = 0; j < col; j++){
            printf("%f\t", *(m+i*col+j));
        }
        printf("\n");
    }
    printf("**********\n");
}



void softmax(double* matrix, int row, int col){
    double tmp;
    int i, j;
    for (i = 0; i < row; i++){
        tmp = 0;
        // Sum the whole row value
        for (j = 0; j < col; j++)
            tmp += *(matrix + i*col + j);
        for (j = 0; j < col; j++)
            *(matrix + i*col + j) = *(matrix + i*col + j) / tmp;
    }
}
