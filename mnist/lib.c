#include "header.h"

extern int layer[3];

void init_model_params(double* W1, double* b1, double* W2, double* b2){
    int row, col, i;
    // Initialize the paramters
    for (row = 0; row < layer[0]; row++)
        for (col = 0; col < layer[1]; col++)
            *(W1 + row*layer[1] + col) = gaussrand()/sqrt(layer[0]);
    for (i = 0; i < layer[1]; i++)
        *(b1 + i) = 0;
    for (row = 0; row < layer[1]; row++)
        for (col = 0; col < layer[2]; col++)
            *(W2 + row*layer[2] + col) = gaussrand()/sqrt(layer[1]);
    for (i = 0; i < layer[2]; i++)
        *(b2 + i) = 0;
}

void gen_batch(double* X_train, int* y_train, double* X_batch, int* y_batch){
    int i, j;
    // You should notice that RAND_MAX is 0x7FFFFFFF
    int batch_index[BATCH_SIZE];
    for (i = 0; i < BATCH_SIZE; i++)
       *(batch_index + i) = rand() % TRAIN_NUM;

    for (i = 0; i < BATCH_SIZE; i++){
        for (j = 0; j < layer[0]; j++){
            *(X_batch + i*layer[0] + j) = *(X_train + batch_index[i]*layer[0] + j);
        }
        for (j = 0; j < CLASS_NUM; j++){
            *(y_batch + i*CLASS_NUM + j) = *(y_train + batch_index[i]*CLASS_NUM + j);
        }
    }
    print_matrix(X_batch, BATCH_SIZE, layer[0]);
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

    return matrix_res;
}

void matrix_add_vector(double* matrix, double* vec, int row, int col){
    int i, j;

    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++)
            *(matrix + i*col + j) += *(vec + j);
}

void matrix_single_op(double* matrix, int row, int col, char* type){
    int i, j;
    if (!strcmp(type, "exp")){
        for (i = 0; i < row; i++)
            for (j = 0; j < col; j++)
                *(matrix + i*col + j) = exp(*(matrix + i*col + j));
    }else if (!strcmp(type, "tanh")){
        for (i = 0; i < row; i++)
            for (j = 0; j < col; j++)
                *(matrix + i*col + j) = tanh(*(matrix + i*col + j));
    }else if (!strcmp(type, "pow2")){
        for (i = 0; i < row; i++)
            for (j = 0; j < col; j++)
                *(matrix + i*col + j) = pow(*(matrix + i*col + j), 2);
    }else{
        printf("No such operation, update you lib.c...");
        exit(1);
    }
}

void matrix_single_const(double* matrix, double num, int row, int col, char* type){
    int i, j;
    if (!strcmp(type, "add")){
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++){
                *(matrix + i*col + j) = (*(matrix + i*col + j)) + num;
            }
        }
    }else if (!strcmp(type, "multi")){
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++){
                *(matrix + i*col + j) = (*(matrix + i*col + j)) * num;
            }
        }
    }
}

double* transpose(double* matrix, int row, int col){
    int i, j;
    double* matrix_res = (double *)malloc(col * row * sizeof(double));
    int row_res = col;
    int col_res = row;
    
    for(i = 0; i < row; ++i)
        for(j = 0; j < col; ++j)
            *(matrix_res + j*col_res + i) = *(matrix + i*col + j);
    
    free(matrix);
    return matrix_res;
}

// You should NOTICE FREE matrix
double* elemwise_multi(double* Ma, double* Mb, int row_a, int col_a, int row_b, int col_b){
    if ((row_a != row_b) || (col_a != col_b)){
        printf("Error when elementwise multiply matrix...");
        exit(1);
    }

    int i, j, k;
    double* matrix_res = (double *)malloc(row_a * col_a * sizeof(double));
    // Actually row_a == row_b and col_a == col_b
    for (i = 0; i < row_a; i++){
        for (j = 0; j < col_b; j++){
            *(matrix_res + i*col_a + j) = ((*(Ma + i*col_a + j)) * (*(Mb + i*col_b + j)));
        }
    }

    return matrix_res;
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

void matrix_add(double* Ma, double* Mb, int row, int col){
    int i, j;
    for (i = 0; i < row; i++){
        for (j = 0; j < col; j++){
            *(Ma + i*col + j) = *(Ma + i*col + j) + *(Mb + i*col + j);
        }
    }
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

    return vec_res;
}

void matrix_copy(double* Ma, double* Mb, int row, int col){
    int i, j;
    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++)
            *(Ma + i*col + j) = *(Mb + i*col + j);
}

double gaussrand()
{
  double x = (double)random() / RAND_MAX,
         y = (double)random() / RAND_MAX,
         z = sqrt(-2 * log(x)) * cos(2 * M_PI * y);
  return z;
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
