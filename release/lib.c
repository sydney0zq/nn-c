// Copyright © 2017-07-18 Sydney <theodoruszq@gmail.com>

#include "header.h"
extern int layer[3];
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

    if (DEBUG_DOT_MULTI){
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

double* transpose(double* matrix, int row, int col){
    int i, j;
    double* matrix_res = (double *)malloc(col * row * sizeof(double));
    int row_res = col;
    int col_res = row;
    // Displaying the matrix[][] 
    if (DEBUG_TRANS_DISPLAY){
        printf("\n*****Matrix To Be Transposed*****\n");
        for(i = 0; i < row; ++i)
            for(j = 0; j < col; ++j){
                printf("%f  ", *(matrix + i*col + j));
                if (j == col-1)
                    printf("\n");
            }
    }
    
    // Finding the transpose of matrix
    for(i = 0; i < row; ++i)
        for(j = 0; j < col; ++j)
            *(matrix_res + j*col_res + i) = *(matrix + i*col + j);
    
    // Displaying the transpose of matrix
    if (DEBUG_TRANS_DISPLAY){
        printf("\n*****Transposed Matrix*****\n");
        for(i = 0; i < row_res; ++i)
            for(j = 0; j < col_res; ++j){
                printf("%f  ", *(matrix_res + i*col_res + j));
                if(j == col_res-1)
                    printf("\n");
            }
    }
    free(matrix);
    return matrix_res;
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

void matrix_single_op(double* matrix, int row, int col, char* type){
    int i, j;
    if (!strcmp(type, "exp")){
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++){
                *(matrix + i*col + j) = exp(*(matrix + i*col + j));
            }
        }
    }else if (!strcmp(type, "tanh")){
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++){
                *(matrix + i*col + j) = tanh(*(matrix + i*col + j));
            }
        }
    }else if (!strcmp(type, "pow2")){
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++){
                *(matrix + i*col + j) = pow(*(matrix + i*col + j), 2);
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

double* matrix_single_const(double* matrix, double num, int row, int col, char* type){
    int i, j;
    double* matrix_res = (double *)malloc(row*col*sizeof(double));
    if (!strcmp(type, "add")){
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++){
                *(matrix_res + i*col + j) = (*(matrix + i*col + j)) + num;
            }
        }
    }else if (!strcmp(type, "multi")){
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++){
                *(matrix_res + i*col + j) = (*(matrix + i*col + j)) * num;
            }
        }
    }

    if (DEBUG_MATRIX_SINGLE_CONST){
        printf("\n*****Print Origin******\n");
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++){
                printf("%f\t", *(matrix + i*col + j));
            }
        }
        printf("\n*****Print Result******\n");
        for (i = 0; i < row; i++){
            for (j = 0; j < col; j++){
                printf("%f\t", *(matrix_res + i*col + j));
            }
        }
    }
    return matrix_res;
}

void matrix_add(double* Ma, double* Mb, int row, int col){
    int i, j;
    for (i = 0; i < row; i++){
        for (j = 0; j < col; j++){
            *(Ma + i*col + j) = *(Ma + i*col + j) + *(Mb + i*col + j);
        }
    }
}

/*
 * Read data from a file which is generated from python script
 * gen_train_data.py
 */
void read_data(char* path, struct data_box* ptr_train_data, double* X){
    FILE* fp;
    struct data_box* base_ptr = ptr_train_data;
    int i;

    if ((fp = fopen(path, "r")) != NULL){
        for (i = 0; i < TRAIN_NUM; i++, ptr_train_data++){
            fscanf(fp, "%lf %lf %d", 
                &(ptr_train_data->xc), &(ptr_train_data->yc), &(ptr_train_data->label));
            *(X + i*layer[0]) = ptr_train_data->xc;
            *(X + i*layer[0] + 1) = ptr_train_data->yc;
        }
    }else{
        printf("Read File Error, Now exiting...");
        exit(1);
    }
    fclose(fp);
    ptr_train_data = base_ptr;
    //for (i = 0; i < TRAIN_NUM; i++, ptr_train_data++){
    //    printf("%f, %f, %d\n", ptr_train_data->xc, ptr_train_data->yc, ptr_train_data->label);   
    //}
}

void debug_print_init_value(double* W1, double* b1, double* W2, double* b2){
    int row, col;
    printf("Now print init value of W1, b1, W2, b2...\n");
    printf("**********W1**********\n");
    for (row = 0; row < layer[0]; row++){
        for (col = 0; col < layer[1]; col++){
            printf("%f\t", *(W1 + row * layer[1] + col));
        }
        printf("\n");
    }
    printf("**********************\n");
    printf("\n**********b1**********\n");
    for (col = 0; col < layer[1]; col++)
        printf("%f\t", *(b1 + col));
    printf("\n**********************\n");
    printf("\n**********W2**********\n");
    for (row = 0; row < layer[1]; row++){
        for (col = 0; col < layer[2]; col++){
            printf("%f\t", *(W2 + row * layer[2] + col));
        }
        printf("\n");
    }
    printf("**********************\n");
    printf("\n**********b2**********\n");
    for (col = 0; col < layer[2]; col++){
        printf("%f\t", *(b2 + col));
    }
    printf("\n*********************\n");
}

double gaussrand()
{
  double x = (double)random() / RAND_MAX,
         y = (double)random() / RAND_MAX,
         z = sqrt(-2 * log(x)) * cos(2 * M_PI * y);
  return z;
}

double calculate_loss(struct data_box* ptr_train_data, double* X, double* W1, double* b1, double* W2, double* b2){
    // Forward propagation to calculate our prediction
    double* z1;
    double* z2;
    double* a1;
    double* exp_scores;
    
    double* probs = (double *)malloc(TRAIN_NUM * layer[2] * sizeof(double));
    int row, col, i;

    // z1 = X.dot(W1) + b1 and a1 = np.tanh(z1)
    z1 = matrix_multi(X, W1, TRAIN_NUM, layer[0], layer[0], layer[1]);  // TODO free
    matrix_add_vector(z1, b1, TRAIN_NUM, layer[1]);
    matrix_single_op(z1, TRAIN_NUM, layer[1], "tanh");
    a1 = z1;

    // z2 = a1.dot(W2) + b2 and exp_scores = np.exp(z2)
    z2 = matrix_multi(a1, W2, TRAIN_NUM, layer[1], layer[1], layer[2]); // TODO free
    matrix_add_vector(z2, b2, TRAIN_NUM, layer[2]);
    matrix_single_op(z2, TRAIN_NUM, layer[2], "exp");
    exp_scores = z2;

    // Caculate the probability
    double tmp;
    for (row = 0; row < TRAIN_NUM; row++){
        tmp = 0;
        // Sum the whole row value
        for (col = 0; col < layer[2]; col++){
            tmp += *(exp_scores + row*layer[2] + col);
        }
        for (col = 0; col < layer[2]; col++){
            *(probs + row*layer[2] + col) = *(exp_scores + row*layer[2] + col) / tmp;
        }
    }

    //print_matrix(probs, TRAIN_NUM, layer[2]);
    // Calculating the loss
    double loss=0;
    for (i = 0; i < TRAIN_NUM; i++){
        //printf("In calc loss: %f\n", log(*(probs+i*layer[2]+(ptr_train_data+i)->label)));
        loss += ( -log(*(probs+i*layer[2]+(ptr_train_data+i)->label)) );
    }
    // Add regulatization term to loss (optional)
    // data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    free(z1);
    free(z2);
    return (1.0*loss) / TRAIN_NUM;
}

void print_matrix(double* m, int row, int col){
    int i, j;
    printf("\n*********\n");
    for (i = 0; i < row; i++){
        for (j = 0; j < col; j++){
            printf("%f\t", *(m+i*col+j));
        }
        printf("\n");
    }
    printf("***********\n");
}
