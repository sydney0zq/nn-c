#include "header.h"

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
       //*(batch_index + i) = i;

    for (i = 0; i < BATCH_SIZE; i++){
        for (j = 0; j < layer[0]; j++){
            *(X_batch + i*layer[0] + j) = *(X_train + batch_index[i]*layer[0] + j);
        }
        for (j = 0; j < CLASS_NUM; j++){
            *(y_batch + i*CLASS_NUM + j) = *(y_train + batch_index[i]*CLASS_NUM + j);
        }
    }
}

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
            *(Ma + i*col + j) += *(Mb + i*col + j);
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

void read_test_data(double* X_test, int* y_test){
    FILE* fp;
    int i, j;

    // Read test data
    if ((fp = fopen("./data/dataset_test.txt", "r")) != NULL){
        for (i = 0; i < TEST_NUM; i++)
            for (j = 0; j < layer[0]; j++)
                fscanf(fp, "%lf", X_test + i*layer[0] + j);
        fclose(fp);
    }else{
        printf("Read File Error, Now exiting...");
        exit(1);
    }
    if ((fp = fopen("./data/dataset_test_label.txt", "r")) != NULL){
        for (i = 0; i < TEST_NUM; i++)
            for (j = 0; j < CLASS_NUM; j++)
                fscanf(fp, "%d", y_test + i*CLASS_NUM + j);
        fclose(fp);
    }else{
        printf("Read File Error, Now exiting...");
        exit(1);
    }

}

void read_model(double* W1, double* b1, double* W2, double* b2){
    int i, j;
    FILE* fp;

    if ((fp = fopen("./model/W1.txt", "r")) != NULL){
        for(i = 0; i < layer[0]; i++)
            for(j = 0; j < layer[1]; j++)
                fscanf(fp, "%lf", W1 + i*layer[1] + j);
        fclose(fp);
    }else{
        printf("Read file error, now exiting...");
        exit(1);
    }

    if ((fp = fopen("./model/b1.txt", "r")) != NULL){
        for(j = 0; j < layer[1]; j++)
            fscanf(fp, "%lf", b1 + j);
        fclose(fp);
    }else{
        printf("Read file error, now exiting...");
        exit(1);
    } 

    if ((fp = fopen("./model/W2.txt", "r")) != NULL){
        for (i = 0; i < layer[1]; i++)
            for(j = 0; j < layer[2]; j++)
                fscanf(fp, "%lf", W2 + i*layer[2] + j);
        fclose(fp);
    }else{
        printf("Read file error, now exiting...");
        exit(1);
    } 

    if ((fp = fopen("./model/b2.txt", "r")) != NULL){
        for(j = 0; j < layer[2]; j++)
            fscanf(fp, "%lf", b2 + j);
        fclose(fp);
    }else{
        printf("Read file error, now exiting...");
        exit(1);
    } 
}

void save_model(double* W1, double* b1, double* W2, double* b2){
    int i, j;
    FILE* fp;

    if ((fp = fopen("./model/W1.txt", "w")) != NULL){
        for (i = 0; i < layer[0]; i++)
            for (j = 0; j < layer[1]; j++)
                fprintf(fp, "%lf ", *(W1 + i*layer[1] + j));
        fclose(fp);
    }else{
        printf("Write file error, now exiting...");
        exit(1);
    }

    if ((fp = fopen("./model/b1.txt", "w")) != NULL){
        for (i = 0; i < layer[1]; i++)
            fprintf(fp, "%lf ", *(b1 + i));
        fclose(fp);
    }else{
        printf("Write file error, now exiting...");
        exit(1);
    }

    if ((fp = fopen("./model/W2.txt", "w")) != NULL){
        for (i = 0; i < layer[1]; i++)
            for (j = 0; j < layer[2]; j++)
                fprintf(fp, "%lf ", *(W2 + i*layer[2] + j));
        fclose(fp);
    }else{
        printf("Write file error, now exiting...");
        exit(1);
    }

    if ((fp = fopen("./model/b2.txt", "w")) != NULL){
        for (i = 0; i < layer[2]; i++)
            fprintf(fp, "%lf ", *(b2 + i));
        fclose(fp);
    }else{
        printf("Write file error, now exiting...");
        exit(1);
    }
}

double test_accuracy(double* X_test, int* y_test, double* W1, double* b1, double* W2, double* b2){
    
    double* z1; double* z2; double* a1; double* exp_scores; double* probs;
    int i, j;
    // z1 = x.dot(W1) + b1 and a1 = np.tanh(z1)
    z1 = matrix_multi(X_test, W1, TEST_NUM, layer[0], layer[0], layer[1]);  // TODO free -
    matrix_add_vector(z1, b1, TEST_NUM, layer[1]);
    matrix_single_op(z1, TEST_NUM, layer[1], "tanh");
    a1 = z1;

    // z2 = a1.dot(W2) + b2 and exp_scores = np.exp(z2)
    z2 = matrix_multi(a1, W2, TEST_NUM, layer[1], layer[1], layer[2]); // TODO free -
    matrix_add_vector(z2, b2, TEST_NUM, layer[2]);
    matrix_single_op(z2, TEST_NUM, layer[2], "exp");
    exp_scores = z2;
    
    // probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    softmax(exp_scores, TEST_NUM, layer[2]);
    probs = exp_scores;

    int cnt = 0;
    int max_in;
    int ground_truth;
    for (i = 0; i < TEST_NUM; i++){
        max_in = max_index(probs + i*CLASS_NUM, CLASS_NUM);
        for (j = 0; j < CLASS_NUM; j++){
            if (*(y_test + i*CLASS_NUM + j) == 1){
                ground_truth = j;
                break;
            }
        }
        if (ground_truth == max_in)  cnt++;
    }
    free(a1); free(probs);
    return cnt*1.0 / TEST_NUM;
}

int max_index(double* vec, int size){
    int i;
    double max = 0;
    int max_index = 0;
    for (i = 0; i < size; i++){
       if (*(vec + i) > max){
           max = *(vec + i);
           max_index = i;
       }
    }
    return max_index;
}

