#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include<unistd.h> 
#include<stdlib.h> 
#define min(a,b)            (((a) < (b)) ? (a) : (b))
// #define BLOCK 169
int BLOCK, NUM_THR;
double *A_p;
double *B_p;
int M, N, K;

void *mult(void* arg)
{
    int *data = (int *)arg;
    double sum = 0.0;
    int i = 0, j =0;

    int row = *data;
    int col = *(data + 1);
    double p[BLOCK];
    for (j = 0; j < BLOCK; j++){
        sum = 0.0;
        for (i = 0; i < K; i++){
            sum += A_p[row * K + i] * B_p[i * N + col + j];
        }
        p[j] = sum;
    }
    free(arg);
    pthread_exit(p);
}

void conv2d(double *C, double *A, double *B, int M_p, int K_p, int N_p){
    M = M_p; N = N_p; K = K_p;
    A_p = (double *)malloc(M*K*sizeof(double));
    B_p = (double *)malloc(N*K*sizeof(double));
    int size = M*N;
    if(size < 30000) BLOCK = 1;
    else BLOCK = 169;
    NUM_THR = size / BLOCK;

    int i, j;
  
    for(i = 0; i < M*K; i++) A_p[i] = A[i];
    for(i = 0; i < N*K; i++) B_p[i] = B[i];

    pthread_t *threads;
    threads = (pthread_t*)malloc(NUM_THR*sizeof(pthread_t));
    // printf("YUUUUP\n");
    int counter = 0;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j = j + BLOCK){
            int *data;
            data = calloc(2,sizeof(int));
            *data = i;
            *(data + 1) = j;
            if(pthread_create(&threads[counter++], NULL, mult, (void*)data) != 0){
                printf("Create failed at %d %d\n", i, j);
                exit(-1);
                return;
            }
        }
    }
    double *res;
    for (i = 0; i < NUM_THR; i++) {
        void *k;
        pthread_join(threads[i], &k);
        res = k;
        j = 0;
        while(j < BLOCK){
            C[i * BLOCK + j] = res[j];
            j++;
        }
    }

    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k * N + j];
            }
            if (C[i * N + j] - sum > 0.1){
                printf("FAILED at C[%d][%d]: %f is not equal to %.3f\n", i, j, sum, C[i * N + j]);
                exit(-1);
                return;
            }
        }
    }
    
    free(A_p);
    free(B_p);
}