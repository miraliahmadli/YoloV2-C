#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>


void conv2d(double *C, double *A, double *B, int M, int K, int N){
    // M x K, K x N -> M x N

    // for(int i = 0; i < N; i++){
    //     for(int j = 0; j < K; j++){
    //         B[i + j*N] = B[j + i*K];
    //     }
    // }

    // __m256d col, row, res;
    // double store_res[4];
    // for(int i = 0; i < M; i++){
    //     for(int j = 0; j < N; j++){
    //         double sum = 0.0;
    //         for(int k = 0; k < K; k = k + 4){
    //             col = _mm256_loadu_pd(&A[k + i*K]);
    //             row = _mm256_loadu_pd(&B[k + j*K]);
    //             // int a_offset = k + i*K;
    //             // int b_offset = k + j*K;
    //             // int b_offset = j + k*N;
    //             // col = _mm256_set_pd(A[a_offset], A[a_offset + 1], A[a_offset + 2], A[a_offset + 3]);
    //             // row = _mm256_set_pd(B[b_offset], B[b_offset + k], B[b_offset + 2*k], B[b_offset + 3*k]);
    //             res = _mm256_mul_pd(col, row);
    //             // _mm256_storeu_pd(store_res, res);
    //             sum += (*(double *)&res[0]) + (*(double *)&res[1]) + 
    //                     (*(double *)&res[2]) + (*(double *)&res[3]);
    //         }

    //         // for(int mid = 0; mid < K; mid++){
    //         //     C[j + i*N] += A[mid + i*K] * B[mid + j*K];
    //         // }
    //         C[j + i*N] = sum;
    //     }
    // }
    // int size = 0; 
    // for (int i = 0; i < M; i++) 
    //     for (int j = 0; j < K; j++) 
    //         if (A[i*K + j] != 0.0) 
    //             size++; 
  
    // double compactMatrixA[size]; 
    // int compact[2][size];
  
    // int k = 0; 
    // for (int i = 0; i < M; i++) 
    //     for (int j = 0; j < K; j++) 
    //         if (A[i*K + j] != 0.0) 
    //         { 
    //             compact[0][k] = i; 
    //             compact[1][k] = j; 
    //             compactMatrixA[k] = A[i*K + j] ; 
    //             k++; 
    //         } 

    // size = 0; 
    // for (int i = 0; i < K; i++) 
    //     for (int j = 0; j < N; j++) 
    //         if (B[i*K + j] != 0) 
    //             size++; 
  
    // // for (int i = 0; i < M; i++) {
    // int b_offset;
    // int row, col;
    // double val;
    // for (int j = 0; j < N; j++) {
    //     b_offset = j * K;
    //     for(int i = 0; i<size; i++){
    //         row = compact[0][i];
    //         col = compact[1][i];
    //         val = compactMatrixA[i];
    //         C[row * N + j] += val*B[col*N + j];
    //     }
    // }
    // }
    // int compactMatrixB[3][size]; 
  
    // k = 0; 
    // for (int i = 0; i < M; i++) 
    //     for (int j = 0; j < K; j++) 
    //         if (B[i*K + j] != 0) 
    //         { 
    //             compactMatrixB[0][k] = i; 
    //             compactMatrixB[1][k] = j; 
    //             compactMatrixB[2][k] = B[i*K + j] ; 
    //             k++; 
    //         } 
    

    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}