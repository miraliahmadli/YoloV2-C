#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_api.h>
#include <cublas_v2.h>

bool CompareFeatureMtoN_gpu(float * featureM, float * featureN, float * result, 
    int count_m, int count_n, int size, int gpu_id) {
    float *dev_featureM = 0;
    float *dev_featureN = 0;
    float *dev_result = 0;
    const float alpha = 1, beta = 0;
    cublasHandle_t git handle;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(gpu_id);
    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        goto out;
    }
    cublasCreate(&handle);

    cudaStatus = cudaMalloc((void**)&dev_featureM, count_m * size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("%s, line %d, cudaMalloc failed!\n", __func__, __LINE__);
        goto out;
    }
    cudaStatus = cudaMalloc((void**)&dev_featureN, count_n * size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("%s, line %d, cudaMalloc failed!\n", __func__, __LINE__);
        goto out;
    }
    cudaStatus = cudaMalloc((void**)&dev_result, count_m * count_n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("%s, line %d, cudaMalloc failed!\n", __func__, __LINE__);
        goto out;
    }

    cudaStatus = cudaMemcpy(dev_featureM, featureM, count_m * size * sizeof(float), 
        cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("%s, line %d, cudaMalloc failed!\n", __func__, __LINE__);
        goto out;
    }
    cudaStatus = cudaMemcpy(dev_featureN, featureN, count_n * size * sizeof(float), 
        cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("%s, line %d, cudaMalloc failed!\n", __func__, __LINE__);
        goto out;
    }

    /*

    CUBLAS assumes that the matrix in the device is stored in column major:

    " where α and β are scalars, and A , B and C are matrices stored in column-major 
    format with dimensions op ( A ) m × k , op ( B ) k × n and C m × n , respectively. 

     Also, for matrix A


     // Multiply the arrays A and B on GPU and save the result in C (coloum-major)
      // C(m,n) = A(m,k) * B(k,n)

      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
     */

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, count_n, count_m, size, 
        &alpha, dev_featureN, size, dev_featureM, size, &beta, dev_result, count_n);
    cudaStatus = cudaThreadSynchronize();

    cudaStatus = cudaMemcpy(result, dev_result, count_m * count_n  * sizeof(float), 
        cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("%s, line %d, cudaMemcpy failed!\n", __func__, __LINE__);
        goto out;
    }

out:
    if(dev_featureM) cudaFree(dev_featureM);
    if(dev_featureN) cudaFree(dev_featureN);
    if(dev_result) cudaFree(dev_result);
    cublasDestroy(handle);
    return cudaStatus == cudaSuccess;
}