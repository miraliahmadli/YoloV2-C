#include <iostream>
#include <algorithm>
#include <stdio.h>
using namespace std;

#define BLOCK_SIZE 16
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[index] = a[index] + b[index];
    }

void print_mat(const char * name, int r, int c, double *m){
    printf("Printing %s\n", name);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%.2lf ", m[i * c + j]);
        }
        printf("\n");
    }
}

/*

    Leaky RELU

*/
__global__ void l_relu(double *res, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n && res[index] < 0.0f){
        res[index] *= 0.1;
    }
}

extern "C" {
    void leaky_relu(double *res, int dim){
        double *dev_res;
        int size = dim * sizeof(double);

        // allocate the memory on the GPU
        HANDLE_ERROR( cudaMalloc( (void**)&dev_res,  size) );
        
        // copy the array 'res' to the GPU
        HANDLE_ERROR( cudaMemcpy( dev_res, res, size, cudaMemcpyHostToDevice ) );

        l_relu<<<(dim + 512 -1) / 512, 512>>>(dev_res, dim);
    
        // copy the array 'res' back from the GPU to the CPU
        HANDLE_ERROR( cudaMemcpy( res, dev_res, size, cudaMemcpyDeviceToHost ) );
    
        // free the memory allocated on the GPU
        cudaFree( dev_res );
    }
}


/*

    Batch Norm

*/
__global__ void b_norm (double *res, double *mean, double *gamma, 
                    double *variance, double epsilon, int n, int oc){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int size = n*oc;
    if (index < size){
        int col = index % oc;
        double divisor = sqrt(variance[col] + epsilon);
        double divident = (res[index] - mean[col]) * gamma[col];
        res[index] = divident / divisor;
    }
}

extern "C" {
    void batch_norm(double *res, double *mean, double *gamma, 
                    double *variance, double epsilon, int n, int oc){
        double *dev_res, *dev_mean, *dev_gamma, *dev_variance;
        int size1 = oc * sizeof(double);
        int size2 = n * size1;

        // allocate the memory on the GPU
        HANDLE_ERROR( cudaMalloc( (void**)&dev_res, size2 ) );

        HANDLE_ERROR( cudaMalloc( (void**)&dev_mean, size1) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_gamma, size1 ) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_variance, size1 ) );
        
        // HANDLE_ERROR( cudaMalloc( (void **)&dev_epsilon, sizeof(double) ) );
        
        // copy the arrays to the GPU
        HANDLE_ERROR( cudaMemcpy( dev_res, res, size2, cudaMemcpyHostToDevice ) );

        HANDLE_ERROR( cudaMemcpy( dev_mean, mean, size1, cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( dev_gamma, gamma, size1, cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( dev_variance, variance, size1, cudaMemcpyHostToDevice ) );

        // Kernel invocation
        b_norm<<<(size2 + size1-1) / size1, size1>>>(dev_res, dev_mean, 
            dev_gamma, dev_variance, epsilon, n, oc);

        // copy the arrays back from the GPU to the CPU
        HANDLE_ERROR( cudaMemcpy( res, dev_res, size2, cudaMemcpyDeviceToHost ) );

        // free the memory allocated on the GPU
        cudaFree( dev_res );
        cudaFree( dev_mean );
        cudaFree( dev_gamma );
        cudaFree( dev_variance );
    }
}

/*

    Add Bias

*/

__global__ void dev_add_bias (double *res, double *bias, int size){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size){
        res[index] += bias[index];
    }
}

extern "C" {
    void add_bias(double * res, double * bias, int n){
        double *dev_res, *dev_bias;
        int size1 = n * sizeof(double);
        // int size2 = n * size1;
    
        // allocate the memory on the GPU
        HANDLE_ERROR( cudaMalloc( (void**)&dev_res, size1) );
    
        HANDLE_ERROR( cudaMalloc( (void**)&dev_bias, size1) );
    
        // copy the arrays to the GPU
        HANDLE_ERROR( cudaMemcpy( dev_res, res, size1, cudaMemcpyHostToDevice ) );
    
        HANDLE_ERROR( cudaMemcpy( dev_bias, bias, size1, cudaMemcpyHostToDevice ) );

        // Kernel invocation

        dev_add_bias<<<(n + 1024 - 1) / 1024, 1024>>>(dev_res, dev_bias, n);
    
        // copy the arrays back from the GPU to the CPU
        HANDLE_ERROR( cudaMemcpy( res, dev_res, size1, cudaMemcpyDeviceToHost ) );
    
        // free the memory allocated on the GPU
        cudaFree( dev_res );
        cudaFree( dev_bias );
    }
}

/*

    MAX Pool

*/

/*

    Convolution

*/
__global__ void multABtoC(double *a,double *b, double *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

extern "C"{
    void conv2d(double *C, double *A, double *B, int m, int n, int k)
    {
        // Allocate memory space on the device 
        double *dev_a, *dev_b, *dev_c;
        cudaMalloc((void **) &dev_a, sizeof(double)*m*n);
        cudaMalloc((void **) &dev_b, sizeof(double)*n*k);
        cudaMalloc((void **) &dev_c, sizeof(double)*m*k);

        // copy matrix A and B from host to device memory
        cudaMemcpy(dev_a, A, sizeof(double)*m*n, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, B, sizeof(double)*n*k, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_c, C, sizeof(double)*m*k, cudaMemcpyHostToDevice);

        unsigned int gridev_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int gridev_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(gridev_cols, gridev_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
        // Launch kernel 
        multABtoC<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, m, n, k);    

        // Transefr results from device to host 
        cudaMemcpy(C, dev_c, sizeof(double)*m*k, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // free memory
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
    }
}