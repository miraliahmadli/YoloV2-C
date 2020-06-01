#include <iostream>
#include <algorithm>
#include <stdio.h>
using namespace std;

#define M 512
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

void print_mat(const char * name, int r, int c, float *m){
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
__global__ void l_relu(float *res, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n && res[index] < 0.0f){
        res[index] *= 0.1;
    }
}

extern "C" {
    void leaky_relu(float *res, int dim){
        float *dev_res;
        int size = dim * sizeof(float);

        // allocate the memory on the GPU
        HANDLE_ERROR( cudaMalloc( (void**)&dev_res,  size) );
        
        // copy the array 'res' to the GPU
        HANDLE_ERROR( cudaMemcpy( dev_res, res, size, cudaMemcpyHostToDevice ) );

        l_relu<<<(dim + M-1) / M, M>>>(dev_res, dim);
    
        // copy the array 'res' back from the GPU to the CPU
        HANDLE_ERROR( cudaMemcpy( res, dev_res, size, cudaMemcpyDeviceToHost ) );
    
        // free the memory allocated on the GPU
        cudaFree( dev_res );
    }
}


/*

    Batch Norm

*/
__global__ void b_norm (float *res, float *mean, float *gamma, 
                    float *variance, float epsilon, int n, int oc){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int size = n*oc;
    if (index < size){
        int col = index % oc;
        float divisor = sqrt(variance[col] + epsilon);
        float divident = (res[index] - mean[col]) * gamma[col];
        res[index] = divident / divisor;
    }
}

extern "C" {
    void batch_norm(float *res, float *mean, float *gamma, 
                    float *variance, float epsilon, int n, int oc){
        float *dev_res, *dev_mean, *dev_gamma, *dev_variance;
        int size1 = oc * sizeof(float);
        int size2 = n * size1;

        // allocate the memory on the GPU
        HANDLE_ERROR( cudaMalloc( (void**)&dev_res, size2 ) );

        HANDLE_ERROR( cudaMalloc( (void**)&dev_mean, size1) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_gamma, size1 ) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_variance, size1 ) );
        
        // HANDLE_ERROR( cudaMalloc( (void **)&dev_epsilon, sizeof(float) ) );
        
        // copy the arrays to the GPU
        HANDLE_ERROR( cudaMemcpy( dev_res, res, size2, cudaMemcpyHostToDevice ) );

        HANDLE_ERROR( cudaMemcpy( dev_mean, mean, size1, cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( dev_gamma, gamma, size1, cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( dev_variance, variance, size1, cudaMemcpyHostToDevice ) );

        // Kernel invocation
        b_norm<<<(size2 + M-1) / M, M>>>(dev_res, dev_mean, 
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

__global__ void dev_add_bias (float *res, float *bias, int n, int oc){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int size = n*oc;
    if (index < size){
        res[index] += bias[index % oc];
    }
}

extern "C" {
    void add_bias(float * res, float * bias, int n, int oc){
        float *dev_res, *dev_bias;
        int size1 = oc * sizeof(float);
        int size2 = n * size1;
    
        // allocate the memory on the GPU
        HANDLE_ERROR( cudaMalloc( (void**)&dev_res, size2) );
    
        HANDLE_ERROR( cudaMalloc( (void**)&dev_bias, size1) );
    
        // copy the arrays to the GPU
        HANDLE_ERROR( cudaMemcpy( dev_res, res, size2, cudaMemcpyHostToDevice ) );
    
        HANDLE_ERROR( cudaMemcpy( dev_bias, bias, size1, cudaMemcpyHostToDevice ) );

        // Kernel invocation

        dev_add_bias<<<(size2 + M-1) / M, M>>>(dev_res, dev_bias, n, oc);
    
        // copy the arrays back from the GPU to the CPU
        HANDLE_ERROR( cudaMemcpy( res, dev_res, size2, cudaMemcpyDeviceToHost ) );
    
        // free the memory allocated on the GPU
        cudaFree( dev_res );
        cudaFree( dev_bias );
    }
}

/*

    MAX Pool

*/

// __global__ void maxpool (float *res, float *prev_res, int *strides, int *ksize, int n, int oc){
//     int index = threadIdx.x + blockIdx.x * blockDim.x;
//     int size = n*oc;
//     if (index < size){
//         res[index] += bias[index % oc];
//     }
// }

// extern "C" {
//     void maxpool2d(float *res, float *prev_res, int *strides, int *ksize, int n, int oc){
//         float *dev_res, *dev_bias;
//         int size1 = oc * sizeof(float);
//         int size2 = n * size1;
    
//         // allocate the memory on the GPU
//         HANDLE_ERROR( cudaMalloc( (void**)&dev_res, size2) );
    
//         HANDLE_ERROR( cudaMalloc( (void**)&dev_bias, size1) );
    
//         // copy the arrays to the GPU
//         HANDLE_ERROR( cudaMemcpy( dev_res, res, size2, cudaMemcpyHostToDevice ) );
    
//         HANDLE_ERROR( cudaMemcpy( dev_bias, bias, size1, cudaMemcpyHostToDevice ) );

//         // Kernel invocation

//         dev_add_bias<<<(size2 + M-1) / M, M>>>(dev_res, dev_bias, n, oc);
    
//         // copy the arrays back from the GPU to the CPU
//         HANDLE_ERROR( cudaMemcpy( res, dev_res, size2, cudaMemcpyDeviceToHost ) );
    
//         // free the memory allocated on the GPU
//         cudaFree( dev_res );
//         cudaFree( dev_bias );
//     }
// }

/*

    CONV 2d

*/