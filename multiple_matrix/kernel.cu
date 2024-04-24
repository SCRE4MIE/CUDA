/*
TESTED ON: RTX 4070 TI AND AMD RYZEN 7 7800X3D
GPU WALL TIME: 155.86s
CPU WALL TIME: 927.14s
==================================================
Multiplication of two 2D matrices.

Each thread corresponds to one operation: multiplying an element 
from matrix A by matrix B and then summing the result to the element 
of the resulting matrix.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void kernel(int*arr_a, int*arr_b, int*arr_c, int m, int n, int k, long long int num_of_operations){

    long long int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
    long long int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if (threadId < num_of_operations){
        int x_a = threadId / (m*n);
        int y_a = threadId % m;

        int x_b = threadId % m;
        int y_b = (threadId / m ) % n;

        int x_c = threadId / (m*n);
        int y_c = (threadId / m ) % n;

        int multiple = arr_a[x_a * m + y_a] * arr_b[x_b * n + y_b];
        atomicAdd(&(arr_c[x_c * n + y_c]), (int)multiple);
    }
}

void displayMatrix(int*array, int m, int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            printf("%d, ", array[i*m+j]);
        }
        printf("\n");
    }
    printf("\n");
}


void mult_matrix_cpu(int *matrix1, int *matrix2, int *result, int rows1, int cols1, int cols2) {
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            int sum = 0;
            for (int k = 0; k < cols1; k++) {
                sum += matrix1[i * cols1 + k] * matrix2[k * cols2 + j];
            }
            result[i * cols2 + j] = sum;
        }
    }
}

int main(void){

    const int m = 10000;
    const int n = 6000;
    const int k = 10000;

    // host pointers
    int *h_arr_a, *h_arr_b, *h_gpu_result, *h_cpu_result;
    // allocate memory for host pointers
    h_arr_a = (int*)malloc(m*n*sizeof(int));
    h_arr_b = (int*)malloc(n*k*sizeof(int));
    h_gpu_result = (int*)malloc(n*n*sizeof(int));
    h_cpu_result = (int*)malloc(n*n*sizeof(int));

    //initialize host pointers
    for(int i=0;i<m*n;i++){
        h_arr_a[i] = 1;
    }
    for(int i=0;i<n*k;i++){
        h_arr_b[i] = 1;
    }
    
    // Start measuring gpu time-----------------
    clock_t begin = clock();

    //device pointers
    cudaError error;
    int *d_arr_a, *d_arr_b, *d_arr_c;
    error = cudaMalloc((void**)&d_arr_a, m*n*sizeof(int));
    if(error != cudaSuccess){
        fprintf(stderr, "Error: %s \n", cudaGetErrorString(error));
    }
    error = cudaMalloc((void**)&d_arr_b, n*k*sizeof(int));
     if(error != cudaSuccess){
        fprintf(stderr, "Error: %s \n", cudaGetErrorString(error));
    }
    error = cudaMalloc((void**)&d_arr_c, n*n*sizeof(int));
     if(error != cudaSuccess){
        fprintf(stderr, "Error: %s \n", cudaGetErrorString(error));
    }
    // copy array a and b to a GPU memory
    cudaMemcpy(d_arr_a, h_arr_a, m*n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_b, h_arr_b, n*k*sizeof(int), cudaMemcpyHostToDevice);

    // kernel launch
    long long int num_of_operations = (long long int)n * n * m; // operation is a multiple a element and b element and += to c result index (num of threads)
    
    dim3 block(32,32,1);
    dim3 grid(18750,18750,1);

    kernel<<<grid, block>>>(d_arr_a, d_arr_b, d_arr_c, m, n, k, num_of_operations);
    cudaMemcpy(h_gpu_result, d_arr_c, n*n*sizeof(int), cudaMemcpyDeviceToHost);

    // Stop measuring gpu time-----------------
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("GPU time: %4.6fs\n\n", time_spent);

    // cpu implementation----------------------------------------

    // Start measuring cpu time-----------------
    begin = clock();
    mult_matrix_cpu(h_arr_a, h_arr_b, h_cpu_result, n, m, n);
    // Stop measuring cpu time-----------------
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("CPU time: %4.6fs\n\n", time_spent);

    // display matrix a, b, gpu result and cpu_result

    // displayMatrix(h_arr_a, m, n);
    // displayMatrix(h_arr_b, n, k);
    // printf("GPU result:\n");
    // displayMatrix(h_gpu_result, n, n);
    // printf("CPU result:\n");
    // displayMatrix(h_cpu_result, n, n);

    // check cpu vs gpu:

    bool checkpoint = true;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(h_cpu_result[i *n + j] != h_gpu_result[i *n + j]){
                checkpoint = false;
                printf("%d != %d\n", h_cpu_result[i *n + j], h_gpu_result[i *n + j]);
            }
        }
    }
    if(checkpoint == false){
        printf("Error, gpu result != cpu result!");
    }
    
    // free memory
    cudaFree(d_arr_a);
    cudaFree(d_arr_b);
    cudaFree(d_arr_c);
    free(h_gpu_result);
    free(h_cpu_result);
    free(h_arr_a);
    free(h_arr_b);

    return 0;
}