#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>


void print_array(int * array, int size){
    for(int i=0;i<size;i++){
        printf("%d, ", array[i]);
    }
    printf("\n\n");
}

bool check_result(int * array, int size){
    for(int i=0;i<size - 1;i++){
        if(array[i] > array[i + 1]){
            return false;
        }
    }
    return true;
}

// ------------- quick sort ---------------------------------------

// Utility function to swap tp integers
void swap(int* p1, int* p2)
{
    int temp;
    temp = *p1;
    *p1 = *p2;
    *p2 = temp;
}

int partition(int arr[], int low, int high)
{
    // choose the pivot
    int pivot = arr[high];

    // Index of smaller element and Indicate
    // the right position of pivot found so far
    int i = (low - 1);

    for (int j = low; j <= high; j++) {
        // If current element is smaller than the pivot
        if (arr[j] < pivot) {
            // Increment index of smaller element
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

// The Quicksort function Implement

void quickSort(int arr[], int low, int high)
{
    // when low is less than high
    if (low < high) {
        // pi is the partition return index of pivot

        int pi = partition(arr, low, high);

        // Recursion Call
        // smaller element than pivot goes left and
        // higher element goes right
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

// ---------------- bubble sort ----------------------------------

void bubbleSort(int arr[], int n) {
    int i, j;
    for (i = 0; i < n-1; i++) {
        // Last i elements are already in place
        for (j = 0; j < n-i-1; j++) {
            // Swap if the element found is greater than the next element
            if (arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

// -------------- gpu kernel --------------------------------------

__global__ void kernel(int*array, int size, int offset){

    int tid = blockIdx.x *blockDim.x + threadIdx.x;
    if(tid < size/2){
        int index_0 = tid * 2;
        int index_1 = index_0 + 1;
        index_0 += offset;
        index_1 += offset;
        if(index_1 == size){
            index_1 = index_0;
            index_0 = 0;
        }
        if(array[index_0] > array[index_1]){
            int temp = array[index_0];
            array[index_0] = array[index_1];
            array[index_1] = temp;
        }
    }

}


int main(void){

    int size = 200000; // array size

    dim3 block(1024, 1 ,1);
    dim3 grid(98,1,1);

    // host pointers
    int *array, *array_q, *array_b;

    //cuda mallochost an array
    cudaError error;
    error = cudaHostAlloc((void**)&array, size*sizeof(int), cudaHostAllocMapped);
    if(error != cudaSuccess){
        fprintf(stderr, "Error: %s \n", cudaGetErrorString(error));
    }
    int *d_array;
    cudaHostGetDevicePointer((void **)&d_array, (void *)array, 0);


    // malloc an array
    array_q = (int*)malloc(size*sizeof(int));
    array_b = (int*)malloc(size*sizeof(int));

    // Fill the arrays with random values
    int v;
    for (int i = 0; i < size; i++) {
        v = rand();
        array[i] = v;
        array_q[i] = v;
        array_b[i] = v;
    }

    // check if array befor sorting is not sorted
    bool checker = check_result(array, size);
    printf("------------------------------------------------------------\n");
    if(checker == true){
        printf("Array GPU+CPU before is sorted\n");
    }else{
        printf("Array GPU+CPU before is not sorted!\n");
    }

    // Start measuring gpu + cpu implementation time-----------------
    clock_t begin = clock();



    // kernel launch
    bool sem = true;
    int offset = 0;
    bool correct = true;
    while(sem){
        for(int i=0;i<50;i++){
            kernel<<<grid, block>>>(d_array, size, offset);
            cudaDeviceSynchronize();
            if(offset == 1){
                offset = 0;
            }else{
                offset = 1;
            }
        }
        // print_array(h_gpu_result, size);
        correct = true;
        
        for(int i=0;i<size-1;i++){
            if(array[i] > array[i+1]){
                correct = false;
                break;
            }
        }
        if(correct){
            sem = false;
        }
        
    }
    // Stop measuring gpu + cpu implementation time-----------------
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("GPU + CPU time: %4.6fs\n\n", time_spent);

    // checking GPU + CPU results
    checker = check_result(array, size);
    if(checker == true){
        printf("Array GPU+CPU is sorted\n");
    }else{
        printf("Array GPU+CPU is not sorted!\n");
    }


    // quick sort ---------------------------------------------------------------------
    // check if array befor sorting is not sorted
    checker = check_result(array_q, size);
    printf("------------------------------------------------------------\n");
    if(checker == true){
        printf("Array by quickSort before is sorted\n");
    }else{
        printf("Array by quickSort before is not sorted!\n");
    }
    begin = clock(); // start measure wall time
    // Function call
    quickSort(array_q, 0, size - 1);

    end = clock(); // end measure wall time
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("quickSort time: %4.6fs\n\n", time_spent);

    // check quickSort results
    checker = check_result(array_q, size);
    if(checker == true){
        printf("Array by quickSort is sorted\n");
    }else{
        printf("Array by quickSort is not sorted!\n");
    }

    // bubble sort ---------------------------------------------------------------------
    // check if array befor sorting is not sorted
    checker = check_result(array_b, size);
    printf("------------------------------------------------------------\n");
    if(checker == true){
        printf("Array by bubbleSort before is sorted\n");
    }else{
        printf("Array by bubbleSort before is not sorted!\n");
    }
    begin = clock(); // start measure wall time
    // Function call
    bubbleSort(array_b, size);

    end = clock(); // end measure wall time
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("quickSort time: %4.6fs\n\n", time_spent);

    // check quickSort results
    checker = check_result(array_b, size);
    if(checker == true){
        printf("Array by bubbleSort is sorted\n");
    }else{
        printf("Array by bubbleSort is not sorted!\n");
    }

    cudaFreeHost(d_array);
    free(array);
    free(array_q);
    free(array_b);
    cudaDeviceReset();
    return 0;
}
