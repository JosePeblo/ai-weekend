__global__
void gemm(float* A, float* B, float* C, int M, int N, int K) {

}

int main(void) {
    float *a, *b, *c;

    cudaMallocManaged(&a, sizeof(float)*9);
    cudaMallocManaged(&b, sizeof(float)*9);
    cudaMallocManaged(&c, sizeof(float)*9);


    // float a[] = { 
    //     1, 2, 3,
    //     4, 5, 6,
    //     7, 8, 9,
    // };

    // float b[] = { 
    //     1, 2, 3,
    //     4, 5, 6,
    //     7, 8, 9,
    // };

    // float c[] = { 
    //     0, 0, 0,
    //     0, 0, 0,
    //     0, 0, 0,
    // };

}