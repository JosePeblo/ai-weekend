kernel 
void test(__global float* a, __global float* b) {
    size_t index = get_global_id(0);
    a[index] = b[index] + b[index+1];
}

kernel 
void gemm(
    global int* restrict a, 
    global int* restrict b, 
    global int* restrict c,
    int M, int N, int K
) {
    // in cuda >
    // globalTID = blockidx.x * blockDim.x + threadidx.x
    // int idx = get_global_id(0);
    int idx = get_global_id(1) * get_global_size(0) + get_global_id(0);
    // get_group_id()
    // int wd = get_work_dim();
    // c[idx] = (float)idx;
    // c[idx] = a[idx];

    int row = get_global_id(1) * K;
    int col = get_global_id(0);

    int sum = 0;

    for(int k = 0; k < K; ++k) {
        sum += a[row + k] * b[k * N + col];
    }

    c[idx] = sum;
}