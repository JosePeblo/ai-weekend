kernel 
void test(__global float* a, __global float* b) {
    size_t index = get_global_id(0);
    a[index] = b[index] + b[index+1];
}

kernel 
void gemm(
    int M, int N, int K,
    global float* restrict a, 
    global float* restrict b, 
    global float* restrict c
) {
    size_t idx = get_global_id(0);
    c[idx] = (float)idx;
}