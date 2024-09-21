#include <iostream>
#include <fstream>
#include <vector>
#include <OpenCL/cl.h>

#define DATA_SIZE 9

void matmul() {

}

int main(void) {
    // OpenCl initialization
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    // Platform and device info
    err = clGetPlatformIDs(1, &platform, nullptr);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr);

    // Context and command queue
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);


    // Define host-side data
    float hostA[DATA_SIZE] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    float hostB[DATA_SIZE] = { 2, 4, 8, 10, 12, 14, 16, 18, 20 };

    // Memory allocation
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_SIZE * sizeof(float), nullptr, &err);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_WRITE, DATA_SIZE * sizeof(float), nullptr, &err);

    err = clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, DATA_SIZE * sizeof(float), hostB, 0, nullptr, nullptr);

    // std::ifstream kernelFile("test.bc", std::ios::binary | std::ios::ate);
    // if(!kernelFile.is_open()) {
    //     std::cerr << "Failed to open kernel file." << std::endl;
    //     return 1;
    // }

    // size_t kernelSize = kernelFile.tellg();
    // kernelFile.seekg(0, std::ios::beg);
    // std::vector<char> kernelSource(kernelSize);
    // kernelFile.read(kernelSource.data(), kernelSize);
    // kernelFile.close();

    // cl_program program = clCreateProgramWithBinary(
    //     context, 
    //     1,
    //     &device, 
    //     &kernelSize, 
    //     (const unsigned char**)&kernelSource[0], 
    //     nullptr, 
    //     &err
    // );

    std::ifstream kernelFile("test.cl");
    if(!kernelFile.is_open()) {
        std::cerr << "Failed to open kernel file." << std::endl;
        return 1;
    }

    std::string kernelSource = std::string(
        std::istreambuf_iterator<char>(kernelFile), 
        std::istreambuf_iterator<char>()
    );

    const char* kernelString = kernelSource.c_str();
    size_t kernelSize = kernelSource.size();

    cl_program program = clCreateProgramWithSource(
        context, 
        1, 
        &kernelString,
        &kernelSize,
        &err
    );

    if(err != CL_SUCCESS) {
        std::cerr << "Failed to create program: " << err << std::endl;
        return 1;
    }


    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build log: " << log.data() << std::endl;
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "test", &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);

    size_t globalWorkSize = DATA_SIZE;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        std::cerr << "Failed to enqueue kernel: " << err << std::endl;
        return 1;
    }

    // Wait for the command queue to finish
    clFinish(queue);

    // Read resutls from the buffer
    clEnqueueReadBuffer(queue, bufferA, CL_TRUE, 0, DATA_SIZE * sizeof(float), hostA, 0, nullptr, nullptr);

    // for(int i = 0; i < 10; ++i) {
    //     printf("hostA[%d] = %.2f\n", i, hostA[i]);
    // }

    // ===============================================MATMUL===============================================
    #define MAT_N 4
    #define MAT_M 5
    #define MAT_SIZE MAT_N*MAT_M

    float matA[MAT_SIZE] = { 0 };
    float matB[MAT_SIZE] = { 0 };
    float matC[MAT_SIZE] = { 0 };

    cl_mem mbuffA = clCreateBuffer(context, CL_MEM_READ_WRITE, MAT_SIZE * sizeof(float), nullptr, &err);
    cl_mem mbuffB = clCreateBuffer(context, CL_MEM_READ_WRITE, MAT_SIZE * sizeof(float), nullptr, &err);
    cl_mem mbuffC = clCreateBuffer(context, CL_MEM_READ_WRITE, MAT_SIZE * sizeof(float), nullptr, &err);

    cl_kernel matmul = clCreateKernel(program, "gemm", &err);

    clSetKernelArg(matmul, 0, sizeof(cl_mem), &mbuffA);
    clSetKernelArg(matmul, 1, sizeof(cl_mem), &mbuffB);
    clSetKernelArg(matmul, 2, sizeof(cl_mem), &mbuffC);

    globalWorkSize = MAT_SIZE;

    clEnqueueNDRangeKernel(queue, matmul, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, mbuffC, CL_TRUE, 0, MAT_SIZE * sizeof(float), matC, 0, nullptr, nullptr);

    clFinish(queue);

    for(size_t i = 0; i < MAT_N; ++i) {
        for(size_t j = 0; j < MAT_M; ++j) {
            if (j != 0) printf(", ");
            printf("%*.0f", 3, matC[i*MAT_M + j]);
        }
        std::cout << std::endl;
    }

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;

}