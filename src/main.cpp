#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <array>
#include <chrono>

#if __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "Tensor.hpp"

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

    std::ifstream kernelFile("src\\gpu\\opencl\\test.cl");
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

    size_t M = 1024;
    size_t N = 1024;
    size_t K = 1024;

    auto matA = Tensor<int>::random(M, K, 0, 10);
    auto matB = Tensor<int>::random(K, N, 0, 10);
    auto matC = Tensor<int>(M, N);

    auto start = std::chrono::steady_clock::now();

    cl_mem mbuffA = clCreateBuffer(context, CL_MEM_READ_WRITE, M*K * sizeof(int), nullptr, &err);
    cl_mem mbuffB = clCreateBuffer(context, CL_MEM_READ_WRITE, K*N * sizeof(int), nullptr, &err);
    cl_mem mbuffC = clCreateBuffer(context, CL_MEM_READ_WRITE, M*N * sizeof(int), nullptr, &err);

    clEnqueueWriteBuffer(queue, mbuffA, CL_TRUE, 0, M*K * sizeof(float), matA.rawPointer(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, mbuffB, CL_TRUE, 0, K*N * sizeof(float), matB.rawPointer(), 0, nullptr, nullptr);

    cl_kernel matmul = clCreateKernel(program, "gemm", &err);

    clSetKernelArg(matmul, 0, sizeof(cl_mem), &mbuffA);
    clSetKernelArg(matmul, 1, sizeof(cl_mem), &mbuffB);
    clSetKernelArg(matmul, 2, sizeof(cl_mem), &mbuffC);
    clSetKernelArg(matmul, 3, sizeof(int)   , &M);
    clSetKernelArg(matmul, 4, sizeof(int)   , &N);
    clSetKernelArg(matmul, 5, sizeof(int)   , &K);

    std::array<size_t, 2> glowWorksize = { M, N };

    clEnqueueNDRangeKernel(
        /*queue*/              queue,                 
        /*kernel*/             matmul,                
        /*work dims*/          2,                     
        /*global work offset*/ nullptr,
        /*global work size*/   glowWorksize.data(),   
        /*local work size*/    nullptr,               
        /*num ev in waitlist*/ 0,
        /*events in waitlist*/ nullptr,               
        /*event*/              nullptr                
    );
    
    clFinish(queue);

    clEnqueueReadBuffer(queue, mbuffC, CL_TRUE, 0, M*N * sizeof(float), matC.rawPointer(), 0, nullptr, nullptr);

    auto end = std::chrono::steady_clock::now();

    auto time = end - start;

    // matA.Print();
    // std::cout << std::endl;
    // matB.Print();
    // std::cout << std::endl;
    // matC.Print();
    // std::cout << std::endl;

    std::cout << "Finished gpu " << std::chrono::duration<double, std::milli>(time).count() << "ms" << std::endl;

    start = std::chrono::steady_clock::now();

    auto cpures = Tensor<int>::matmul(matA, matB);

    end = std::chrono::steady_clock::now();

    time = end - start;

    std::cout << "Finished cpu " << std::chrono::duration<double, std::milli>(time).count() << "ms" << std::endl;
    
    // cpures.Print();

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // const float ptr1[] = {
    //     1, 2, 3, 1, 
    //     4, 5, 6, 1,
    //     7, 8, 9, 1,
    // };

    // const float ptr2[] = {
    //     1, 2, 3, 1,
    //     4, 5, 6, 1,
    //     7, 8, 9, 1,
    //     1, 1, 1, 1,
    // };

    // const float ptr1[] = {
    //     1, 2, 3, 1, 
    //     4, 5, 6, 1,
    // };

    // const float ptr2[] = {
    //     1, 2,
    //     4, 5,
    //     7, 8,
    //     1, 1,
    // };

    // Tensor<float> tensa(ptr1, 2, 4);
    // Tensor<float> tensb(ptr2, 4, 2);

    // auto res = Tensor<float>::matmul(tensa, tensb);

    // std::cout << std::endl;

    // res.Print();

    return 0;

}