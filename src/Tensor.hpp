#pragma once
#include <cstring>
#include <iostream>
#include <CL/cl.h>
#include <random>

template<typename T>
class Tensor {
public:
    Tensor() = delete;

    Tensor(size_t m, size_t n): m_M(m), m_N(n) {
        m_data = new T[m_M*m_N];
        // for(size_t i = 0; i < m_M*m_N; ++i) {
        //     m_data[i] = 0;
        // }
    }

    Tensor(const T* ptr, size_t m, size_t n): m_M(m), m_N(n) {
        m_data = new T[m_M*m_N];
        std::memcpy(m_data, ptr, m_M*m_N*sizeof(T));
    }

    void Print() const {
        for(size_t i = 0; i<m_M; ++i) {
            for(size_t j = 0; j<m_N; ++j) {
                if (j != 0) printf(", ");
                // printf("%*.1f", 5, m_data[i*m_N + j]);
                printf("%*d", 5, m_data[i*m_N + j]);
            }
            std::cout << std::endl;
        }
    }

    // TODO: Free the memory please!!!

    size_t getDimM() const { return m_M; }
    size_t getDimN() const { return m_N; }

    T* rawPointer() const { return m_data; }

    // static Tensor<float> random(size_t M, size_t N, float min, float max) {
    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::uniform_real_distribution<float> distr(min, max);

    //     Tensor<float> res(M, N);
        
    //     for(int i = 0; i < M*N; ++i) {
    //         res.m_data[i] = distr(gen);
    //     }

    //     return res;
    // }

    static Tensor<int> random(size_t M, size_t N, int min, int max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distr(min, max);

        Tensor<int> res(M, N);
        
        for(int i = 0; i < M*N; ++i) {
            res.m_data[i] = distr(gen);
        }

        return res;
    }

    static Tensor<T> matmul(const Tensor<T> A, const Tensor<T> B) {
        if(A.m_N != B.m_M) {
            throw std::runtime_error("Invalid matrix shapes");
        }

        size_t M = A.m_M;
        size_t N = B.m_N;
        size_t K = A.m_N;

        Tensor<T> C(M, N);

        for(size_t m = 0; m < M; m++) {
            for(size_t n = 0; n < N; n++) {
                T sum = 0;
                for(size_t k = 0; k < K; k++) {
                    sum += A.m_data[m * K + k] * B.m_data[k * N + n];
                }
                C.m_data[m * N + n] = sum;
            }
        }

        // for(size_t m = 0; m < M; m++) {
        //     for(size_t k = 0; k < K; k++) {
        //         for(size_t n = 0; n < N; n++) {
        //             C.m_data[m * N + n] += A.m_data[m * K + k] * B.m_data[k * N + n];
        //         }
        //     }
        // }

        return C;
    }

private:
    T* m_data;
    size_t m_M;
    size_t m_N;
};
