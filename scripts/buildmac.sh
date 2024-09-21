# clang -Xclang -finclude-default-header -cl-std=CL2.0 -emit-llvm -c test.cl
clang++ main.cpp -framework OpenCL -o app