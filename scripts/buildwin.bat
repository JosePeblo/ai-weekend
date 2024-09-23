g++ .\src\*.cpp ^
-I./vendor/OpenCL/inc ^
-L./vendor/OpenCL/lib/x64 ^
-lOpenCL ^
-o app.exe
@REM -O3 -march=native -funroll-loops ^