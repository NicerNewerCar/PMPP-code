cmake_minimum_required(VERSION 3.8)
project(basic_cuda LANGUAGES CXX)

find_package(CUDA)

cuda_add_executable(basic_ex main.cpp vecAdd.h vecAdd.cu)
