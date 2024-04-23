cmake_minimum_required(VERSION 3.8)
project(basic_cuda LANGUAGES CXX)

find_package(CUDA REQUIRED)

cuda_add_executable(blur main.cpp blur.h blur.cu stb_image.h stb_image_write.h)
