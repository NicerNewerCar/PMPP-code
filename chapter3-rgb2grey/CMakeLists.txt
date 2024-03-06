cmake_minimum_required(VERSION 3.8)
project(basic_cuda LANGUAGES CXX)

find_package(CUDA REQUIRED)

cuda_add_executable(rgb2grey main.cpp rgb2grey.h rgb2grey.cu stb_image.h stb_image_write.h)
