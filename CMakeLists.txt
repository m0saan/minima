cmake_minimum_required(VERSION 3.25)
project(minima)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)
include_directories(include/cpu_backend)

add_executable(minima
        include/cpu_backend/aligned_buffer.h
        include/cpu_backend/operations.h
        src/cpu_backend/aligned_array.cc
        src/cpu_backend/operations.cc
        src/main.cc)
