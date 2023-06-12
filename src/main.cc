#include "../include/cpu_backend/ndarray.h"


// Path: src/main.cc
int main() {
  minima::cpu::AlignedBuffer a(10);
  std::cout << a << std::endl;
  return 0;
}