#include "../include/cpu_backend/aligned_buffer.h"
#include "../include/cpu_backend/operations.h"

using namespace minima::cpu;

// Path: src/main.cc
int main() {
  minima::cpu::AlignedBuffer a(10);
  std::cout << a << std::endl;

  minima::cpu::fill(&a, 1.0);
  std::cout << a << std::endl;


  return 0;
}