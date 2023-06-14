#include "../include/cpu_backend/aligned_buffer.h"
#include "../include/cpu_backend/operations.h"
#include <vector>

using namespace minima::cpu;

std::ostream& operator<<(std::ostream& os, const std::vector<int>& v) {
    os << "[";
    for (int i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i != v.size() - 1) 
            os << ", ";
    }
    os << "]";
    return os;
}


// Path: src/main.cc
int main() {
  minima::cpu::AlignedBuffer a(24);
  minima::cpu::AlignedBuffer b(4);
  // std::cout << a << std::endl;

  // minima::cpu::fill(&a, 1.0);
  for (size_t i = 0; i < a.size(); i++)
	{
		a.set_element(i, i);
	}
  std::cout << a << std::endl;

  // (4, 6), (4, 1)
  // minima::cpu::compact(a, &b, {4,6}, {6, 1}, 0);


  // std::vector<int> shape = {2,3};

  // std::vector<int> indices(shape.size(), 0);  // All indices start at 0
  // minima::cpu::scalar_power(a, 2, &b);
  minima::cpu::reduce_sum(a, &b, 6);

  std::cout << b << std::endl;



  return 0;
}