#ifndef MINIMA_CPU_ALIGNED_BUFFER_H
#define MINIMA_CPU_ALIGNED_BUFFER_H

// #include <pybind11/numpy.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>

#include <cstddef>
#include <cstdlib>
#include <new>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>


namespace minima {
namespace cpu {

constexpr size_t kAlignment = 256;
constexpr size_t kTile = 8;
using ScalarT = float;
constexpr size_t kElemSize = sizeof(ScalarT);


/**
 * This is a utility class for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least kTile * kElemSize
 * 
 */

class AlignedBuffer {
 public:
  explicit AlignedBuffer(const size_t& size);
  ~AlignedBuffer();
  size_t PtrAsInt() const;
  size_t size() const;

  friend std::ostream& operator<< (std::ostream& out, const AlignedBuffer& buffer);


 private:
  ScalarT* ptr_;
  size_t size_;
};

void fill(AlignedBuffer* out, ScalarT value);
void compact(const AlignedBuffer& a, AlignedBuffer& out, std::vector<uint16_t> shape, std::vector<uint16_t> strides, size_t offeset);



}  // namespace cpu
}  // namespace minima

#endif  // MINIMA_CPU_ALIGNED_BUFFER_H