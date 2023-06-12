#include "../../include/cpu_backend/aligned_buffer.h"
#include <cstddef>

namespace minima {
namespace cpu {

AlignedBuffer::AlignedBuffer(const size_t& size) {
  int ret = posix_memalign(reinterpret_cast<void**>(&buffer_), kAlignment, size * kElemSize);
  if (ret != 0) throw std::bad_alloc();
  this->size_ = size;
}

AlignedBuffer::~AlignedBuffer() {
  free(this->buffer_);
}

size_t AlignedBuffer::PtrAsInt() const {
  return reinterpret_cast<size_t>(buffer_);
}

std::ostream& operator<< (std::ostream& out, const AlignedBuffer& aligned_buffer) {
    out << "[";
    for (size_t i = 0; i < aligned_buffer.size_; ++i) {
        out << aligned_buffer.buffer_[i];
        if (i != aligned_buffer.size_ - 1) out << ", ";
    }
    out << "]";
    return out;
}

size_t AlignedBuffer::size() const {
    return this->size_;
}

void AlignedBuffer::set_element(size_t index, ScalarT value) {
  if (index < size_) {
    buffer_[index] = value;
  } else {
    throw std::out_of_range("Index out of range");
  }
}

ScalarT AlignedBuffer::get_element(size_t index) const {
  if (index < size_) {
    return buffer_[index];
  } else {
    throw std::out_of_range("Index out of range");
  }
}

}  // namespace cpu
}  // namespace minima
