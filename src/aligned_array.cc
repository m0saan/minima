#include "../include/cpu_backend/aligned_buffer.h"

namespace minima {
namespace cpu {

AlignedBuffer::AlignedBuffer(const size_t& size) {
  int ret = posix_memalign(reinterpret_cast<void**>(&ptr_), kAlignment, size * kElemSize);
  if (ret != 0) throw std::bad_alloc();
  size_ = size;
}

AlignedBuffer::~AlignedBuffer() {
  free(ptr_);
}

size_t AlignedBuffer::PtrAsInt() const {
  return reinterpret_cast<size_t>(ptr_);
}

std::ostream& operator<< (std::ostream& out, const AlignedBuffer& buffer) {
    out << "[";
    for (size_t i = 0; i < buffer.size_; ++i) {
        out << buffer.ptr_[i];
        if (i != buffer.size_ - 1) out << ", ";
    }
    out << "]";
    return out;
}

}  // namespace cpu
}  // namespace minima
