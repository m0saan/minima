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
 * @class AlignedBuffer
 * @brief AlignedBuffer is a utility class for maintaining an array aligned to a specified size in memory.
 *
 * The alignment is useful for optimization of memory access in certain situations,
 * such as in scientific or numerical applications.
 */
class AlignedBuffer {
 public:

  /**
   * @brief Construct an AlignedBuffer.
   * 
   * This constructor creates a buffer of a specified size, aligning the buffer to the ALIGNMENT size.
   *
   * @param size The size of the buffer to allocate.
   */
  explicit AlignedBuffer(const size_t& size);

  /**
   * @brief Destructor for the AlignedBuffer.
   *
   * This destructor frees the memory allocated for the buffer.
   */
  ~AlignedBuffer();

  /**
   * @brief Get the pointer address as an integer.
   *
   * @return The address of the buffer, cast to a size_t.
   */
  size_t PtrAsInt() const;

  /**
   * @brief Get the size of the buffer.
   *
   * @return The size of the buffer.
   */
  size_t size() const;

  /**
   * @brief Set an element in the buffer.
   *
   * @param index The index of the element to set.
   * @param value The value to set.
   */
  void set_element(size_t index, ScalarT value);

  /**
   * @brief Get an element from the buffer.
   *
   * @param index The index of the element to get.
   *
   * @return The value of the element at the specified index.
   */
  ScalarT get_element(size_t index) const;

  /**
   * @brief Overload the << operator for the AlignedBuffer.
   *
   * @param out The output stream to write to.
   * @param buffer The buffer to write.
   *
   * @return The output stream.
   */
  friend std::ostream& operator<< (std::ostream& out, const AlignedBuffer& buffer);

 private:

  /// @brief The buffer.
  ScalarT* buffer_;

  /// @brief The size of the buffer.
  size_t size_;
};


}  // namespace cpu
}  // namespace minima

#endif  // MINIMA_CPU_ALIGNED_BUFFER_H