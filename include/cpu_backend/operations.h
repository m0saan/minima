#ifndef MINIMA_CPU_OPERATIONS_H
#define MINIMA_CPU_OPERATIONS_H

#include "aligned_buffer.h"

namespace minima {
namespace cpu {

/**
 * @brief Fill an AlignedBuffer with a specific value.
 *
 * @param out The buffer to be filled.
 * @param value The value to fill the buffer with.
 */
void fill(AlignedBuffer *out, const ScalarT &value);

/**
 * @brief Compact a given buffer with a specific shape and stride, with an
 * offset.
 *
 * @param a The buffer to be compacted.
 * @param out The buffer to store the result of the compaction.
 * @param shape The shape of the compacted buffer.
 * @param strides The stride for the compacting operation.
 * @param offset The offset for the compacting operation.
 */
void compact(const AlignedBuffer &a, AlignedBuffer *out,
             const std::vector<uint16_t> &shape,
             const std::vector<uint16_t> &strides, size_t offset);

/**
 * @brief Sets items in a non-compact array.
 * @param a Compact array whose items will be written to out.
 * @param out Non-compact array whose items are to be written.
 * @param shape Shapes of each dimension for a and out.
 * @param strides Strides of the out array.
 * @param offset Offset of the out array.
 */
void ewise_setitem(const AlignedBuffer &a, AlignedBuffer *out,
                   std::vector<uint32_t> shape, std::vector<uint32_t> strides,
                   size_t offset);

/**
 * @brief Sets items in a non-compact array.
 * @param size Number of elements to write in out array.
 * @param val Scalar value to write.
 * @param out Non-compact array whose items are to be written.
 * @param shape Shapes of each dimension of out.
 * @param strides Strides of the out array.
 * @param offset Offset of the out array.
 */
void scalar_setitem(const size_t size, ScalarT val, AlignedBuffer *out,
                    std::vector<uint32_t> shape, std::vector<uint32_t> strides,
                    size_t offset);

/**
 * @brief Element-wise addition of two buffers.
 * @param a First input buffer.
 * @param b Second input buffer.
 * @param out Output buffer where results are written.
 */
void ewise_add(const AlignedBuffer &a, const AlignedBuffer &b,
               AlignedBuffer *out);

/**
 * @brief Adds a scalar value to each element of a buffer.
 * @param a Input buffer.
 * @param val Scalar value to add.
 * @param out Output buffer where results are written.
 */
void scalar_add(const AlignedBuffer &a, ScalarT val, AlignedBuffer *out);

/**
 * @brief Multiplies two matrices using a naive three-loop algorithm.
 * @param a First input matrix.
 * @param b Second input matrix.
 * @param out Output matrix where results are written.
 * @param m Rows of a / out.
 * @param n Columns of a / rows of b.
 * @param p Columns of b / out.
 */
void matmul(const AlignedBuffer &a, const AlignedBuffer &b, AlignedBuffer *out,
            uint32_t m, uint32_t n, uint32_t p);

/**
 * @brief Multiplies together two TILE x TILE matrices and adds the result to
 * out.
 * @param a First input matrix.
 * @param b Second input matrix.
 * @param out Output matrix where results are written.
 */
void aligned_dot(const float *__restrict__ a, const float *__restrict__ b,
                 float *__restrict__ out);

/**
 * @brief Performs matrix multiplication on tiled representations of array.
 * @param a First input matrix.
 * @param b Second input matrix.
 * @param out Output matrix where results are written.
 * @param m Rows of a / out.
 * @param n Columns of a / rows of b.
 * @param p Columns of b / out.
 */
void matmul_tiled(const AlignedBuffer &a, const AlignedBuffer &b,
                  AlignedBuffer *out, uint32_t m, uint32_t n, uint32_t p);

/**
 * @brief Reduces an array to a single maximum value.
 * @param a Input buffer.
 * @param out Output buffer where maximum value is written.
 * @param reduce_size Number of elements in input buffer to reduce.
 */
void reduce_max(const AlignedBuffer &a, AlignedBuffer *out, size_t reduce_size);

/**
 * @brief Reduces an array to the sum of its elements.
 * @param a Input buffer.
 * @param out Output buffer where sum is written.
 * @param reduce_size Number of elements in input buffer to reduce.
 */
void reduce_sum(const AlignedBuffer &a, AlignedBuffer *out, size_t reduce_size);

} // namespace cpu
} // namespace minima

#endif // MINIMA_CPU_OPERATIONS_H
