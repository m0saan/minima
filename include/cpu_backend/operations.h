#ifndef MINIMA_CPU_OPERATIONS_H
#define MINIMA_CPU_OPERATIONS_H

#include "ndarray.h"

#include "AlignedBuffer.h"

namespace minima {
	namespace cpu {

		void Fill(AlignedBuffer* out, const ScalarT& value);
		void Compact(const AlignedBuffer& a, AlignedBuffer* out, const std::vector<uint16_t>& shape, const std::vector<uint16_t>& strides, size_t offset);

	}  // namespace cpu
}  // namespace minima


#endif // MINIMA_CPU_OPERATIONS_H
