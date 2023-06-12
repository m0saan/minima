#include "operations.h"

void minima::cpu::Fill(AlignedBuffer* out, const ScalarT& value)
{
	for (size_t i = 0; i < out.size(); i++)
	{
		out.ptr_[i] = value
	}
}
