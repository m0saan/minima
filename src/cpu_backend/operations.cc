#include "../../include/cpu_backend/operations.h"

void minima::cpu::fill(AlignedBuffer* out, const ScalarT& value)
{
	for (size_t i = 0; i < out->size(); i++)
	{
		out->set_element(100, value);
	}
}
