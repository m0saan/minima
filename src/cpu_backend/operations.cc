#include "../../include/cpu_backend/operations.h"
#include <numeric>
#include <vector>

void minima::cpu::fill(AlignedBuffer *out, const ScalarT &value) {
  for (size_t i = 0; i < out->size(); i++) {
    out->set_element(i, value);
  }
}

// Helper function to calculate the index in stride-based representation
size_t calculate_index(const std::vector<uint32_t> &indices,
                       const std::vector<uint32_t> &strides,
                       size_t offset = 0) {
    return std::inner_product(indices.begin(), indices.end(), strides.begin(), offset);
}

// Helper function to increment the multidimensional index
void increment_indices(std::vector<uint32_t>& indices, 
                       const std::vector<uint32_t>& shape) {
    // Start from the last dimension
    for (size_t dim = shape.size() - 1; dim >= 0; --dim) {
        // If the index in this dimension is less than its maximum
        // (shape[dim] - 1), it means we can still increment the index
        // in this dimension
        if (indices[dim] < shape[dim] - 1) {
            ++indices[dim];
            break;
        } else {
            // For all dimensions that can't be incremented further,
            // their indices are reset to 0 (like an odometer)
            indices[dim] = 0;
        }
    }
}


void minima::cpu::compact(const AlignedBuffer& a, AlignedBuffer* out, const std::vector<uint32_t>& shape,
             const std::vector<uint32_t>& strides, size_t offset) {
    std::vector<uint32_t> indices(shape.size(), 0); 

    for (size_t i = 0; i < out->size(); ++i) {
        size_t index = calculate_index(indices, strides, offset);
        out->set_element(i, a.get_element(index));
        increment_indices(indices, shape);
    } 
}

void minima::cpu::ewise_setitem(const AlignedBuffer& a, AlignedBuffer* out, const std::vector<uint32_t>& shape,
                  const std::vector<uint32_t>& strides, size_t offset) {

   std::vector<uint32_t> indices(shape.size(), 0);

    for (size_t i = 0; i < a.size(); ++i) {
        size_t index = calculate_index(indices, strides, offset);
        out->set_element(index, a.get_element(index));
        increment_indices(indices, shape);
    }
}

void minima::cpu::scalar_setitem(const size_t size, ScalarT val, AlignedBuffer* out, const std::vector<uint32_t>& shape,
                   const std::vector<uint32_t>& strides, size_t offset) {
    std::vector<uint32_t> indices(shape.size(), 0);

    for (size_t i = 0; i < size; ++i) {
        size_t index = calculate_index(indices, strides, offset);
        out->set_element(index, val);
        increment_indices(indices, shape);
    }
}


template <typename F>
void eWiseOperation(const minima::cpu::AlignedBuffer& a, const minima::cpu::AlignedBuffer& b, minima::cpu::AlignedBuffer* out, F func) {
  assert(a.size() == b.size());
  for (size_t i = 0; i < a.size(); i++) {
    out->set_element(i, func(a.get_element(i), b.get_element(i)));
  }
}

template <typename F>
void scalarOperation(const minima::cpu::AlignedBuffer& a, minima::cpu::ScalarT val, minima::cpu::AlignedBuffer* out, F func) {
  for (size_t i = 0; i < a.size(); i++) {
    out->set_element(i, func(a.get_element(i), val));
  }
}

template <typename F>
void UnaryOperation(const minima::cpu::AlignedBuffer& a,  minima::cpu::AlignedBuffer* out, F func) {
  for (size_t i = 0; i < a.size(); i++) {
    out->set_element(i, func(a.get_element(i)));
    // out->ptr[i] = func(a.ptr[i]); :3
  }
}

// Function to check for null pointers
void checkNullPointers(const minima::cpu::AlignedBuffer* a, const minima::cpu::AlignedBuffer* b, minima::cpu::AlignedBuffer* out) {
    if (a == nullptr || out == nullptr || (b != nullptr && b == nullptr)) {
        throw std::invalid_argument("Null pointer passed to function");
    }
}

// Function to check size match
void checkSizeMatch(const minima::cpu::AlignedBuffer& a, const minima::cpu::AlignedBuffer& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Size mismatch between input and output arrays");
    }
}


void minima::cpu::ewise_add(const AlignedBuffer& a, const AlignedBuffer& b, AlignedBuffer* out) {
    checkNullPointers(&a, nullptr, out);
    checkSizeMatch(a, *out); 
   eWiseOperation(a, b, out, std::plus<ScalarT>());
}

void minima::cpu::scalar_add(const AlignedBuffer& a, ScalarT val, AlignedBuffer* out) {
    checkNullPointers(&a, nullptr, out);
    checkSizeMatch(a, *out); 
    scalarOperation(a, val, out, std::plus<ScalarT>());
}

void minima::cpu::ewise_mul(const AlignedBuffer& a, const AlignedBuffer& b, AlignedBuffer* out) {
    checkNullPointers(&a, nullptr, out);
    checkSizeMatch(a, *out); 
   eWiseOperation(a, b, out, std::multiplies<>());
}

void minima::cpu::scalar_mul(const AlignedBuffer& a, ScalarT val, AlignedBuffer* out) {
    checkNullPointers(&a, nullptr, out);
    checkSizeMatch(a, *out); 
    scalarOperation(a, val, out, std::multiplies<ScalarT>());
}

void minima::cpu::ewise_div(const AlignedBuffer& a, const AlignedBuffer& b, AlignedBuffer* out) {
    checkNullPointers(&a, nullptr, out);
    checkSizeMatch(a, *out); 
    eWiseOperation(a, b, out, std::divides<ScalarT>());
}

void minima::cpu::scalar_div(const AlignedBuffer& a, ScalarT val, AlignedBuffer* out) {
    checkNullPointers(&a, nullptr, out);
    checkSizeMatch(a, *out); 
    scalarOperation(a, val, out, std::divides<ScalarT>());

}

void minima::cpu::scalar_power(const AlignedBuffer& a, ScalarT val, AlignedBuffer* out) {
    checkNullPointers(&a, nullptr, out);
    checkSizeMatch(a, *out); 
    scalarOperation(a, val, out, [val](ScalarT base, ScalarT exponent) { return std::pow(base, val); });
}

void minima::cpu::ewise_log(const AlignedBuffer& a, AlignedBuffer* out) {
    checkNullPointers(&a, nullptr, out);
    checkSizeMatch(a, *out);
    UnaryOperation(a, out, [](ScalarT s) { return std::log(s); } );
}

void minima::cpu::ewise_exp(const AlignedBuffer& a, AlignedBuffer* out) {
    checkNullPointers(&a, nullptr, out);
    checkSizeMatch(a, *out);    
    UnaryOperation(a, out, [](ScalarT s) { return std::exp(s); });
}

void minima::cpu::ewise_tanh(const AlignedBuffer& a, AlignedBuffer* out) {
    checkNullPointers(&a, nullptr, out);
    checkSizeMatch(a, *out);
    UnaryOperation(a, out, [](ScalarT s) { return std::tanh(s); });
}

void minima::cpu::ewise_maximum(const AlignedBuffer& a, const AlignedBuffer& b, AlignedBuffer* out) {
    checkNullPointers(&a, &b, out);
    checkSizeMatch(a, *out);
    eWiseOperation(a, b, out, [](ScalarT x, ScalarT y) { return std::max(x, y); });
}


void minima::cpu::scalar_maximum(const AlignedBuffer& a, ScalarT val, AlignedBuffer* out) {
  checkNullPointers(&a, nullptr, out);
  checkSizeMatch(a, *out);
  scalarOperation(a, val, out, [](ScalarT x, ScalarT val) { return std::max(x, val); });
}


void minima::cpu::ewise_eq(const AlignedBuffer& a, const AlignedBuffer& b, AlignedBuffer* out) {
    checkNullPointers(&a, nullptr, out);
    checkSizeMatch(a, *out);
    eWiseOperation(a, b, out, [](ScalarT x, ScalarT y) { return x == y; });
}

void  minima::cpu::scalar_eq(const AlignedBuffer& a, ScalarT val, AlignedBuffer* out) {
    checkNullPointers(&a, nullptr, out);
    checkSizeMatch(a, *out);
    scalarOperation(a, val, out, [](ScalarT x, ScalarT val) { return x == val; });
}

void  minima::cpu::ewise_ge(const AlignedBuffer& a, const AlignedBuffer& b, AlignedBuffer* out) {
    checkNullPointers(&a, nullptr, out);
    checkSizeMatch(a, *out);
    eWiseOperation(a, b, out, [](ScalarT x, ScalarT y) { return x >= y; });
}

void  minima::cpu::scalar_ge(const AlignedBuffer& a, ScalarT val, AlignedBuffer* out) {
    checkNullPointers(&a, nullptr, out);
    checkSizeMatch(a, *out);
    scalarOperation(a, val, out, [](ScalarT x, ScalarT val) { return x >= val; });
}

template <typename ReduceOp>
void Reduce(const minima::cpu::AlignedBuffer& a, minima::cpu::AlignedBuffer* out, size_t reduce_size, ReduceOp op) {
    if (!out) {
        throw std::invalid_argument("Output buffer is a nullptr.");
    }

    for(size_t i = 0; i < out->size(); ++i) {
        const minima::cpu::ScalarT* start_ptr = a.data() + (i * reduce_size);
        const minima::cpu::ScalarT* end_ptr = a.data() + ((i+1) * reduce_size);
        out->set_element(i, op(start_ptr, end_ptr));
    }
}

void minima::cpu::reduce_max(const AlignedBuffer& a, AlignedBuffer* out, size_t reduce_size) {
    Reduce(a, out, reduce_size, [](const ScalarT* start, const ScalarT* end){
        return *std::max_element(start, end);
    });
}

void minima::cpu::reduce_sum(const AlignedBuffer& a, AlignedBuffer* out, size_t reduce_size) {
    Reduce(a, out, reduce_size, [](const ScalarT* start, const ScalarT* end){
        return std::accumulate(start, end, 0.0f);
    });
}