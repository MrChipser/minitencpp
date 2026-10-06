#include "tensor.hpp"

#include <limits>

size_type element_count(const std::vector<index_type>& shape) {
    size_type count = 1;

    for (index_type dimension : shape) {
        if (dimension < 0) {
            throw std::invalid_argument("Tensor dimensions cannot be negative");
        }

        const auto dimension_size = static_cast<size_type>(dimension);
        if (dimension_size != 0 && count > std::numeric_limits<size_type>::max() / dimension_size) {
            throw std::overflow_error("Tensor is too large");
        }

        count *= dimension_size;
    }

    return count;
}

Tensor::Tensor(const std::vector<index_type>& shape) : shape_(shape) {
    data_.resize(element_count(shape));
    compute_strides();
}

Tensor::Tensor(std::initializer_list<index_type> shape) : shape_(shape) {
    data_.resize(element_count(shape));
    compute_strides();
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    size_type stride = 1;
    for (size_type i = shape_.size() - 1; i >= 0; i--) {
        strides_[i] = stride;
        stride *= static_cast<size_type>(shape_[i]);
    }
}

