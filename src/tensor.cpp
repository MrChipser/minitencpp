#include "tensor.hpp"
#include <limits>

//helper function to compute the total number of elements in a tensor given its shape
//also handles overflow and negative dimension checks
size_type element_count(const std::vector<index_type>& shape) {
    size_type count = 1;
    const size_type max_elements = std::vector<value_type>().max_size();

    for (index_type dimension : shape) {
        if (dimension < 0) {
            throw std::invalid_argument("Tensor dimensions cannot be negative");
        }

        const auto dimension_size = static_cast<size_type>(dimension);
        if (dimension_size != 0 and
            (count > std::numeric_limits<size_type>::max() / dimension_size or
             count > max_elements / dimension_size)) {
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
    //strides[i] = product of shape[i+1] * shape[i+2] * ... * shape[n-1]
    for (size_type i = shape_.size(); i-- > 0;) {
        strides_[i] = stride;
        stride *= static_cast<size_type>(shape_[i]);
    }
}

size_type Tensor::flat_index(const std::vector<index_type>& indices) const {
    size_type flat_index = 0;
    for (size_type i = 0; i < indices.size(); i++) {
        if(indices[i] < 0 or indices[i] >= shape_[i]) {
            throw std::invalid_argument("Size of indices must fit inside tensor dimensions and cannot be negative");
        }
        flat_index += strides_[i]*static_cast<size_type>(indices[i]);
    }

    return  flat_index;
}

value_type& Tensor::operator()(const std::vector<index_type>& indices) {
    if (indices.size()  != shape_.size()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    return  data_[flat_index(indices)];
}

const value_type& Tensor::operator()(const std::vector<index_type>& indices) const {
    if (indices.size()  != shape_.size()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    return  data_[flat_index(indices)];
}

const std::vector<value_type>& Tensor::data() const {
    return data_;
}

const std::vector<index_type>& Tensor::shape() const {
    return shape_;
}

const std::vector<size_type>& Tensor::strides() const {
    return strides_;
}

size_type Tensor::numel() const {
    return data_.size();
}

size_type Tensor::size() const {
    return data_.size();
}

size_type Tensor::rank() const {
    return shape_.size();
}

size_type Tensor::size(size_type axis) const {
    if (axis >= shape_.size()) {
        throw std::out_of_range("Tensor axis is out of bounds");
    }

    return static_cast<size_type>(shape_[axis]);
}