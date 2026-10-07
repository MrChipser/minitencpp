#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <vector>

using index_type = std::int64_t;  // dimensions and user-facing indices
using size_type = std::size_t;   // allocation sizes and element counts
using value_type = float;         // default tensor data type

class Tensor {
    private:
        //Tensor caracteristics
        std::vector<index_type> shape_;
        std::vector<value_type> data_;
        std::vector<size_type> strides_;

        //compute the strides of the tensor based on its shape
        void compute_strides();
        //compute the flat index of the wanted data point
        size_type flat_index(const std::vector<index_type>& indices) const;
    
    public:
        //constructors, initializer_list allows to create a tensor with a list of dimensions, e.g. Tensor({2, 3, 4})
        Tensor(const std::vector<index_type>& shape);
        Tensor(std::initializer_list<index_type> shape); //Tensor({1,2})

        //overloaded operator() to access tensor elements using a vector of indices
        value_type& operator()(const std::vector<index_type>& indices);
        const value_type& operator()(const std::vector<index_type>& indices) const;

        //getters for the tensor's characteristics
        const std::vector<value_type>& data() const;
        const std::vector<index_type>& shape() const;
        const std::vector<size_type>& strides() const;
};