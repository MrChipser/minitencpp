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
        std::vector<index_type> shape_;
        std::vector<value_type> data_;
        std::vector<size_type> strides_;

        void compute_strides();
    
    public:
        Tensor(const std::vector<index_type>& shape);
        Tensor(std::initializer_list<index_type> shape); //Tensor({1,2})



        value_type& operator()(const std::vector<index_type>& indices);
        const value_type operator()(const std::vector<index_type>& indices) const;

        const std::vector<value_type>& data() const;
        const std::vector<index_type>& shape() const;
        const std::vector<size_type>& strides() const;
};