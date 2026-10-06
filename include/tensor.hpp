#include <vector>
#include <initializer_list>
#include <stdexcept>

class Tensor {
    private:
        std::vector<int> shape_;
        std::vector<float> data_;
        std::vector<int> strides_;

        void  compute_strides();
    
    public:
        Tensor(const std::vector<int>& shape);
        Tensor(std::initializer_list<int> shape); //Tensor({1,2})

        float& operator()(const std::vector<int>& indices);
        const float operator()(const std::vector<int>& indices) const;

        const std::vector<float>& data() const;
        const std::vector<int>& shape() const;
        const std::vector<int>& strides() const;
};