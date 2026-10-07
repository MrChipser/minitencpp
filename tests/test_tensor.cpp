#include "tensor.hpp"
#include <gtest/gtest.h>
#include <limits>

TEST(TensorTest, ShapeAndStridesCorrect) {
    Tensor tensor({4,5,6,7});
    EXPECT_EQ(tensor.shape(), (std::vector<index_type>{4, 5, 6, 7}));
    EXPECT_EQ(tensor.strides(), (std::vector<size_type>{210, 42, 7, 1}));
    EXPECT_EQ(tensor.data().size(), 840);
}

TEST(TensorTest, IndexingUsesRowMajorOrder) {
    Tensor tensor({2, 3});
    tensor({0, 0}) = 1.0f;
    tensor({1, 2}) = 6.0f;

    EXPECT_FLOAT_EQ(tensor({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(tensor({1, 2}), 6.0f);
}

TEST(TensorTest, ConstAccessCorrect) {
    Tensor mutable_tensor({4,4});
    mutable_tensor({1,2}) = 7.5f;

    const Tensor& tensor = mutable_tensor;

    EXPECT_FLOAT_EQ(tensor({1, 2}), 7.5f);


}

TEST(TensorTest, RejectsInvalidIndexes) {
    Tensor tensor({2, 3});

    EXPECT_THROW(tensor({1}), std::invalid_argument);
    EXPECT_THROW(tensor({-1, 0}), std::invalid_argument);
    EXPECT_THROW(tensor({2, 0}), std::invalid_argument);
    EXPECT_THROW(tensor({0, 3}), std::invalid_argument);
}

TEST(TensorTest, RejectsNegativeDimensions) {
    EXPECT_THROW(Tensor({2, -1}), std::invalid_argument);
}

TEST(TensorTest, RejectsShapeTooLarge) {
    EXPECT_THROW(Tensor({std::numeric_limits<index_type>::max(), 2}), std::overflow_error);
}