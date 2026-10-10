#include "tensor.hpp"
#include <gtest/gtest.h>
#include <limits>

using miniten::Tensor;

TEST(TensorTest, ShapeAndStridesCorrect) {
    Tensor tensor({4,5,6,7});
    EXPECT_EQ(tensor.shape(), (std::vector<miniten::index_type>{4, 5, 6, 7}));
    EXPECT_EQ(tensor.strides(), (std::vector<miniten::size_type>{210, 42, 7, 1}));
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
    EXPECT_THROW(Tensor({std::numeric_limits<miniten::index_type>::max(), 2}), std::overflow_error);
}

TEST(TensorTest, FillCorrect) {
    Tensor tensor({2,3});
    tensor.fill(4.5f);

    EXPECT_FLOAT_EQ(tensor({0, 0}), 4.5f);
    EXPECT_FLOAT_EQ(tensor({1, 2}), 4.5f);
}

TEST(TensorTest, ScalarMultiplicationNew) {
    Tensor tensor({5,6,7});
    tensor.fill(1.0f);
    tensor = tensor * 5.6f;

    EXPECT_FLOAT_EQ(tensor({0, 0, 0}), 5.6f);
    EXPECT_FLOAT_EQ(tensor({4, 5, 6}), 5.6f);
}

TEST(TensorTest, ScalarMultiplicationCurrent) {
    Tensor tensor({5,6,7});
    tensor.fill(2.0f);
    tensor *= 5.0f;

    EXPECT_FLOAT_EQ(tensor({0, 0, 0}), 10.0f);
    EXPECT_FLOAT_EQ(tensor({4, 5, 6}), 10.0f);
}