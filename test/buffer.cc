#include <cstdint>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "replicable_random_device.h"
#include "xnnpack/buffer.h"

TEST(Tensor, Basic) {
  xnnpack::ReplicableRandomDevice rng;

  xnnpack::Tensor<uint8_t> test({3, 4, 5});
  ASSERT_THAT(test.extents(), testing::ElementsAre(3, 4, 5));
  ASSERT_THAT(test.strides(), testing::ElementsAre(20, 5, 1));
  xnnpack::fill_uniform_random_bits(test.data(), test.size(), rng);

  ASSERT_EQ(&test(0, 0, 0), test.base());
  ASSERT_EQ(&test(1, 0, 0), test.base() + 20);
  ASSERT_EQ(&test(0, 1, 0), test.base() + 5);
  ASSERT_EQ(&test(0, 0, 1), test.base() + 1);

  xnnpack::Tensor<uint8_t> transposed = test.transpose({2, 1, 0});
  ASSERT_EQ(test.base(), transposed.base());
  ASSERT_THAT(transposed.extents(), testing::ElementsAre(5, 4, 3));
  ASSERT_THAT(transposed.strides(), testing::ElementsAre(1, 5, 20));

  xnnpack::Tensor<uint8_t> transposed_copy = transposed.deep_copy();
  ASSERT_THAT(transposed_copy.extents(), testing::ElementsAre(5, 4, 3));
  ASSERT_THAT(transposed_copy.strides(), testing::ElementsAre(12, 3, 1));
  ASSERT_NE(transposed_copy.base(), transposed.base());
  ASSERT_EQ(transposed_copy(0, 0, 0), transposed(0, 0, 0));
  ASSERT_EQ(transposed_copy(1, 0, 0), transposed(1, 0, 0));
  ASSERT_EQ(transposed_copy(2, 1, 1), transposed(2, 1, 1));

  ASSERT_TRUE(transposed_copy.is_contiguous());
  transposed_copy = transposed_copy.slice({1, 1, 1}, {4, 3, 2});
  ASSERT_FALSE(transposed_copy.is_contiguous());

  xnnpack::Tensor<uint8_t> sliced = transposed_copy.deep_copy();
  ASSERT_TRUE(sliced.is_contiguous());
  ASSERT_EQ(sliced(0, 0, 0), transposed(1, 1, 1));
  ASSERT_EQ(sliced(1, 0, 0), transposed(2, 1, 1));
}
