#include "src/xnnpack/buffer.h"

#include <algorithm>
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "test/replicable_random_device.h"

namespace xnnpack {

TEST(Tensor, Basic) {
  ReplicableRandomDevice rng;

  Tensor<uint8_t> test({3, 4, 5});
  ASSERT_THAT(test.extents(), testing::ElementsAre(3, 4, 5));
  ASSERT_THAT(test.strides(), testing::ElementsAre(20, 5, 1));
  fill_uniform_random_bits(test.data(), test.size(), rng);

  ASSERT_EQ(&test(0, 0, 0), test.base());
  ASSERT_EQ(&test(1, 0, 0), test.base() + 20);
  ASSERT_EQ(&test(0, 1, 0), test.base() + 5);
  ASSERT_EQ(&test(0, 0, 1), test.base() + 1);

  Tensor<uint8_t> transposed = test.transpose({2, 1, 0});
  ASSERT_EQ(test.base(), transposed.base());
  ASSERT_THAT(transposed.extents(), testing::ElementsAre(5, 4, 3));
  ASSERT_THAT(transposed.strides(), testing::ElementsAre(1, 5, 20));

  Tensor<uint8_t> transposed_copy = transposed.deep_copy();
  ASSERT_THAT(transposed_copy.extents(), testing::ElementsAre(5, 4, 3));
  ASSERT_THAT(transposed_copy.strides(), testing::ElementsAre(12, 3, 1));
  ASSERT_NE(transposed_copy.base(), transposed.base());
  ASSERT_EQ(transposed_copy(0, 0, 0), transposed(0, 0, 0));
  ASSERT_EQ(transposed_copy(1, 0, 0), transposed(1, 0, 0));
  ASSERT_EQ(transposed_copy(2, 1, 1), transposed(2, 1, 1));

  ASSERT_TRUE(transposed_copy.is_contiguous());
  transposed_copy = transposed_copy.slice({1, 1, 1}, {4, 3, 2});
  ASSERT_FALSE(transposed_copy.is_contiguous());

  Tensor<uint8_t> sliced = transposed_copy.deep_copy();
  ASSERT_TRUE(sliced.is_contiguous());
  ASSERT_EQ(sliced(0, 0, 0), transposed(1, 1, 1));
  ASSERT_EQ(sliced(1, 0, 0), transposed(2, 1, 1));
}

auto ElementsAreIndices(std::initializer_list<std::vector<size_t>> expected) {
  return testing::ElementsAreArray(expected);
}

TEST(EnumerateIndices, Rank0) {
  // Note this is an array of an empty array.
  ASSERT_THAT(EnumerateIndices({}), ElementsAreIndices({{}}));
}

TEST(EnumerateIndices, Rank1) {
  ASSERT_THAT(EnumerateIndices({0}), ElementsAreIndices({}));
  ASSERT_THAT(EnumerateIndices({1}), ElementsAreIndices({{0}}));
  ASSERT_THAT(EnumerateIndices({3}), ElementsAreIndices({{0}, {1}, {2}}));
}

TEST(EnumerateIndices, Rank2) {
  ASSERT_THAT(EnumerateIndices({0, 0}), ElementsAreIndices({}));
  ASSERT_THAT(EnumerateIndices({1, 0}), ElementsAreIndices({}));
  ASSERT_THAT(EnumerateIndices({0, 1}), ElementsAreIndices({}));
  ASSERT_THAT(EnumerateIndices({1, 1}), ElementsAreIndices({{0, 0}}));
  ASSERT_THAT(EnumerateIndices({1, 2}), ElementsAreIndices({{0, 0}, {0, 1}}));
  ASSERT_THAT(EnumerateIndices({2, 1}), ElementsAreIndices({{0, 0}, {1, 0}}));
  ASSERT_THAT(EnumerateIndices({2, 2}), ElementsAreIndices({
                                            {0, 0},
                                            {0, 1},
                                            {1, 0},
                                            {1, 1},
                                        }));
  ASSERT_THAT(EnumerateIndices({2, 3}), ElementsAreIndices({
                                            {0, 0},
                                            {0, 1},
                                            {0, 2},
                                            {1, 0},
                                            {1, 1},
                                            {1, 2},
                                        }));
  ASSERT_THAT(EnumerateIndices({3, 2}), ElementsAreIndices({
                                            {0, 0},
                                            {0, 1},
                                            {1, 0},
                                            {1, 1},
                                            {2, 0},
                                            {2, 1},
                                        }));
}

TEST(EnumerateIndices, Rank4) {
  std::vector<size_t> extents = {1, 2, 3, 4};
  std::vector<std::vector<size_t>> expected;
  do {
    expected.clear();
    for (size_t i = 0; i < extents[0]; i++) {
      for (size_t j = 0; j < extents[1]; j++) {
        for (size_t k = 0; k < extents[2]; k++) {
          for (size_t l = 0; l < extents[3]; l++) {
            expected.push_back({i, j, k, l});
          }
        }
      }
    }
    ASSERT_THAT(EnumerateIndices(extents), testing::ElementsAreArray(expected));
  } while (std::next_permutation(extents.begin(), extents.end()));
}

}  // namespace xnnpack
