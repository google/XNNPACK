#include "src/xnnpack/buffer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "test/replicable_random_device.h"

namespace xnnpack {

TEST(Buffer, GuardBytes) {
#if XNN_COMPILER_HAS_FEATURE(memory_sanitizer)
  GTEST_SKIP() << "We don't expect msan to catch out of bounds writes.";
#endif
  // The regex matching of the error message is very limited on some platforms,
  // so we don't bother checking the error (which will vary if running with
  // a sanitizer).
  ASSERT_DEATH_IF_SUPPORTED(
      {
        Buffer<int> buffer(10);
        static_cast<volatile int*>(buffer.data())[10] = 0;
      },
      "");

  ASSERT_DEATH_IF_SUPPORTED(
      {
        Buffer<int> buffer(10);
        static_cast<volatile int*>(buffer.data())[-1] = 0;
      },
      "");
}

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

  Tensor<int> scalar(std::vector<size_t>{});
  scalar() = 1;
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

template <typename T, typename... Args>
void TestDatatypeGeneratorFloats(float min, float max, Args... args) {
  ReplicableRandomDevice rng;

  int inf_count = 0;
  int nan_count = 0;
  int subnormal_count = 0;
  float min_val = NumericLimits<T>::infinity();
  float max_val = -NumericLimits<T>::infinity();
  constexpr int kSamples = 100000;
  DatatypeGenerator<T> gen(args...);
  for (int i = 0; i < kSamples; ++i) {
    float x = gen(rng);
    min_val = std::min(min_val, x);
    max_val = std::max(max_val, x);
    if (std::isinf(x)) ++inf_count;
    if (std::isnan(x)) ++nan_count;
    if (std::abs(x) != 0.0f &&
        std::abs(x) < static_cast<float>(NumericLimits<T>::smallest_normal())) {
      ++subnormal_count;
    }
  }
  EXPECT_LE(min_val, min);
  EXPECT_GE(max_val, max);
  // Don't allow more than 0.1% of samples to be infinity.
  EXPECT_LT(inf_count, kSamples / 1000);
  EXPECT_EQ(nan_count, 0);
  EXPECT_EQ(subnormal_count, 0);
}

template <typename T>
void TestDatatypeGeneratorFloats() {
  const float min = NumericLimits<T>::min();
  const float max = NumericLimits<T>::max();
  const float inf = NumericLimits<T>::infinity();

  TestDatatypeGeneratorFloats<T>(min * 0.5f, max * 0.5f);
  TestDatatypeGeneratorFloats<T>(min * 0.5f, max * 0.5f, min, max);
  TestDatatypeGeneratorFloats<T>(max * 0.01f, max * 0.99f, 0.0f, max);
  TestDatatypeGeneratorFloats<T>(min * 0.99f, min * 0.01f, min, 0.0f);
  TestDatatypeGeneratorFloats<T>(max * 0.01f, max * 0.99f, 0.0f, inf);
  TestDatatypeGeneratorFloats<T>(min * 0.99f, min * 0.01f, -inf, 0.0f);
  TestDatatypeGeneratorFloats<T>(-0.99f, 0.99f, -1.0f, 1.0f);
  TestDatatypeGeneratorFloats<T>(-9.9f, 9.9f, -10.0f, 10.0f);
}

TEST(DatatypeGenerator, Float) { TestDatatypeGeneratorFloats<float>(); }
TEST(DatatypeGenerator, Float16) { TestDatatypeGeneratorFloats<xnn_float16>(); }
TEST(DatatypeGenerator, BFloat16) {
  TestDatatypeGeneratorFloats<xnn_bfloat16>();
}

TEST(Pad, RepeatEdge1D) {
  Tensor<int> x({2});
  x(0) = 3;
  x(1) = 5;
  Tensor<int> padded = x.pad({2}, {3});
  ASSERT_THAT(padded.extents(), testing::ElementsAre(7));
  ASSERT_THAT(padded, testing::ElementsAre(3, 3, 3, 5, 5, 5, 5));
}

TEST(Pad, RepeatEdge2Dx) {
  Tensor<int> x({2, 2});
  x(0, 0) = 3;
  x(0, 1) = 5;
  x(1, 0) = 7;
  x(1, 1) = 9;
  Tensor<int> padded = x.pad({0, 2}, {0, 1});
  ASSERT_THAT(padded.extents(), testing::ElementsAre(2, 5));
  ASSERT_THAT(padded, testing::ElementsAre(3, 3, 3, 5, 5, 7, 7, 7, 9, 9));
}

TEST(Pad, RepeatEdge2D) {
  Tensor<int> x({2, 2});
  x(0, 0) = 3;
  x(0, 1) = 5;
  x(1, 0) = 7;
  x(1, 1) = 9;
  Tensor<int> padded = x.pad({1, 2}, {0, 1});
  ASSERT_THAT(padded.extents(), testing::ElementsAre(3, 5));
  ASSERT_THAT(padded, testing::ElementsAre(3, 3, 3, 5, 5, 3, 3, 3, 5, 5, 7, 7,
                                           7, 9, 9));
}

TEST(Pad, Constant1D) {
  Tensor<int> x({2});
  x(0) = 3;
  x(1) = 5;
  Tensor<int> padded = x.pad(-1, {2}, {3});
  ASSERT_THAT(padded.extents(), testing::ElementsAre(7));
  ASSERT_THAT(padded, testing::ElementsAre(-1, -1, 3, 5, -1, -1, -1));
}

TEST(Pad, Constant2Dx) {
  Tensor<int> x({2, 2});
  x(0, 0) = 3;
  x(0, 1) = 5;
  x(1, 0) = 7;
  x(1, 1) = 9;
  Tensor<int> padded = x.pad(-1, {0, 2}, {0, 1});
  ASSERT_THAT(padded.extents(), testing::ElementsAre(2, 5));
  ASSERT_THAT(padded, testing::ElementsAre(-1, -1, 3, 5, -1, -1, -1, 7, 9, -1));
}

TEST(Pad, Constant2D) {
  Tensor<int> x({2, 2});
  x(0, 0) = 3;
  x(0, 1) = 5;
  x(1, 0) = 7;
  x(1, 1) = 9;
  Tensor<int> padded = x.pad(-1, {1, 2}, {0, 1});
  ASSERT_THAT(padded.extents(), testing::ElementsAre(3, 5));
  ASSERT_THAT(padded, testing::ElementsAre(-1, -1, -1, -1, -1, -1, -1, 3, 5, -1,
                                           -1, -1, 7, 9, -1));
}

}  // namespace xnnpack
