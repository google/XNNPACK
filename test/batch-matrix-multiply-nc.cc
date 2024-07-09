// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "batch-matrix-multiply-operator-tester.h"

struct BatchMatMulTesterParams {
  std::string name;
  std::vector<size_t> batch_dims_a;
  std::vector<size_t> batch_dims_b;
  size_t m = 17;
  size_t k = 23;
  size_t n = 19;
  bool transpose_b = false;
  size_t iterations = 3;
  enum xnn_status expected_status_reshape = xnn_status_success;
};

template <typename T>
std::ostream& PrintVector(std::ostream& os, const std::vector<T>& v,
                          const char* separator = ", ") {
  if (!v.empty()) {
    std::copy(v.begin(), v.end() - 1, std::ostream_iterator<T>(os, separator));
    os << v.back();
  }
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const BatchMatMulTesterParams& params) {
  os << "{batch_dims_a=[";
  PrintVector(os, params.batch_dims_a);
  os << "], batch_dims_b=[";
  PrintVector(os, params.batch_dims_b);
  os << "], m=" << params.m << ", k=" << params.k << ", n=" << params.n
     << ", transpose_b=" << params.transpose_b
     << ", iterations=" << params.iterations
     << ", expected_status_reshape=" << params.expected_status_reshape << "}";
  return os;
}

// Creates all possible combinations of masked/unmasked batch dimensions.
std::vector<BatchMatMulTesterParams> CreateBatchTestParams() {
  std::vector<BatchMatMulTesterParams> params;

  // Iterate over all combinations of batch dimensions.
  for (int num_batch_dims = 0; num_batch_dims < XNN_MAX_TENSOR_DIMS - 1;
       ++num_batch_dims) {
    // Loop over the bitwise masks for the dimensions of the input `A`.
    for (uint32_t mask_dims_a = 0; mask_dims_a < (1U << num_batch_dims);
         ++mask_dims_a) {
      // Loop over the bitwise masks for the dimensions of the input `B`.
      for (uint32_t mask_dims_b = 0; mask_dims_b < (1U << num_batch_dims);
           ++mask_dims_b) {
        // Skip parameterizations that would have overlapping broadcast masks.
        if (mask_dims_a & mask_dims_b) {
          continue;
        }

        // Switch transpose flag.
        for (int transpose_b = 0; transpose_b <= 1; ++transpose_b) {
          // Populate the batch dimensions.
          std::vector<size_t> batch_dims_a(num_batch_dims);
          std::vector<size_t> batch_dims_b(num_batch_dims);
          std::iota(batch_dims_a.begin(), batch_dims_a.end(), 3);
          std::iota(batch_dims_b.begin(), batch_dims_b.end(), 3);

          // Mask out the broadcast dimensions in both inputs.
          for (uint32_t i = 0; i < num_batch_dims; ++i) {
            if (mask_dims_a & (1U << i)) {
              batch_dims_a[i] = 1;
            }
            if (mask_dims_b & (1U << i)) {
              batch_dims_b[i] = 1;
            }
          }

          // Create the test name.
          std::ostringstream oss;
          oss << "batch_matmul_A_";
          PrintVector(oss, batch_dims_a, "_");
          oss << "_B_";
          PrintVector(oss, batch_dims_b, "_");
          if (transpose_b) {
            oss << "_transpose_b";
          }

          // Add the test parameters.
          BatchMatMulTesterParams p;
          p.name = oss.str();
          p.batch_dims_a = batch_dims_a;
          p.batch_dims_b = batch_dims_b;
          p.transpose_b = transpose_b > 0;
          params.push_back(std::move(p));
        }
      }
    }
  }

  return params;
}

// Creates combinations of different matrix sizes, with or without
// transposition.
std::vector<BatchMatMulTesterParams> CreateRegularTestParams() {
  std::vector<BatchMatMulTesterParams> params;

  // Batch size.
  for (const auto& batch : {std::make_pair(1UL, "unit_batch"),
                            std::make_pair(5UL, "small_batch")}) {
    // Switch transpose flag.
    for (int transpose_b = 0; transpose_b <= 1; ++transpose_b) {
      // Create the test name.
      std::ostringstream oss;
      oss << batch.second << "_batch";
      if (transpose_b) {
        oss << "_transpose_b";
      }

      // Add the test parameters.
      BatchMatMulTesterParams p1;
      p1.name = std::string(batch.second) +
                                (transpose_b ? "_transpose_b" : "");
      p1.batch_dims_a = {batch.first};
      p1.batch_dims_b = {batch.first};
      p1.transpose_b = transpose_b > 0;
      params.push_back(std::move(p1));
      BatchMatMulTesterParams p2;
      p2.name = std::string(batch.second) + "_bigger_matrices" +
                                (transpose_b ? "_transpose_b" : "");
      p2.batch_dims_a = {batch.first};
      p2.batch_dims_b = {batch.first};
      p2.m = 37;
      p2.k = 101;
      p2.n = 71;
      p2.transpose_b = transpose_b > 0;
      params.push_back(std::move(p2));
    }
  }

  return params;
}

using BatchMatMulTest = testing::TestWithParam<BatchMatMulTesterParams>;

TEST_P(BatchMatMulTest, TestF32) {
  const BatchMatMulTesterParams& params = GetParam();
  BatchMatMulOperatorTester()
      .batch_dims_a(params.batch_dims_a)
      .batch_dims_b(params.batch_dims_b)
      .m(params.m)
      .k(params.k)
      .n(params.n)
      .transpose_b(params.transpose_b)
      .iterations(params.iterations)
      .expected_status_reshape(params.expected_status_reshape)
      .TestF32();
}

TEST_P(BatchMatMulTest, TestF16) {
  const BatchMatMulTesterParams& params = GetParam();
  BatchMatMulOperatorTester()
      .batch_dims_a(params.batch_dims_a)
      .batch_dims_b(params.batch_dims_b)
      .m(params.m)
      .k(params.k)
      .n(params.n)
      .transpose_b(params.transpose_b)
      .iterations(params.iterations)
      .expected_status_reshape(params.expected_status_reshape)
      .TestF16();
}

TEST_P(BatchMatMulTest, TestQD8F32QC8W) {
  const BatchMatMulTesterParams& params = GetParam();
  BatchMatMulOperatorTester()
      .batch_dims_a(params.batch_dims_a)
      .batch_dims_b(params.batch_dims_b)
      .m(params.m)
      .k(params.k)
      .n(params.n)
      .transpose_b(params.transpose_b)
      .iterations(params.iterations)
      .expected_status_reshape(params.expected_status_reshape)
      .TestQD8F32QC8W();
}

// Create tests for different batch sizes with different amounts of
// broadcasting, with and without transposition.
INSTANTIATE_TEST_SUITE_P(
    Batches, BatchMatMulTest, testing::ValuesIn(CreateBatchTestParams()),
    [](const testing::TestParamInfo<BatchMatMulTest::ParamType>& info) {
      return info.param.name;
    });

// Create tests for different matrix sizes with and without transposition.
INSTANTIATE_TEST_SUITE_P(
    Sizes, BatchMatMulTest, testing::ValuesIn(CreateRegularTestParams()),
    [](const testing::TestParamInfo<BatchMatMulTest::ParamType>& info) {
      return info.param.name;
    });

// Additional failure tests for incompatible batch dimensions.
TEST(BatchMatMulTest, bad_broadcast_a_4_b_6_fails) {
  BatchMatMulOperatorTester()
      .batch_dims_a({4})
      .batch_dims_b({6})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .expected_status_reshape(xnn_status_invalid_parameter)
      .TestF32();
}

TEST(BatchMatMulTest, bad_broadcast_a_1_6_3_b_6_1_5_fails) {
  BatchMatMulOperatorTester()
      .batch_dims_a({1, 6, 3})
      .batch_dims_b({6, 1, 5})
      .m(17)
      .k(23)
      .n(19)
      .iterations(3)
      .expected_status_reshape(xnn_status_invalid_parameter)
      .TestF32();
}
