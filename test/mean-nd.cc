// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "mean-operator-tester.h"

#include "xnnpack/normalization.h"

constexpr size_t kDim1 = 2;
constexpr size_t kDim2 = 3;
constexpr size_t kDim3 = 5;
constexpr size_t kDim4 = 7;
constexpr size_t kDim5 = 11;
constexpr size_t kDim6 = 13;


TEST(MEAN_ND_F16, reduce_all) {
  MeanOperatorTester()
    .input_shape({kDim1})
    .reduction_axes({0})
    .TestF16();
}

TEST(MEAN_ND_F16, reduce_first_axis) {
  MeanOperatorTester()
    .input_shape({kDim1, kDim2})
    .reduction_axes({0})
    .TestF16();
}

TEST(MEAN_ND_F16, reduce_last_axis) {
  MeanOperatorTester()
    .input_shape({kDim1, kDim2})
    .reduction_axes({1})
    .TestF16();
}

TEST(MEAN_ND_F16, reduce_2d) {
  std::vector<size_t> reduction_axes;
  for (uint32_t bm1 = 1; bm1 < (uint32_t(1) << 2); bm1++) {
    const bool reduce_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool reduce_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;

    reduction_axes.clear();
    if (reduce_dim1) {
      reduction_axes.push_back(0);
    }
    if (reduce_dim2) {
      reduction_axes.push_back(1);
    }
    MeanOperatorTester()
      .input_shape({kDim1, kDim2})
      .reduction_axes(reduction_axes)
      .TestF16();
  }
}

TEST(MEAN_ND_F16, reduce_3d) {
  std::vector<size_t> reduction_axes;
  for (uint32_t bm1 = 1; bm1 < (uint32_t(1) << 3); bm1++) {
    const bool reduce_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool reduce_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool reduce_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;

    const std::vector<size_t> input_shape{{kDim1, kDim2, kDim3}};
    reduction_axes.clear();
    if (reduce_dim1) {
      reduction_axes.push_back(0);
    }
    if (reduce_dim2) {
      reduction_axes.push_back(1);
    }
    if (reduce_dim3) {
      reduction_axes.push_back(2);
    }

    size_t num_normalized_input_dims = input_shape.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_input_shape;
    std::copy(input_shape.cbegin(), input_shape.cend(), normalized_input_shape.begin());
    size_t num_normalized_reduction_axes = reduction_axes.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_reduction_axes;
    std::copy(reduction_axes.cbegin(), reduction_axes.cend(), normalized_reduction_axes.begin());
    xnn_normalize_reduction(
      &num_normalized_reduction_axes, normalized_reduction_axes.data(),
      &num_normalized_input_dims, normalized_input_shape.data());
    if (num_normalized_reduction_axes != 1) {
      continue;  // unsupported reduction configuration, will fail if we proceed
    }

    MeanOperatorTester()
      .input_shape(input_shape)
      .reduction_axes(reduction_axes)
      .TestF16();
  }
}

TEST(MEAN_ND_F16, reduce_4d) {
  std::vector<size_t> reduction_axes;
  for (uint32_t bm1 = 1; bm1 < (uint32_t(1) << 4); bm1++) {
    const bool reduce_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool reduce_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool reduce_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool reduce_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;

    const std::vector<size_t> input_shape{{kDim1, kDim2, kDim3, kDim4}};
    reduction_axes.clear();
    if (reduce_dim1) {
      reduction_axes.push_back(0);
    }
    if (reduce_dim2) {
      reduction_axes.push_back(1);
    }
    if (reduce_dim3) {
      reduction_axes.push_back(2);
    }
    if (reduce_dim4) {
      reduction_axes.push_back(3);
    }

    size_t num_normalized_input_dims = input_shape.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_input_shape;
    std::copy(input_shape.cbegin(), input_shape.cend(), normalized_input_shape.begin());
    size_t num_normalized_reduction_axes = reduction_axes.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_reduction_axes;
    std::copy(reduction_axes.cbegin(), reduction_axes.cend(), normalized_reduction_axes.begin());
    xnn_normalize_reduction(
      &num_normalized_reduction_axes, normalized_reduction_axes.data(),
      &num_normalized_input_dims, normalized_input_shape.data());
    if (num_normalized_reduction_axes != 1) {
      continue;  // unsupported reduction configuration, will fail if we proceed
    }

    MeanOperatorTester()
      .input_shape(input_shape)
      .reduction_axes(reduction_axes)
      .TestF16();
  }
}

TEST(MEAN_ND_F16, reduce_5d) {
  std::vector<size_t> reduction_axes;
  for (uint32_t bm1 = 1; bm1 < (uint32_t(1) << 5); bm1++) {
    const bool reduce_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool reduce_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool reduce_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool reduce_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const bool reduce_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;

    const std::vector<size_t> input_shape{{kDim1, kDim2, kDim3, kDim4, kDim5}};
    reduction_axes.clear();
    if (reduce_dim1) {
      reduction_axes.push_back(0);
    }
    if (reduce_dim2) {
      reduction_axes.push_back(1);
    }
    if (reduce_dim3) {
      reduction_axes.push_back(2);
    }
    if (reduce_dim4) {
      reduction_axes.push_back(3);
    }
    if (reduce_dim5) {
      reduction_axes.push_back(4);
    }

    size_t num_normalized_input_dims = input_shape.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_input_shape;
    std::copy(input_shape.cbegin(), input_shape.cend(), normalized_input_shape.begin());
    size_t num_normalized_reduction_axes = reduction_axes.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_reduction_axes;
    std::copy(reduction_axes.cbegin(), reduction_axes.cend(), normalized_reduction_axes.begin());
    xnn_normalize_reduction(
      &num_normalized_reduction_axes, normalized_reduction_axes.data(),
      &num_normalized_input_dims, normalized_input_shape.data());
    if (num_normalized_reduction_axes != 1) {
      continue;  // unsupported reduction configuration, will fail if we proceed
    }

    MeanOperatorTester()
      .input_shape(input_shape)
      .reduction_axes(reduction_axes)
      .TestF16();
  }
}

TEST(MEAN_ND_F16, reduce_6d) {
  std::vector<size_t> reduction_axes;
  for (uint32_t bm1 = 1; bm1 < (uint32_t(1) << 6); bm1++) {
    const bool reduce_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool reduce_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool reduce_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool reduce_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const bool reduce_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
    const bool reduce_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;

    const std::vector<size_t> input_shape{{kDim1, kDim2, kDim3, kDim4, kDim5, kDim6}};
    reduction_axes.clear();
    if (reduce_dim1) {
      reduction_axes.push_back(0);
    }
    if (reduce_dim2) {
      reduction_axes.push_back(1);
    }
    if (reduce_dim3) {
      reduction_axes.push_back(2);
    }
    if (reduce_dim4) {
      reduction_axes.push_back(3);
    }
    if (reduce_dim5) {
      reduction_axes.push_back(4);
    }
    if (reduce_dim6) {
      reduction_axes.push_back(5);
    }

    size_t num_normalized_input_dims = input_shape.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_input_shape;
    std::copy(input_shape.cbegin(), input_shape.cend(), normalized_input_shape.begin());
    size_t num_normalized_reduction_axes = reduction_axes.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_reduction_axes;
    std::copy(reduction_axes.cbegin(), reduction_axes.cend(), normalized_reduction_axes.begin());
    xnn_normalize_reduction(
      &num_normalized_reduction_axes, normalized_reduction_axes.data(),
      &num_normalized_input_dims, normalized_input_shape.data());
    if (num_normalized_reduction_axes != 1) {
      continue;  // unsupported reduction configuration, will fail if we proceed
    }

    MeanOperatorTester()
      .input_shape(input_shape)
      .reduction_axes(reduction_axes)
      .TestF16();
  }
}

TEST(MEAN_ND_F16, reduce_6d_multithreaded) {
  std::vector<size_t> reduction_axes;
  for (uint32_t bm1 = 1; bm1 < (uint32_t(1) << 6); bm1++) {
    const bool reduce_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool reduce_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool reduce_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool reduce_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const bool reduce_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
    const bool reduce_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;

    const std::vector<size_t> input_shape{{kDim1, kDim2, kDim3, kDim4, kDim5, kDim6}};
    reduction_axes.clear();
    if (reduce_dim1) {
      reduction_axes.push_back(0);
    }
    if (reduce_dim2) {
      reduction_axes.push_back(1);
    }
    if (reduce_dim3) {
      reduction_axes.push_back(2);
    }
    if (reduce_dim4) {
      reduction_axes.push_back(3);
    }
    if (reduce_dim5) {
      reduction_axes.push_back(4);
    }
    if (reduce_dim6) {
      reduction_axes.push_back(5);
    }

    size_t num_normalized_input_dims = input_shape.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_input_shape;
    std::copy(input_shape.cbegin(), input_shape.cend(), normalized_input_shape.begin());
    size_t num_normalized_reduction_axes = reduction_axes.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_reduction_axes;
    std::copy(reduction_axes.cbegin(), reduction_axes.cend(), normalized_reduction_axes.begin());
    xnn_normalize_reduction(
      &num_normalized_reduction_axes, normalized_reduction_axes.data(),
      &num_normalized_input_dims, normalized_input_shape.data());
    if (num_normalized_reduction_axes != 1) {
      continue;  // unsupported reduction configuration, will fail if we proceed
    }

    MeanOperatorTester()
      .input_shape(input_shape)
      .reduction_axes(reduction_axes)
      .multithreaded(true)
      .TestF16();
  }
}

TEST(MEAN_ND_F32, reduce_all) {
  MeanOperatorTester()
    .input_shape({kDim1})
    .reduction_axes({0})
    .TestF32();
}

TEST(MEAN_ND_F32, reduce_first_axis) {
  MeanOperatorTester()
    .input_shape({kDim1, kDim2})
    .reduction_axes({0})
    .TestF32();
}

TEST(MEAN_ND_F32, reduce_last_axis) {
  MeanOperatorTester()
    .input_shape({kDim1, kDim2})
    .reduction_axes({1})
    .TestF32();
}

TEST(MEAN_ND_F32, reduce_2d) {
  std::vector<size_t> reduction_axes;
  for (uint32_t bm1 = 1; bm1 < (uint32_t(1) << 2); bm1++) {
    const bool reduce_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool reduce_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;

    reduction_axes.clear();
    if (reduce_dim1) {
      reduction_axes.push_back(0);
    }
    if (reduce_dim2) {
      reduction_axes.push_back(1);
    }
    MeanOperatorTester()
      .input_shape({kDim1, kDim2})
      .reduction_axes(reduction_axes)
      .TestF32();
  }
}

TEST(MEAN_ND_F32, reduce_3d) {
  std::vector<size_t> reduction_axes;
  for (uint32_t bm1 = 1; bm1 < (uint32_t(1) << 3); bm1++) {
    const bool reduce_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool reduce_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool reduce_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;

    const std::vector<size_t> input_shape{{kDim1, kDim2, kDim3}};
    reduction_axes.clear();
    if (reduce_dim1) {
      reduction_axes.push_back(0);
    }
    if (reduce_dim2) {
      reduction_axes.push_back(1);
    }
    if (reduce_dim3) {
      reduction_axes.push_back(2);
    }

    size_t num_normalized_input_dims = input_shape.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_input_shape;
    std::copy(input_shape.cbegin(), input_shape.cend(), normalized_input_shape.begin());
    size_t num_normalized_reduction_axes = reduction_axes.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_reduction_axes;
    std::copy(reduction_axes.cbegin(), reduction_axes.cend(), normalized_reduction_axes.begin());
    xnn_normalize_reduction(
      &num_normalized_reduction_axes, normalized_reduction_axes.data(),
      &num_normalized_input_dims, normalized_input_shape.data());
    if (num_normalized_reduction_axes != 1) {
      continue;  // unsupported reduction configuration, will fail if we proceed
    }

    MeanOperatorTester()
      .input_shape(input_shape)
      .reduction_axes(reduction_axes)
      .TestF32();
  }
}

TEST(MEAN_ND_F32, reduce_4d) {
  std::vector<size_t> reduction_axes;
  for (uint32_t bm1 = 1; bm1 < (uint32_t(1) << 4); bm1++) {
    const bool reduce_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool reduce_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool reduce_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool reduce_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;

    const std::vector<size_t> input_shape{{kDim1, kDim2, kDim3, kDim4}};
    reduction_axes.clear();
    if (reduce_dim1) {
      reduction_axes.push_back(0);
    }
    if (reduce_dim2) {
      reduction_axes.push_back(1);
    }
    if (reduce_dim3) {
      reduction_axes.push_back(2);
    }
    if (reduce_dim4) {
      reduction_axes.push_back(3);
    }

    size_t num_normalized_input_dims = input_shape.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_input_shape;
    std::copy(input_shape.cbegin(), input_shape.cend(), normalized_input_shape.begin());
    size_t num_normalized_reduction_axes = reduction_axes.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_reduction_axes;
    std::copy(reduction_axes.cbegin(), reduction_axes.cend(), normalized_reduction_axes.begin());
    xnn_normalize_reduction(
      &num_normalized_reduction_axes, normalized_reduction_axes.data(),
      &num_normalized_input_dims, normalized_input_shape.data());
    if (num_normalized_reduction_axes != 1) {
      continue;  // unsupported reduction configuration, will fail if we proceed
    }

    MeanOperatorTester()
      .input_shape(input_shape)
      .reduction_axes(reduction_axes)
      .TestF32();
  }
}

TEST(MEAN_ND_F32, reduce_5d) {
  std::vector<size_t> reduction_axes;
  for (uint32_t bm1 = 1; bm1 < (uint32_t(1) << 5); bm1++) {
    const bool reduce_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool reduce_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool reduce_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool reduce_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const bool reduce_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;

    const std::vector<size_t> input_shape{{kDim1, kDim2, kDim3, kDim4, kDim5}};
    reduction_axes.clear();
    if (reduce_dim1) {
      reduction_axes.push_back(0);
    }
    if (reduce_dim2) {
      reduction_axes.push_back(1);
    }
    if (reduce_dim3) {
      reduction_axes.push_back(2);
    }
    if (reduce_dim4) {
      reduction_axes.push_back(3);
    }
    if (reduce_dim5) {
      reduction_axes.push_back(4);
    }

    size_t num_normalized_input_dims = input_shape.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_input_shape;
    std::copy(input_shape.cbegin(), input_shape.cend(), normalized_input_shape.begin());
    size_t num_normalized_reduction_axes = reduction_axes.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_reduction_axes;
    std::copy(reduction_axes.cbegin(), reduction_axes.cend(), normalized_reduction_axes.begin());
    xnn_normalize_reduction(
      &num_normalized_reduction_axes, normalized_reduction_axes.data(),
      &num_normalized_input_dims, normalized_input_shape.data());
    if (num_normalized_reduction_axes != 1) {
      continue;  // unsupported reduction configuration, will fail if we proceed
    }

    MeanOperatorTester()
      .input_shape(input_shape)
      .reduction_axes(reduction_axes)
      .TestF32();
  }
}

TEST(MEAN_ND_F32, reduce_6d) {
  std::vector<size_t> reduction_axes;
  for (uint32_t bm1 = 1; bm1 < (uint32_t(1) << 6); bm1++) {
    const bool reduce_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool reduce_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool reduce_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool reduce_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const bool reduce_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
    const bool reduce_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;

    const std::vector<size_t> input_shape{{kDim1, kDim2, kDim3, kDim4, kDim5, kDim6}};
    reduction_axes.clear();
    if (reduce_dim1) {
      reduction_axes.push_back(0);
    }
    if (reduce_dim2) {
      reduction_axes.push_back(1);
    }
    if (reduce_dim3) {
      reduction_axes.push_back(2);
    }
    if (reduce_dim4) {
      reduction_axes.push_back(3);
    }
    if (reduce_dim5) {
      reduction_axes.push_back(4);
    }
    if (reduce_dim6) {
      reduction_axes.push_back(5);
    }

    size_t num_normalized_input_dims = input_shape.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_input_shape;
    std::copy(input_shape.cbegin(), input_shape.cend(), normalized_input_shape.begin());
    size_t num_normalized_reduction_axes = reduction_axes.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_reduction_axes;
    std::copy(reduction_axes.cbegin(), reduction_axes.cend(), normalized_reduction_axes.begin());
    xnn_normalize_reduction(
      &num_normalized_reduction_axes, normalized_reduction_axes.data(),
      &num_normalized_input_dims, normalized_input_shape.data());
    if (num_normalized_reduction_axes != 1) {
      continue;  // unsupported reduction configuration, will fail if we proceed
    }

    MeanOperatorTester()
      .input_shape(input_shape)
      .reduction_axes(reduction_axes)
      .TestF32();
  }
}

TEST(MEAN_ND_F32, reduce_6d_multithreaded) {
  std::vector<size_t> reduction_axes;
  for (uint32_t bm1 = 1; bm1 < (uint32_t(1) << 6); bm1++) {
    const bool reduce_dim1 = (bm1 & (uint32_t(1) << 0)) != 0;
    const bool reduce_dim2 = (bm1 & (uint32_t(1) << 1)) != 0;
    const bool reduce_dim3 = (bm1 & (uint32_t(1) << 2)) != 0;
    const bool reduce_dim4 = (bm1 & (uint32_t(1) << 3)) != 0;
    const bool reduce_dim5 = (bm1 & (uint32_t(1) << 4)) != 0;
    const bool reduce_dim6 = (bm1 & (uint32_t(1) << 5)) != 0;

    const std::vector<size_t> input_shape{{kDim1, kDim2, kDim3, kDim4, kDim5, kDim6}};
    reduction_axes.clear();
    if (reduce_dim1) {
      reduction_axes.push_back(0);
    }
    if (reduce_dim2) {
      reduction_axes.push_back(1);
    }
    if (reduce_dim3) {
      reduction_axes.push_back(2);
    }
    if (reduce_dim4) {
      reduction_axes.push_back(3);
    }
    if (reduce_dim5) {
      reduction_axes.push_back(4);
    }
    if (reduce_dim6) {
      reduction_axes.push_back(5);
    }

    size_t num_normalized_input_dims = input_shape.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_input_shape;
    std::copy(input_shape.cbegin(), input_shape.cend(), normalized_input_shape.begin());
    size_t num_normalized_reduction_axes = reduction_axes.size();
    std::array<size_t, XNN_MAX_TENSOR_DIMS> normalized_reduction_axes;
    std::copy(reduction_axes.cbegin(), reduction_axes.cend(), normalized_reduction_axes.begin());
    xnn_normalize_reduction(
      &num_normalized_reduction_axes, normalized_reduction_axes.data(),
      &num_normalized_input_dims, normalized_input_shape.data());
    if (num_normalized_reduction_axes != 1) {
      continue;  // unsupported reduction configuration, will fail if we proceed
    }

    MeanOperatorTester()
      .input_shape(input_shape)
      .reduction_axes(reduction_axes)
      .multithreaded(true)
      .TestF32();
  }
}