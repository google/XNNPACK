// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/subgraph.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "src/xnnpack/datatype.h"
#include "test/subgraph/runtime-tester.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

TEST(SUBGRAPH, hanging_nodes) {
  SubgraphTester tester(6);
  tester.AddDynamicTensorF32({1, 256, 256, 3}, 0)
      .AddStaticTensorF32({32, 3, 3, 3}, TensorType::kDense, 1)
      .AddStaticTensorF32({32}, TensorType::kDense, 2)
      .AddDynamicTensorF32({1, 128, 128, 32}, 3)
      .AddOutputTensorF32({32}, 4)
      .AddDynamicTensorF32({32}, 5)
      .AddConvolution2D(
          ConvolutionParams{
              Padding{1, 1, 1, 1},
              Kernel{3, 3},
              Subsampling{2, 2},
              Dilation{1, 1},
              /*groups=*/1,
              /*group_input_channels=*/3,
              /*group_output_channels=*/32,
          },
          0, 1, 2, 3)
      .AddGlobalAveragePooling(3, 4)
      // Add hanging node
      .AddGlobalAveragePooling(3, 5)
      .Optimize();

  // The hanging node is no longer there.
  ASSERT_EQ(tester.NumNodes(), 2);
}

TEST(SUBGRAPH, multiple_outputs_with_hanging_nodes) {
  SubgraphTester tester(4);
  tester.AddDynamicTensorF32({96}, 0)
      .AddDynamicTensorF32({32}, 1)
      .AddDynamicTensorF32({32}, 2)
      .AddOutputTensorF32({32}, 3)
      // Add split3 with 1 consumed output and two unconsumed outputs.
      .AddEvenSplit(0, 0, {1, 2, 3})
      .Optimize();

  // The node is still there.
  ASSERT_EQ(tester.NumNodes(), 1);
  // And all four values also.
  ASSERT_EQ(tester.NumValues(), 4);
  // The first two outputs are optimized away.
  ASSERT_EQ(tester.Value(1)->type, xnn_value_type_invalid);
  ASSERT_EQ(tester.Value(2)->type, xnn_value_type_invalid);
  // The last output is consumed.
  ASSERT_EQ(tester.Value(3)->type, xnn_value_type_dense_tensor);
}

TEST(SUBGRAPH, even_split3_first_two_outputs_optimized_away) {
  RuntimeTester tester(5);
  constexpr size_t size = 9;
  float inputs[size] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  tester.AddStaticTensorF32({size}, TensorType::kDense, 0, 0, inputs)
      .AddDynamicTensorF32({3}, 1)
      .AddDynamicTensorF32({3}, 2)
      .AddOutputTensorF32({3}, 3)
      // Add split3 with 1 consumed output and two unconsumed outputs.
      .AddEvenSplit(0, 0, {1, 2, 3});
  // Regression test for a crash where we could not deal with a split where the
  // 0th output is not used (and optimized away).
  auto output = tester.RunWithFusion<float>();
  xnnpack::Buffer<float> expected = {6, 7, 8};
  ASSERT_EQ(expected, output);
}

TEST(SUBGRAPH, create_subgraph_invalid_external_value_ids) {
  ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);
  xnn_subgraph_t subgraph = nullptr;
  EXPECT_EQ(xnn_create_subgraph(XNN_INVALID_VALUE_ID, /*flags=*/0, &subgraph),
            xnn_status_invalid_parameter);
  EXPECT_EQ(subgraph, nullptr);
}

TEST(SUBGRAPH, reserve_values_overflow) {
  ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_create_subgraph(/*external_value_ids=*/1, /*flags=*/0,
                                &subgraph),
            xnn_status_success);
  EXPECT_EQ(xnn_subgraph_reserve_values(subgraph, SIZE_MAX),
            xnn_status_out_of_memory);
  EXPECT_EQ(xnn_subgraph_reserve_values(
                subgraph, static_cast<size_t>(XNN_INVALID_VALUE_ID)),
            xnn_status_out_of_memory);
  ASSERT_EQ(xnn_delete_subgraph(subgraph), xnn_status_success);
}

TEST(SUBGRAPH, reserve_nodes_overflow) {
  ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_create_subgraph(/*external_value_ids=*/1, /*flags=*/0,
                                &subgraph),
            xnn_status_success);
  EXPECT_EQ(xnn_subgraph_reserve_nodes(subgraph, SIZE_MAX),
            xnn_status_out_of_memory);
  EXPECT_EQ(xnn_subgraph_reserve_nodes(
                subgraph, static_cast<size_t>(XNN_INVALID_NODE_ID)),
            xnn_status_out_of_memory);
  ASSERT_EQ(xnn_delete_subgraph(subgraph), xnn_status_success);
}

size_t ExpectedLog2SizeBits(xnn_datatype datatype) {
  switch (datatype) {
    case xnn_datatype_invalid:
      return -1;
    case xnn_datatype_qint2:
    case xnn_datatype_qcint2:
      return 1;
    case xnn_datatype_qint4:
    case xnn_datatype_qcint4:
    case xnn_datatype_qbint4:
      return 2;
    case xnn_datatype_qint8:
    case xnn_datatype_pqint8:
    case xnn_datatype_quint8:
    case xnn_datatype_qcint8:
    case xnn_datatype_qdint8:
    case xnn_datatype_qduint8:
    case xnn_datatype_qpint8:
      return 3;
    case xnn_datatype_fp16:
    case xnn_datatype_bf16:
    case xnn_datatype_pfp16:
      return 4;
    case xnn_datatype_qint32:
    case xnn_datatype_qcint32:
    case xnn_datatype_int32:
    case xnn_datatype_fp32:
    case xnn_datatype_pfp32:
      return 5;
  }
  XNN_UNREACHABLE;
  return -1;
}

const xnn_datatype kAllDatatypes[] = {
    xnn_datatype_invalid, xnn_datatype_fp32,   xnn_datatype_fp16,
    xnn_datatype_qint8,   xnn_datatype_quint8, xnn_datatype_qint32,
    xnn_datatype_qcint8,  xnn_datatype_qcint32, xnn_datatype_qcint4,
    xnn_datatype_qdint8,  xnn_datatype_qpint8, xnn_datatype_int32,
    xnn_datatype_qbint4,  xnn_datatype_pfp32,  xnn_datatype_bf16,
    xnn_datatype_qduint8, xnn_datatype_pfp16,  xnn_datatype_pqint8,
    xnn_datatype_qcint2,  xnn_datatype_qint4,  xnn_datatype_qint2,
};

TEST(Datatype, Log2SizeBits) {
  for (xnn_datatype datatype : kAllDatatypes) {
    EXPECT_EQ(xnn_datatype_log2_size_bits(datatype),
              ExpectedLog2SizeBits(datatype));
  }
}

TEST(Datatype, SizeBits) {
  for (xnn_datatype datatype : kAllDatatypes) {
    if (datatype == xnn_datatype_invalid) {
      EXPECT_EQ(xnn_datatype_size_bits(datatype), 0u);
    } else {
      EXPECT_EQ(xnn_datatype_size_bits(datatype),
                size_t{1} << ExpectedLog2SizeBits(datatype));
    }
  }
}

TEST(Datatype, SizeBytes) {
  for (xnn_datatype datatype : kAllDatatypes) {
    if (datatype == xnn_datatype_invalid) {
      EXPECT_EQ(xnn_datatype_size_bytes(datatype), 0u);
      continue;
    }
    const size_t log2_size_bits = ExpectedLog2SizeBits(datatype);
    if (log2_size_bits < 3) {
      continue;
    }
    EXPECT_EQ(xnn_datatype_log2_size_bytes(datatype), log2_size_bits - 3);
    EXPECT_EQ(xnn_datatype_size_bytes(datatype),
              size_t{1} << (log2_size_bits - 3));
  }
}

}  // namespace xnnpack
