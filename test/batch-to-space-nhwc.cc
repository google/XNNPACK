// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "batch-to-space-operator-tester.h"

#include <xnnpack/config.h>

#include <gtest/gtest.h>


namespace xnnpack {
namespace {

template<class T>
class BatchToSpaceTest: public testing::Test {
 public:
  TestConfiguration<T>& Configure() { return test_configuration_; }

  Tensor<T> BuildExpectedResult() {
    Tensor<T> ref;
    ComputeBatchToSpaceReferenceData(test_configuration_, ref);
    return ref;
  }

  std::vector<T> Run() {
    std::vector<T> output;
    RunBatchToSpace(test_configuration_, output);
    return output;
  }

  static size_t TileSize() {
    static const struct xnn_transpose_config* const xnn_transpose_conf = xnn_init_transpose_config();
    return xnn_transpose_conf->xx.tile_size;
  }

  static size_t TileSizePow(size_t exp) {
    return exp == 0 ? 1 : TileSize() * TileSizePow(exp-1);
  }

  template<class Container1, class Container2 = std::vector<int>>
  static void ExpectThatElementsAre(const Container1& a, const Container2& b) {
    for(size_t i = 0; i < a.size(); ++i) {
      EXPECT_EQ(a[i], b[i]) << "index: " << i;
    }
  }

 private:
  TestConfiguration<T> test_configuration_;
};

using TestTypes = ::testing::Types<int32_t, int16_t, int8_t>;
TYPED_TEST_SUITE(BatchToSpaceTest, TestTypes);

TYPED_TEST(BatchToSpaceTest, ReferenceBuilderTwoElements) {
  this->Configure()
      .input_shape({2, 1, 1, 1})
      .block_shape({2, 1})
      .Finalize();
  auto expected = this->BuildExpectedResult();
  this->ExpectThatElementsAre(expected.shape, {1, 2, 1, 1});
  this->ExpectThatElementsAre(expected.data, {0, 1});
}

TYPED_TEST(BatchToSpaceTest, ReferenceBuilderWeb1) {
  this->Configure()
      .input_shape({4, 1, 1, 1})
      .block_shape({2, 2})
      .Finalize();
  auto expected = this->BuildExpectedResult();
  this->ExpectThatElementsAre(expected.shape, {1, 2, 2, 1});
  this->ExpectThatElementsAre(expected.data, {0, 1, 2, 3});
}

TYPED_TEST(BatchToSpaceTest, ReferenceBuilderWeb2) {
  this->Configure()
      .input_shape({4, 1, 1, 3})
      .block_shape({2, 2})
      .Finalize();
  auto expected = this->BuildExpectedResult();
  this->ExpectThatElementsAre(expected.shape, {1, 2, 2, 3});
  this->ExpectThatElementsAre(expected.data, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
}

TYPED_TEST(BatchToSpaceTest, ReferenceBuilderWeb3) {
  this->Configure()
      .input_shape({4, 2, 2, 1})
      .block_shape({2, 2})
      .Finalize();
  auto expected = this->BuildExpectedResult();
  this->ExpectThatElementsAre(expected.shape, {1, 4, 4, 1});
  this->ExpectThatElementsAre(expected.data, {0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15});
}

TYPED_TEST(BatchToSpaceTest, ReferenceBuilderWeb4NoCrop) {
  this->Configure()
      .input_shape({8, 1, 3, 1})
      .block_shape({2, 2})
      .Finalize();
  auto expected = this->BuildExpectedResult();
  this->ExpectThatElementsAre(expected.shape, {2, 2, 6, 1});
  this->ExpectThatElementsAre(expected.data, {0, 6, 1, 7, 2, 8, 12, 18, 13, 19, 14, 20, 3, 9, 4, 10, 5, 11, 15, 21, 16, 22, 17, 23});
}

TYPED_TEST(BatchToSpaceTest, ReferenceBuilderWeb4) {
  this->Configure()
      .input_shape({8, 1, 3, 1})
      .block_shape({2, 2})
      .crop({0, 0, 2, 0})
      .Finalize();
  auto expected = this->BuildExpectedResult();
  this->ExpectThatElementsAre(expected.shape, {2, 2, 4, 1});
  this->ExpectThatElementsAre(expected.data, {1, 7, 2, 8, 13, 19, 14, 20, 4, 10, 5, 11, 16, 22, 17, 23});
}

TYPED_TEST(BatchToSpaceTest, ReferenceBuilderCropTopTooBig) {
  this->Configure()
      .input_shape({8, 4, 5, 3})
      .block_shape({2, 4})
      .crop({8, 0, 0, 0})
      .Finalize();
  auto expected = this->BuildExpectedResult();
  this->ExpectThatElementsAre(expected.shape, {1, 0, 20, 3});
  this->ExpectThatElementsAre(expected.data, {});
}

TYPED_TEST(BatchToSpaceTest, ReferenceBuilderCropBottomTooBig) {
  this->Configure()
      .input_shape({8, 4, 5, 3})
      .block_shape({2, 4})
      .crop({0, 8, 0, 0})
      .Finalize();
  auto expected = this->BuildExpectedResult();
  this->ExpectThatElementsAre(expected.shape, {1, 0, 20, 3});
  this->ExpectThatElementsAre(expected.data, {});
}

TYPED_TEST(BatchToSpaceTest, ReferenceBuilderCropVerticalTooBig) {
  this->Configure()
      .input_shape({8, 4, 5, 3})
      .block_shape({2, 4})
      .crop({3, 5, 0, 0})
      .Finalize();
  auto expected = this->BuildExpectedResult();
  this->ExpectThatElementsAre(expected.shape, {1, 0, 20, 3});
  this->ExpectThatElementsAre(expected.data, {});
}

TYPED_TEST(BatchToSpaceTest, ReferenceBuilderCropLeftTooBig) {
  this->Configure()
      .input_shape({8, 4, 5, 3})
      .block_shape({2, 4})
      .crop({0, 0, 20, 0})
      .Finalize();
  auto expected = this->BuildExpectedResult();
  this->ExpectThatElementsAre(expected.shape, {1, 8, 0, 3});
  this->ExpectThatElementsAre(expected.data, {});
}

TYPED_TEST(BatchToSpaceTest, ReferenceBuilderCropRightTooBig) {
  this->Configure()
      .input_shape({8, 4, 5, 3})
      .block_shape({2, 4})
      .crop({0, 0, 0, 20})
      .Finalize();
  auto expected = this->BuildExpectedResult();
  this->ExpectThatElementsAre(expected.shape, {1, 8, 0, 3});
  this->ExpectThatElementsAre(expected.data, {});
}

TYPED_TEST(BatchToSpaceTest, ReferenceBuilderCropHorizontalTooBig) {
  this->Configure()
      .input_shape({8, 4, 5, 3})
      .block_shape({2, 4})
      .crop({0, 0, 12, 8})
      .Finalize();
  auto expected = this->BuildExpectedResult();
  this->ExpectThatElementsAre(expected.shape, {1, 8, 0, 3});
  this->ExpectThatElementsAre(expected.data, {});
}

TYPED_TEST(BatchToSpaceTest, TwoElements) {
  this->Configure()
      .input_shape({2, 1, 1, 1})
      .block_shape({2, 1})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, Web1) {
  this->Configure()
      .input_shape({4, 1, 1, 1})
      .block_shape({2, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, Web2) {
  this->Configure()
      .input_shape({4, 1, 1, 3})
      .block_shape({2, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, Web3) {
  this->Configure()
      .input_shape({4, 2, 2, 1})
      .block_shape({2, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, Web4NoCrop) {
  this->Configure()
      .input_shape({8, 1, 3, 1})
      .block_shape({2, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, Web4) {
  this->Configure()
      .input_shape({8, 1, 3, 1})
      .block_shape({2, 2})
      .crop({0, 0, 2, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, CropTopTooBig) {
  this->Configure()
      .input_shape({8, 4, 5, 3})
      .block_shape({2, 4})
      .crop({8, 0, 0, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, CropBottomTooBig) {
  this->Configure()
      .input_shape({8, 4, 5, 3})
      .block_shape({2, 4})
      .crop({0, 8, 0, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, CropVerticalTooBig) {
  this->Configure()
      .input_shape({8, 4, 5, 3})
      .block_shape({2, 4})
      .crop({3, 5, 0, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, CropLeftTooBig) {
  this->Configure()
      .input_shape({8, 4, 5, 3})
      .block_shape({2, 4})
      .crop({0, 0, 20, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, CropRightTooBig) {
  this->Configure()
      .input_shape({8, 4, 5, 3})
      .block_shape({2, 4})
      .crop({0, 0, 0, 20})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, CropHorizontalTooBig) {
  this->Configure()
      .input_shape({8, 4, 5, 3})
      .block_shape({2, 4})
      .crop({0, 0, 12, 8})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, TwoElementsC2) {
  this->Configure()
      .input_shape({2, 1, 1, 2})
      .block_shape({2, 1})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, Web1C3) {
  this->Configure()
      .input_shape({4, 1, 1, 3})
      .block_shape({2, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, Web2C5) {
  this->Configure()
      .input_shape({4, 1, 1, 5})
      .block_shape({2, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, Web3C4) {
  this->Configure()
      .input_shape({4, 2, 2, 4})
      .block_shape({2, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, Web4NoCropC3) {
  this->Configure()
      .input_shape({8, 1, 3, 3})
      .block_shape({2, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, Web4C3) {
  this->Configure()
      .input_shape({8, 1, 3, 3})
      .block_shape({2, 2})
      .crop({0, 0, 2, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, OneImageOnePixel) {
  this->Configure()
      .input_shape({1, 1, 1, 1})
      .block_shape({1, 1})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, OneImageOnePixelMultipleChannels) {
  this->Configure()
      .input_shape({1, 1, 1, 4})
      .block_shape({1, 1})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, OneImagePixelRow) {
  this->Configure()
      .input_shape({1, 1, 4, 1})
      .block_shape({1, 1})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, OneImagePixelCol) {
  this->Configure()
      .input_shape({1, 4, 1, 1})
      .block_shape({1, 1})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, OneImage4x5x6) {
  this->Configure()
      .input_shape({1, 4, 5, 6})
      .block_shape({1, 1})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, FourImagesOnePixel) {
  this->Configure()
      .input_shape({4, 1, 1, 1})
      .block_shape({1, 1})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, FourImages16x16x3Identity) {
  this->Configure()
      .input_shape({4, 15, 16, 3})
      .block_shape({1, 1})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 1D4x16x16x3CropT1) {
  this->Configure()
      .input_shape({4, 15, 16, 3})
      .block_shape({1, 1})
      .crop({1, 0, 0, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 1D4x16x16x3CropT2) {
  this->Configure()
      .input_shape({4, 15, 16, 3})
      .block_shape({1, 1})
      .crop({2, 0, 0, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 1D4x16x16x3CropB1) {
  this->Configure()
      .input_shape({4, 15, 16, 3})
      .block_shape({1, 1})
      .crop({0, 1, 0, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 1D4x16x16x3CropB2) {
  this->Configure()
      .input_shape({4, 15, 16, 3})
      .block_shape({1, 1})
      .crop({0, 2, 0, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 1D4x16x16x3CropL1) {
  this->Configure()
      .input_shape({4, 15, 16, 3})
      .block_shape({1, 1})
      .crop({0, 0, 1, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 1D4x16x16x3CropL2) {
  this->Configure()
      .input_shape({4, 15, 16, 3})
      .block_shape({1, 1})
      .crop({0, 0, 2, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 1D4x16x16x3CropR1) {
  this->Configure()
      .input_shape({4, 15, 16, 3})
      .block_shape({1, 1})
      .crop({0, 0, 0, 1})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 1D4x16x16x3CropR2) {
  this->Configure()
      .input_shape({4, 15, 16, 3})
      .block_shape({1, 1})
      .crop({0, 0, 0, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 1D4x16x16x3CropT2B2) {
  this->Configure()
      .input_shape({4, 15, 16, 3})
      .block_shape({1, 1})
      .crop({2, 2, 0, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 1D4x16x16x3CropL2R2) {
  this->Configure()
      .input_shape({4, 15, 16, 3})
      .block_shape({1, 1})
      .crop({0, 0, 2, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 1D4x16x16x3CropT2B2L2R2) {
  this->Configure()
      .input_shape({4, 15, 16, 3})
      .block_shape({1, 1})
      .crop({2, 2, 2, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockHeight) {
  this->Configure()
      .input_shape({16, 6, 8, 3})
      .block_shape({8, 1})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockHeightCropT2) {
  this->Configure()
      .input_shape({16, 6, 8, 3})
      .block_shape({8, 1})
      .crop({2, 0, 0, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockHeightCropB2) {
  this->Configure()
      .input_shape({16, 6, 8, 3})
      .block_shape({8, 1})
      .crop({0, 2, 0, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockHeightCropL2) {
  this->Configure()
      .input_shape({16, 6, 8, 3})
      .block_shape({8, 1})
      .crop({0, 0, 2, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockHeightCropR2) {
  this->Configure()
      .input_shape({16, 6, 8, 3})
      .block_shape({8, 1})
      .crop({0, 0, 0, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockHeightCropT2B2) {
  this->Configure()
      .input_shape({16, 6, 8, 3})
      .block_shape({8, 1})
      .crop({2, 2, 0, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockHeightCropL2R2) {
  this->Configure()
      .input_shape({16, 6, 8, 3})
      .block_shape({8, 1})
      .crop({0, 0, 2, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockHeightCropT2B2L2R2) {
  this->Configure()
      .input_shape({16, 6, 8, 3})
      .block_shape({8, 1})
      .crop({2, 2, 2, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockHeightCropT5B4L3R2) {
  this->Configure()
      .input_shape({16, 6, 8, 3})
      .block_shape({8, 1})
      .crop({5, 4, 3, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthConstant) {
  this->Configure()
      .input_shape({16, 9, 8, 3})
      .block_shape({1, 8})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthConstantCropT2) {
  this->Configure()
      .input_shape({16, 9, 8, 3})
      .block_shape({1, 8})
      .crop({2, 0, 0, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthConstantCropB2) {
  this->Configure()
      .input_shape({16, 9, 8, 3})
      .block_shape({1, 8})
      .crop({0, 2, 0, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthConstantCropL2) {
  this->Configure()
      .input_shape({16, 9, 8, 3})
      .block_shape({1, 8})
      .crop({0, 0, 2, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthConstantCropR2) {
  this->Configure()
      .input_shape({16, 9, 8, 3})
      .block_shape({1, 8})
      .crop({0, 0, 0, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthConstantCropT2B2) {
  this->Configure()
      .input_shape({16, 9, 8, 3})
      .block_shape({1, 8})
      .crop({2, 2, 0, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthConstantCropL2R2) {
  this->Configure()
      .input_shape({16, 9, 8, 3})
      .block_shape({1, 8})
      .crop({0, 0, 2, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthConstantCropT2B2L2R2) {
  this->Configure()
      .input_shape({16, 9, 8, 3})
      .block_shape({1, 8})
      .crop({2, 2, 2, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthConstantCropT5B4L3R2) {
  this->Configure()
      .input_shape({16, 9, 8, 3})
      .block_shape({1, 8})
      .crop({5, 4, 3, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthVariable) {
  this->Configure()
      .input_shape({16, 9, 8, 10})
      .block_shape({1, 8})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthVariableCropT2) {
  this->Configure()
      .input_shape({16, 9, 8, 10})
      .block_shape({1, 8})
      .crop({2, 0, 0, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthVariableCropB2) {
  this->Configure()
      .input_shape({16, 9, 8, 10})
      .block_shape({1, 8})
      .crop({0, 2, 0, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthVariableCropL2) {
  this->Configure()
      .input_shape({16, 9, 8, 10})
      .block_shape({1, 8})
      .crop({0, 0, 2, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthVariableCropR2) {
  this->Configure()
      .input_shape({16, 9, 8, 10})
      .block_shape({1, 8})
      .crop({0, 0, 0, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthVariableCropT2B2) {
  this->Configure()
      .input_shape({16, 9, 8, 10})
      .block_shape({1, 8})
      .crop({2, 2, 0, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthVariableCropL2R2) {
  this->Configure()
      .input_shape({16, 9, 8, 10})
      .block_shape({1, 8})
      .crop({0, 0, 2, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthVariableCropT2B2L2R2) {
  this->Configure()
      .input_shape({16, 9, 8, 10})
      .block_shape({1, 8})
      .crop({2, 2, 2, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DBlockWidthVariableCropT5B4L3R2) {
  this->Configure()
      .input_shape({16, 9, 8, 10})
      .block_shape({1, 8})
      .crop({5, 4, 3, 2})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 4DTileSameLineLeftRightCropVariableKernel) {
  this->Configure()
      .input_shape({16, 4, 4, 5})
      .block_shape({2, 4})
      .crop({0, 0, 5, 9})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 4DTileSameLineLeftRightCropConstKernel) {
  this->Configure()
      .input_shape({16, 4, 4, 5})
      .block_shape({2, 4})
      .crop({0, 0, 5, 9})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 4DTileLeftCropOnlyVariableKernel) {
  this->Configure()
      .input_shape({16, 4, 4, 5})
      .block_shape({2, 4})
      .crop({0, 0, 5, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 4DTileLeftCropOnlyConstKernel) {
  this->Configure()
      .input_shape({16, 4, 4, 1})
      .block_shape({2, 4})
      .crop({0, 0, 5, 0})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 4DTileRightCropOnlyVariableKernel) {
  this->Configure()
      .input_shape({16, 4, 4, 5})
      .block_shape({2, 4})
      .crop({0, 0, 0, 9})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 4DTileRightCropOnlyConstKernel) {
  this->Configure()
      .input_shape({16, 4, 4, 1})
      .block_shape({2, 4})
      .crop({0, 0, 0, 9})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 4DTileLeftRightCropVariableKernel) {
  this->Configure()
      .input_shape({16, 4, 4, 5})
      .block_shape({2, 4})
      .crop({0, 0, 1, 1})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 4DTileLeftRightCropConstKernel) {
  this->Configure()
      .input_shape({16, 4, 4, 1})
      .block_shape({2, 4})
      .crop({0, 0, 1, 1})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 4DTileBlockWidthBiggerThanTileSize) {
  this->Configure()
      .input_shape({8 * this->TileSize(), 4, 4, 5})
      .block_shape({2, 2 * this->TileSize()})
      .crop({0, 0, 5, 9})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 4DTileBlockHeightBiggerThanTileSize) {
  this->Configure()
      .input_shape({16 * this->TileSize(), 4, 4, 5})
      .block_shape({2 * this->TileSize(), 4})
      .crop({0, 0, 5, 9})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 4DTileBlockDimsBiggerThanTileSize) {
  this->Configure()
      .input_shape({6 * this->TileSizePow(2), 4, 4, 5})
      .block_shape({2 * this->TileSize(), 3 * this->TileSize()})
      .crop({0, 0, 5, 9})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DTileHorizontalBlockFullSubtileTranspose) {
  this->Configure()
      .input_shape({this->TileSize(), 4, 3 * this->TileSize(), 3})
      .block_shape({1, this->TileSize()})
      .crop({0, 0, 1, 1})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DTileHorizontalBlockFullSubtileOverlapsTwoBatchItems) {
  this->Configure()
      .input_shape({16, 4, 4, 1})
      .block_shape({1, 4})
      .crop({0, 0, 1, 1})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

TYPED_TEST(BatchToSpaceTest, 2DTileVerticalBlockFullSubtileTranspose) {
  this->Configure()
      .input_shape({4 * this->TileSize(), 4, 4, 1})
      .block_shape({4 * this->TileSize(), 1})
      .crop({1, 1, 1, 1})
      .Finalize();
  this->ExpectThatElementsAre(this->Run(), this->BuildExpectedResult().data);
}

}  // namespace
}  // namespace xnnpack
