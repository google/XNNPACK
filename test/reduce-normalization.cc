#include <gtest/gtest.h>

#include <xnnpack.h>
#include <xnnpack/normalization.h>

#include "reduce-normalization-tester.h"


TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_all) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({0, 1, 2, 3, 4})
    .expected_shape({2*3*5*7*11})
    .expected_axes({0})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis0) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({0})
    .expected_shape({2, 3*5*7*11})
    .expected_axes({0})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis1) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({1})
    .expected_shape({2, 3, 5*7*11})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis2) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({2})
    .expected_shape({2*3, 5, 7*11})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis3) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({3})
    .expected_shape({2*3*5, 7, 11})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis4) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({4})
    .expected_shape({2*3*5*7, 11})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis01) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({0, 1})
    .expected_shape({2*3, 5*7*11})
    .expected_axes({0})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis02) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({0, 2})
    .expected_shape({2, 3, 5, 7*11})
    .expected_axes({0, 2})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis03) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({0, 3})
    .expected_shape({2, 3*5, 7, 11})
    .expected_axes({0, 2})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis04) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({0, 4})
    .expected_shape({2, 3*5*7, 11})
    .expected_axes({0, 2})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis12) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({1, 2})
    .expected_shape({2, 3*5, 7*11})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis13) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({1, 3})
    .expected_shape({2, 3, 5, 7, 11})
    .expected_axes({1, 3})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis14) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({1, 4})
    .expected_shape({2, 3, 5*7, 11})
    .expected_axes({1, 3})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis23) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({2, 3})
    .expected_shape({2*3, 5*7, 11})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis24) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({2, 4})
    .expected_shape({2*3, 5, 7, 11})
    .expected_axes({1, 3})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis34) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({3, 4})
    .expected_shape({2*3*5, 7*11})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis012) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({0, 1, 2})
    .expected_shape({2*3*5, 7*11})
    .expected_axes({0})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis013) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({0, 1, 3})
    .expected_shape({2*3, 5, 7, 11})
    .expected_axes({0, 2})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis014) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({0, 1, 4})
    .expected_shape({2*3, 5*7, 11})
    .expected_axes({0, 2})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis023) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({0, 2, 3})
    .expected_shape({2, 3, 5*7, 11})
    .expected_axes({0, 2})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis024) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({0, 2, 4})
    .expected_shape({2, 3, 5, 7, 11})
    .expected_axes({0, 2, 4})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis034) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({0, 3, 4})
    .expected_shape({2, 3*5, 7*11})
    .expected_axes({0, 2})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis123) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({1, 2, 3})
    .expected_shape({2, 3*5*7, 11})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis124) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({1, 2, 4})
    .expected_shape({2, 3*5, 7, 11})
    .expected_axes({1, 3})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis134) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({1, 3, 4})
    .expected_shape({2, 3, 5, 7*11})
    .expected_axes({1, 3})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis234) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({2, 3, 4})
    .expected_shape({2*3, 5*7*11})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis0123) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({0, 1, 2, 3})
    .expected_shape({2*3*5*7, 11})
    .expected_axes({0})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis0124) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({0, 1, 2, 4})
    .expected_shape({2*3*5, 7, 11})
    .expected_axes({0, 2})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis0134) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({0, 1, 3, 4})
    .expected_shape({2*3, 5, 7*11})
    .expected_axes({0, 2})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis0234) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({0, 2, 3, 4})
    .expected_shape({2, 3, 5*7*11})
    .expected_axes({0, 2})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_axis1234) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .axes({1, 2, 3, 4})
    .expected_shape({2, 3*5*7*11})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_5D_reduce_none) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7, 11})
    .expected_shape({2*3*5*7*11})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_4D_reduce_all) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7})
    .axes({0, 1, 2, 3})
    .expected_shape({2*3*5*7})
    .expected_axes({0})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_4D_reduce_axis0) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7})
    .axes({0})
    .expected_shape({2, 3*5*7})
    .expected_axes({0})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_4D_reduce_axis1) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7})
    .axes({1})
    .expected_shape({2, 3, 5*7})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_4D_reduce_axis2) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7})
    .axes({2})
    .expected_shape({2*3, 5, 7})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_4D_reduce_axis3) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7})
    .axes({3})
    .expected_shape({2*3*5, 7})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_4D_reduce_axis01) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7})
    .axes({0, 1})
    .expected_shape({2*3, 5*7})
    .expected_axes({0})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_4D_reduce_axis02) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7})
    .axes({0, 2})
    .expected_shape({2, 3, 5, 7})
    .expected_axes({0, 2})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_4D_reduce_axis03) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7})
    .axes({0, 3})
    .expected_shape({2, 3*5, 7})
    .expected_axes({0, 2})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_4D_reduce_axis12) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7})
    .axes({1, 2})
    .expected_shape({2, 3*5, 7})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_4D_reduce_axis13) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7})
    .axes({1, 3})
    .expected_shape({2, 3, 5, 7})
    .expected_axes({1, 3})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_4D_reduce_axis23) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7})
    .axes({2, 3})
    .expected_shape({2*3, 5*7})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_4D_reduce_axis012) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7})
    .axes({0, 1, 2})
    .expected_shape({2*3*5, 7})
    .expected_axes({0})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_4D_reduce_axis013) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7})
    .axes({0, 1, 3})
    .expected_shape({2*3, 5, 7})
    .expected_axes({0, 2})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_4D_reduce_axis023) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7})
    .axes({0, 2, 3})
    .expected_shape({2, 3, 5*7})
    .expected_axes({0, 2})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_4D_reduce_axis123) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7})
    .axes({1, 2, 3})
    .expected_shape({2, 3*5*7})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_4D_reduce_none) {
  ReduceNormalizationTester()
    .shape({2, 3, 5, 7})
    .expected_shape({2*3*5*7})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_3D_reduce_all) {
  ReduceNormalizationTester()
    .shape({2, 3, 5})
    .axes({0, 1, 2})
    .expected_shape({2*3*5})
    .expected_axes({0})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_3D_reduce_axis0) {
  ReduceNormalizationTester()
    .shape({2, 3, 5})
    .axes({0})
    .expected_shape({2, 3*5})
    .expected_axes({0})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_3D_reduce_axis1) {
  ReduceNormalizationTester()
    .shape({2, 3, 5})
    .axes({1})
    .expected_shape({2, 3, 5})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_3D_reduce_axis2) {
  ReduceNormalizationTester()
    .shape({2, 3, 5})
    .axes({2})
    .expected_shape({2*3, 5})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_3D_reduce_axis01) {
  ReduceNormalizationTester()
    .shape({2, 3, 5})
    .axes({0, 1})
    .expected_shape({2*3, 5})
    .expected_axes({0})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_3D_reduce_axis02) {
  ReduceNormalizationTester()
    .shape({2, 3, 5})
    .axes({0, 2})
    .expected_shape({2, 3, 5})
    .expected_axes({0, 2})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_3D_reduce_axis12) {
  ReduceNormalizationTester()
    .shape({2, 3, 5})
    .axes({1, 2})
    .expected_shape({2, 3*5})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_3D_reduce_none) {
  ReduceNormalizationTester()
    .shape({2, 3, 5})
    .expected_shape({2*3*5})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_2D_reduce_all) {
  ReduceNormalizationTester()
    .shape({2, 3})
    .axes({0, 1})
    .expected_shape({2*3})
    .expected_axes({0})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_2D_reduce_axis0) {
  ReduceNormalizationTester()
    .shape({2, 3})
    .axes({0})
    .expected_shape({2, 3})
    .expected_axes({0})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_2D_reduce_axis1) {
  ReduceNormalizationTester()
    .shape({2, 3})
    .axes({1})
    .expected_shape({2, 3})
    .expected_axes({1})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_2D_reduce_none) {
  ReduceNormalizationTester()
    .shape({2, 3})
    .expected_shape({2*3})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_1D_reduce_all) {
  ReduceNormalizationTester()
    .shape({2})
    .axes({0})
    .expected_shape({2})
    .expected_axes({0})
    .Test();
}

TEST(REDUCE_NORMALIZATION_TEST, normalize_1D_reduce_none) {
  ReduceNormalizationTester()
    .shape({2})
    .expected_shape({2})
    .Test();
}
