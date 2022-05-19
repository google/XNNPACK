#include <gtest/gtest.h>

#include <xnnpack.h>
#include <xnnpack/normalization.h>

#include "transpose-normalization-tester.h"

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_adjacent_1_dims) {
    TransposeNormalizationTester()
      .num_dims(4)
      .element_size(4)
      .perm({0,1,3,2})
      .shape({1,1,60,2400})
      .expected_normalized_shape({60,2400})
      .expected_normalized_perm({1,0})
      .expected_normalized_dims(2)
      .expected_element_size(4)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_1D) {
    TransposeNormalizationTester()
      .num_dims(1)
      .element_size(4)
      .perm({0})
      .shape({37})
      .expected_normalized_shape({1})
      .expected_normalized_perm({0})
      .expected_normalized_dims(1)
      .expected_element_size(37*4)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_2D_flatten_1D) {
    TransposeNormalizationTester()
      .num_dims(2)
      .element_size(4)
      .perm({0,1})
      .shape({37,19})
      .expected_normalized_shape({1})
      .expected_normalized_perm({0})
      .expected_normalized_dims(1)
      .expected_element_size(37*19*4)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_2D_flatten_2D) {
    TransposeNormalizationTester()
      .num_dims(2)
      .element_size(4)
      .perm({1,0})
      .shape({23,17})
      .expected_normalized_shape({23,17})
      .expected_normalized_perm({1,0})
      .expected_normalized_dims(2)
      .expected_element_size(4)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_3D_redundant_dim) {
    TransposeNormalizationTester()
      .num_dims(3)
      .element_size(4)
      .perm({0,2,1})
      .shape({2,1,3})
      .expected_normalized_shape({1})
      .expected_normalized_perm({0})
      .expected_normalized_dims(1)
      .expected_element_size(24)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_3D_all_ones) {
    TransposeNormalizationTester()
      .num_dims(3)
      .element_size(4)
      .perm({2,1,0})
      .shape({1,1,1})
      .expected_normalized_shape({1})
      .expected_normalized_perm({0})
      .expected_normalized_dims(1)
      .expected_element_size(4)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_3D_flatten_1D) {
    TransposeNormalizationTester()
      .num_dims(3)
      .element_size(4)
      .perm({0,1,2})
      .shape({101,13,7})
      .expected_normalized_shape({1})
      .expected_normalized_perm({0})
      .expected_normalized_dims(1)
      .expected_element_size(101*13*7*4)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_3D_flatten_2D) {
    TransposeNormalizationTester()
      .num_dims(3)
      .element_size(4)
      .perm({2,0,1})
      .shape({101,13,7})
      .expected_normalized_shape({101*13,7})
      .expected_normalized_perm({1,0})
      .expected_normalized_dims(2)
      .expected_element_size(4)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_3D_flatten_element_size_2D) {
    TransposeNormalizationTester()
      .num_dims(3)
      .element_size(4)
      .perm({1,0,2})
      .shape({101,13,7})
      .expected_normalized_shape({101,13})
      .expected_normalized_perm({1,0})
      .expected_normalized_dims(2)
      .expected_element_size(7*4)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_3D_flatten_3D) {
    TransposeNormalizationTester()
      .num_dims(3)
      .element_size(4)
      .perm({2,1,0})
      .shape({101,13,7})
      .expected_normalized_shape({101,13,7})
      .expected_normalized_perm({2,1,0})
      .expected_normalized_dims(3)
      .expected_element_size(4)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_3D_redundant_dim_first) {
    TransposeNormalizationTester()
      .num_dims(3)
      .element_size(4)
      .perm({0,2,1})
      .shape({1,19,13})
      .expected_normalized_shape({19,13})
      .expected_normalized_perm({1,0})
      .expected_normalized_dims(2)
      .expected_element_size(4)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_4D_to_1D) {
    TransposeNormalizationTester()
      .num_dims(4)
      .element_size(4)
      .perm({0, 2, 3, 1})
      .shape({2, 2, 1, 1})
      .expected_normalized_shape({1})
      .expected_normalized_perm({0})
      .expected_normalized_dims(1)
      .expected_element_size(16)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_4D_flatten_element_size_2D) {
    TransposeNormalizationTester()
      .num_dims(4)
      .element_size(1)
      .perm({1,0,2,3})
      .shape({101,13,7,19})
      .expected_normalized_shape({101,13})
      .expected_normalized_perm({1,0})
      .expected_normalized_dims(2)
      .expected_element_size(1*7*19)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_4D_flatten_3D) {
    TransposeNormalizationTester()
      .num_dims(4)
      .element_size(2)
      .perm({0,3,1,2})
      .shape({19,31,41,7})
      .expected_normalized_shape({19,31*41,7})
      .expected_normalized_perm({0,2,1})
      .expected_normalized_dims(3)
      .expected_element_size(2)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_5D_double_flatten_3D) {
    TransposeNormalizationTester()
      .num_dims(5)
      .element_size(4)
      .perm({4,2,3,0,1})
      .shape({19,13,31,41,7})
      .expected_normalized_shape({19*13,31*41,7})
      .expected_normalized_perm({2,1,0})
      .expected_normalized_dims(3)
      .expected_element_size(4)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_5D_flatten_3D) {
    TransposeNormalizationTester()
      .num_dims(5)
      .element_size(2)
      .perm({4,3,0,1,2})
      .shape({19,13,31,41,7})
      .expected_normalized_shape({19*13*31,41,7})
      .expected_normalized_perm({2,1,0})
      .expected_normalized_dims(3)
      .expected_element_size(2)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_5D_flatten_4D) {
    TransposeNormalizationTester()
      .num_dims(5)
      .element_size(2)
      .perm({4,3,1,2,0})
      .shape({19,13,31,41,7})
      .expected_normalized_shape({19,13*31,41,7})
      .expected_normalized_perm({3,2,1,0})
      .expected_normalized_dims(4)
      .expected_element_size(2)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_5D_flatten_2D) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(4)
      .perm({4,5,0,1,2,3})
      .shape({53,19,13,31,41,7})
      .expected_normalized_shape({53*19*13*31,41*7})
      .expected_normalized_perm({1,0})
      .expected_normalized_dims(2)
      .expected_element_size(4)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_6D_flatten_3D) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(4)
      .perm({0,1,2,3,5,4})
      .shape({53,19,13,31,41,7})
      .expected_normalized_shape({53*19*13*31,41,7})
      .expected_normalized_perm({0,2,1})
      .expected_normalized_dims(3)
      .expected_element_size(4)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_6D_double_flatten_3D) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(4)
      .perm({0,3,1,2,4,5})
      .shape({53,19,13,31,41,7})
      .expected_normalized_shape({53,19*13,31})
      .expected_normalized_perm({0,2,1})
      .expected_normalized_dims(3)
      .expected_element_size(4*41*7)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_6D_double_flatten_4D) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(4)
      .perm({4,5,3,1,2,0})
      .shape({53,19,13,31,41,7})
      .expected_normalized_shape({53,19*13,31,41*7})
      .expected_normalized_perm({3,2,1,0})
      .expected_normalized_dims(4)
      .expected_element_size(4)
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_6D_flatten_ones) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(4)
      .perm({5,4,3,2,1,0})
      .shape({23,1,1,1,17,13})
      .expected_normalized_shape({23,17,13})
      .expected_normalized_perm({2,1,0})
      .expected_normalized_dims(3)
      .expected_element_size(4)
      .Test();
}
