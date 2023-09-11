#include <gtest/gtest.h>

#include <xnnpack.h>
#include <xnnpack/normalization.h>

#include "transpose-normalization-tester.h"

TEST(TRANSPOSE_NORMALIZATION_TEST, input_stride_fold_0_1) {
    TransposeNormalizationTester()
      .num_dims(4)
      .element_size(4)
      .perm({0,1,3,2})
      .shape({5,4,3,2})
      .input_stride({24,6,2,1})
      .expected_shape({20,3,2})
      .expected_perm({0,2,1})
      .expected_dims(3)
      .expected_element_size(4)
      .expected_input_stride({24,8,4})
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, output_stride_fold_0_1) {
    TransposeNormalizationTester()
      .num_dims(4)
      .element_size(4)
      .perm({0,1,3,2})
      .shape({5,4,3,2})
      .output_stride({24,6,3,1})
      .expected_shape({20,3,2})
      .expected_perm({0,2,1})
      .expected_dims(3)
      .expected_element_size(4)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, output_stride) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(1)
      .perm({1,0,3,2,5,4})
      .shape({5,4,3,2,6,7})
      .output_stride({1260,252,126,42,6,1})
      .expected_shape({5,4,3,2,6,7})
      .expected_perm({1,0,3,2,5,4})
      .expected_dims(6)
      .expected_element_size(1)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, output_stride_fold_2_3) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(1)
      .perm({1,0,2,3,5,4})
      .shape({5,4,3,2,6,7})
      .output_stride({1260,252,84,42,6,1})
      .expected_shape({5,4,6,6,7})
      .expected_perm({1,0,2,4,3})
      .expected_dims(5)
      .expected_element_size(1)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, output_stride_no_fold_2_3) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(1)
      .perm({1,0,2,3,5,4})
      .shape({5,4,3,2,6,7})
      .output_stride({1275,255,85,42,6,1})
      .expected_shape({5,4,3,2,6,7})
      .expected_perm({1,0,2,3,5,4})
      .expected_dims(6)
      .expected_element_size(1)
      .calculate_expected_input_stride()
      .expected_output_stride({1275,255,85,42,6,1})
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, output_stride_fold_2_3_with_large_strides) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(1)
      .perm({1,0,2,3,5,4})
      .shape({5,4,3,2,6,7})
      .output_stride({1290,258,86,43,6,1})
      .expected_shape({5,4,6,6,7})
      .expected_perm({1,0,2,4,3})
      .expected_dims(5)
      .expected_element_size(1)
      .calculate_expected_input_stride()
      .expected_output_stride({1290,258,43,6,1})
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, output_stride_fold_last_dim) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(1)
      .perm({1,0,2,4,3,5})
      .shape({5,4,3,2,6,7})
      .output_stride({1260,252,84,14,7,1})
      .expected_shape({5,4,3,2,6})
      .expected_perm({1,0,2,4,3})
      .expected_dims(5)
      .expected_element_size(7)
      .calculate_expected_input_stride()
      .expected_output_stride({1260,252,84,14,7})
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, output_stride_remove_dim_size_1) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(1)
      .perm({1,0,3,2,5,4})
      .shape({5,4,1,2,6,7})
      .output_stride({420,84,42,42,6,1})
      .expected_shape({5,4,2,6,7})
      .expected_perm({1,0,2,4,3})
      .expected_dims(5)
      .expected_element_size(1)
      .calculate_expected_input_stride()
      .expected_output_stride({420,84,42,6,1})
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, output_stride_no_remove_dim_size_1) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(1)
      .perm({1,0,3,2,5,4})
      .shape({5,4,1,2,6,7})
      .output_stride({420,86,43,42,6,1})
      .expected_shape({5,4,1,2,6,7})
      .expected_perm({1,0,3,2,5,4})
      .expected_dims(6)
      .expected_element_size(1)
      .calculate_expected_input_stride()
      .expected_output_stride({420,86,43,42,6,1})
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, output_stride_remove_dim_size_1_4bytes) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(4)
      .perm({1,0,3,2,5,4})
      .shape({5,4,1,2,6,7})
      .output_stride({420,84,42,42,6,1})
      .expected_shape({5,4,2,6,7})
      .expected_perm({1,0,2,4,3})
      .expected_dims(5)
      .expected_element_size(4)
      .calculate_expected_input_stride()
      .expected_output_stride({1680,336,168,24,4})
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, input_output_stride_remove_dim_size_1) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(4)
      .perm({1,0,3,2,5,4})
      .shape({5,4,1,2,6,7})
      .input_stride({336,84,84,42,7,1})
      .output_stride({420,84,42,42,6,1})
      .expected_shape({5,4,2,6,7})
      .expected_perm({1,0,2,4,3})
      .expected_dims(5)
      .expected_element_size(4)
      .expected_input_stride({1344,336,168,28,4})
      .expected_output_stride({1680,336,168,24,4})
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, input_output_stride_remove_fold_large_element) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(1)
      .perm({1,0,3,2,5,4})
      .shape({5,4,1,2,1,7})
      .input_stride({56,14,14,7,7,1})
      .output_stride({70,14,7,7,1,1})
      .expected_shape({5,4})
      .expected_perm({1,0})
      .expected_dims(2)
      .expected_element_size(14)
      .expected_input_stride({56,14})
      .expected_output_stride({70,14})
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, input_output_stride_no_remove_dim_size_1) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(1)
      .perm({1,0,3,2,5,4})
      .shape({5,4,1,2,1,7})
      .input_stride({70,15,14,7,7,1})
      .output_stride({140,28,14,14,2,1})
      .expected_shape({5,4,1,2,1,7})
      .expected_perm({1,0,3,2,5,4})
      .expected_dims(6)
      .expected_element_size(1)
      .expected_input_stride({70,15,14,7,7,1})
      .expected_output_stride({140,28,14,14,2,1})
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, input_output_stride_no_remove_dim_1_fold) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(1)
      .perm({5,4,1,2,0,3})
      .shape({4,9,7,2,1,6})
      .input_stride({882,98,14,7,6,1})
      .output_stride({505,505,56,8,2,1})
      .expected_shape({4,63,2,1,6})
      .expected_perm({4,3,1,0,2})
      .expected_dims(5)
      .expected_element_size(1)
      .expected_input_stride({882,14,7,6,1})
      .expected_output_stride({505,505,8,2,1})
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, input_stride) {
    TransposeNormalizationTester()
      .num_dims(4)
      .element_size(4)
      .perm({3,1,2,0})
      .shape({5,4,3,2})
      .input_stride({24,6,2,1})
      .expected_shape({5,12,2})
      .expected_perm({2,1,0})
      .expected_dims(3)
      .expected_element_size(4)
      .expected_input_stride({96,8,4})
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, input_stride_remove_dim_1) {
    TransposeNormalizationTester()
      .num_dims(4)
      .element_size(4)
      .perm({3,2,1,0})
      .shape({5,4,1,2})
      .input_stride({8,2,2,1})
      .expected_shape({5,4,2})
      .expected_perm({2,1,0})
      .expected_dims(3)
      .expected_element_size(4)
      .expected_input_stride({32,8,4})
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, input_stride_elem_size_1) {
    TransposeNormalizationTester()
      .num_dims(4)
      .element_size(1)
      .perm({3,2,1,0})
      .shape({5,4,2,2})
      .input_stride({16,4,2,1})
      .expected_shape({5,4,2,2})
      .expected_perm({3,2,1,0})
      .expected_dims(4)
      .expected_element_size(1)
      .expected_input_stride({16,4,2,1})
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, input_stride_strided) {
    TransposeNormalizationTester()
      .num_dims(5)
      .element_size(1)
      .perm({3,1,2,4,0})
      .shape({5,4,2,2,3})
      .input_stride({96,24,6,3,1})
      .expected_shape({5,4,2,2,3})
      .expected_perm({3,1,2,4,0})
      .expected_dims(5)
      .expected_element_size(1)
      .expected_input_stride({96,24,6,3,1})
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, input_stride_strided_size_1_dims) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(1)
      .perm({2,1,3,5,4,0})
      .shape({5,4,2,1,1,7})
      .input_stride({112,28,14,14,7,1})
      .expected_shape({5,4,2,1,7})
      .expected_perm({2,1,4,3,0})
      .expected_dims(5)
      .expected_element_size(1)
      .expected_input_stride({112,28,14,7,1})
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, input_stride_size_1_dims_flatten) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(1)
      .perm({3,4,5,2,1,0})
      .shape({1,4,1,3,5,7})
      .input_stride({420,105,105,35,7,1})
      .expected_shape({4,105})
      .expected_perm({1,0})
      .expected_dims(2)
      .expected_element_size(1)
      .expected_input_stride({105,1})
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, input_stride_size_1_dims_flatten_strided) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(1)
      .perm({3,4,5,2,1,0})
      .shape({1,4,1,3,5,7})
      .input_stride({660,165,165,55,11,1})
      .expected_shape({4,15,7})
      .expected_perm({1,2,0})
      .expected_dims(3)
      .expected_element_size(1)
      .expected_input_stride({165,11,1})
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, input_stride_size_1_dims_flatten_strided_copy) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(1)
      .perm({1,0,3,4,2,5})
      .shape({1,4,1,3,5,7})
      .input_stride({668,167,115,35,7,1})
      .expected_shape({4,1,15})
      .expected_perm({0,2,1})
      .expected_dims(3)
      .expected_element_size(7)
      .expected_input_stride({167,115,7})
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, input_stride_size_1_dims_flatten_last_dim_strided_copy) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(4)
      .perm({0,1,3,2,4,5})
      .shape({2,4,1,3,5,7})
      .input_stride({700,160,150,36,7,1})
      .expected_shape({2,4,1,3,1})
      .expected_perm({0,1,3,2,4})
      .expected_dims(5)
      .expected_element_size(140)
      .expected_input_stride({2800,640,600,144,140})
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, input_output_stride_size_1_dims_flatten_last_dim_strided_copy) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(4)
      .perm({0,1,3,2,4,5})
      .shape({2,4,1,3,5,7})
      .input_stride({700,160,150,36,7,1})
      .output_stride({420,105,35,35,7,1})
      .expected_shape({2,4,1,3,1})
      .expected_perm({0,1,3,2,4})
      .expected_dims(5)
      .expected_element_size(140)
      .expected_input_stride({2800,640,600,144,140})
      .expected_output_stride({1680,420,140,140,140})
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, input_output_stride_flatten_last_dim_strided_copy) {
    TransposeNormalizationTester()
      .num_dims(5)
      .element_size(4)
      .perm({0,3,2,1,4})
      .shape({4,2,3,5,7})
      .output_stride({240,48,16,8,1})
      .expected_shape({4,2,3,5,1})
      .expected_perm({0,3,2,1,4})
      .expected_dims(5)
      .expected_element_size(28)
      .calculate_expected_input_stride()
      .expected_output_stride({960,192,64,32,28})
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, input_output_stride_nofold_contiguous_remove_last_dim) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(1)
      .perm({2, 3, 0, 4, 1, 5})
      .shape({1, 1, 2, 3, 3, 1})
      .input_stride({54, 18, 9, 3, 1, 1})
      .output_stride({49, 21, 7, 3, 1, 1})
      .expected_shape({1, 1, 2, 3, 3})
      .expected_perm({2, 3, 0, 4, 1})
      .expected_dims(5)
      .expected_element_size(1)
      .expected_input_stride({54, 18, 9, 3, 1})
      .expected_output_stride({49, 21, 7, 3, 1})
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, adjacent_1_dims) {
    TransposeNormalizationTester()
      .num_dims(4)
      .element_size(4)
      .perm({0,1,3,2})
      .shape({1,1,60,2400})
      .expected_shape({60,2400})
      .expected_perm({1,0})
      .expected_dims(2)
      .expected_element_size(4)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_1D) {
    TransposeNormalizationTester()
      .num_dims(1)
      .element_size(4)
      .perm({0})
      .shape({37})
      .expected_shape({1})
      .expected_perm({0})
      .expected_dims(1)
      .expected_element_size(37*4)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_2D_flatten_1D) {
    TransposeNormalizationTester()
      .num_dims(2)
      .element_size(4)
      .perm({0,1})
      .shape({37,19})
      .expected_shape({1})
      .expected_perm({0})
      .expected_dims(1)
      .expected_element_size(37*19*4)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_2D_flatten_2D) {
    TransposeNormalizationTester()
      .num_dims(2)
      .element_size(4)
      .perm({1,0})
      .shape({23,17})
      .expected_shape({23,17})
      .expected_perm({1,0})
      .expected_dims(2)
      .expected_element_size(4)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_3D_redundant_dim) {
    TransposeNormalizationTester()
      .num_dims(3)
      .element_size(4)
      .perm({0,2,1})
      .shape({2,1,3})
      .expected_shape({1})
      .expected_perm({0})
      .expected_dims(1)
      .expected_element_size(24)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_3D_all_ones) {
    TransposeNormalizationTester()
      .num_dims(3)
      .element_size(4)
      .perm({2,1,0})
      .shape({1,1,1})
      .expected_shape({1})
      .expected_perm({0})
      .expected_dims(1)
      .expected_element_size(4)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_3D_flatten_1D) {
    TransposeNormalizationTester()
      .num_dims(3)
      .element_size(4)
      .perm({0,1,2})
      .shape({101,13,7})
      .expected_shape({1})
      .expected_perm({0})
      .expected_dims(1)
      .expected_element_size(101*13*7*4)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_3D_flatten_2D) {
    TransposeNormalizationTester()
      .num_dims(3)
      .element_size(4)
      .perm({2,0,1})
      .shape({101,13,7})
      .expected_shape({101*13,7})
      .expected_perm({1,0})
      .expected_dims(2)
      .expected_element_size(4)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_3D_flatten_element_size_2D) {
    TransposeNormalizationTester()
      .num_dims(3)
      .element_size(4)
      .perm({1,0,2})
      .shape({101,13,7})
      .expected_shape({101,13})
      .expected_perm({1,0})
      .expected_dims(2)
      .expected_element_size(7*4)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_3D_flatten_3D) {
    TransposeNormalizationTester()
      .num_dims(3)
      .element_size(4)
      .perm({2,1,0})
      .shape({101,13,7})
      .expected_shape({101,13,7})
      .expected_perm({2,1,0})
      .expected_dims(3)
      .expected_element_size(4)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_3D_redundant_dim_first) {
    TransposeNormalizationTester()
      .num_dims(3)
      .element_size(4)
      .perm({0,2,1})
      .shape({1,19,13})
      .expected_shape({19,13})
      .expected_perm({1,0})
      .expected_dims(2)
      .expected_element_size(4)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_4D_to_1D) {
    TransposeNormalizationTester()
      .num_dims(4)
      .element_size(4)
      .perm({0, 2, 3, 1})
      .shape({2, 2, 1, 1})
      .expected_shape({1})
      .expected_perm({0})
      .expected_dims(1)
      .expected_element_size(16)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_4D_flatten_element_size_2D) {
    TransposeNormalizationTester()
      .num_dims(4)
      .element_size(1)
      .perm({1,0,2,3})
      .shape({101,13,7,19})
      .expected_shape({101,13})
      .expected_perm({1,0})
      .expected_dims(2)
      .expected_element_size(1*7*19)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_4D_flatten_3D) {
    TransposeNormalizationTester()
      .num_dims(4)
      .element_size(2)
      .perm({0,3,1,2})
      .shape({19,31,41,7})
      .expected_shape({19,31*41,7})
      .expected_perm({0,2,1})
      .expected_dims(3)
      .expected_element_size(2)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_5D_double_flatten_3D) {
    TransposeNormalizationTester()
      .num_dims(5)
      .element_size(4)
      .perm({4,2,3,0,1})
      .shape({19,13,31,41,7})
      .expected_shape({19*13,31*41,7})
      .expected_perm({2,1,0})
      .expected_dims(3)
      .expected_element_size(4)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_5D_flatten_3D) {
    TransposeNormalizationTester()
      .num_dims(5)
      .element_size(2)
      .perm({4,3,0,1,2})
      .shape({19,13,31,41,7})
      .expected_shape({19*13*31,41,7})
      .expected_perm({2,1,0})
      .expected_dims(3)
      .expected_element_size(2)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_5D_flatten_4D) {
    TransposeNormalizationTester()
      .num_dims(5)
      .element_size(2)
      .perm({4,3,1,2,0})
      .shape({19,13,31,41,7})
      .expected_shape({19,13*31,41,7})
      .expected_perm({3,2,1,0})
      .expected_dims(4)
      .expected_element_size(2)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_5D_flatten_2D) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(4)
      .perm({4,5,0,1,2,3})
      .shape({53,19,13,31,41,7})
      .expected_shape({53*19*13*31,41*7})
      .expected_perm({1,0})
      .expected_dims(2)
      .expected_element_size(4)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_6D_flatten_3D) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(4)
      .perm({0,1,2,3,5,4})
      .shape({53,19,13,31,41,7})
      .expected_shape({53*19*13*31,41,7})
      .expected_perm({0,2,1})
      .expected_dims(3)
      .expected_element_size(4)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_6D_double_flatten_3D) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(4)
      .perm({0,3,1,2,4,5})
      .shape({53,19,13,31,41,7})
      .expected_shape({53,19*13,31})
      .expected_perm({0,2,1})
      .expected_dims(3)
      .expected_element_size(4*41*7)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_6D_double_flatten_4D) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(4)
      .perm({4,5,3,1,2,0})
      .shape({53,19,13,31,41,7})
      .expected_shape({53,19*13,31,41*7})
      .expected_perm({3,2,1,0})
      .expected_dims(4)
      .expected_element_size(4)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}

TEST(TRANSPOSE_NORMALIZATION_TEST, normalize_6D_flatten_ones) {
    TransposeNormalizationTester()
      .num_dims(6)
      .element_size(4)
      .perm({5,4,3,2,1,0})
      .shape({23,1,1,1,17,13})
      .expected_shape({23,17,13})
      .expected_perm({2,1,0})
      .expected_dims(3)
      .expected_element_size(4)
      .calculate_expected_input_stride()
      .calculate_expected_output_stride()
      .Test();
}
