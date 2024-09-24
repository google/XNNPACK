#include <iostream>
#include <string>
#include <random>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/internal.h"
#include "xnnpack/math.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/packq.h"
#include "xnnpack/requantization.h"
#include "xnnpack/subgraph.h"

void run() {
	  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  std::mt19937 rng;
  
  std::uniform_real_distribution<float> f32dist;
  f32dist = std::uniform_real_distribution<float>(-1.f, 1.0f);
  size_t batch_size = 1;
  size_t input_channels = 112;//1536;//6144/4;
  size_t output_channels = 33;//1536/16;
  float output_min = -std::numeric_limits<float>::infinity();
  float output_max = std::numeric_limits<float>::infinity();
  std::vector<size_t> input_dims{batch_size, input_channels};
  std::vector<size_t> kernel_dims{output_channels, input_channels};
  std::vector<size_t> output_dims{batch_size, output_channels};
  std::vector<size_t> bias_dims{output_channels};
  std::vector<float> input(batch_size * input_channels);
  std::vector<float> kernel(input_channels * output_channels + XNN_EXTRA_BYTES/sizeof(float));
  std::vector<float> bias(output_channels);
  std::vector<float> subgraph_output(output_channels * batch_size);
  std::vector<float> reference_output(output_channels * batch_size, 0.f);
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  //std::fill(bias.begin(), bias.end(), 0);
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));
  for (int i = 0; i < batch_size; ++i) {
	  for (int j = 0; j < output_channels; ++j) {
		  for (int k = 0; k < input_channels; ++k) {
		    //reference_output[j + i * output_channels] += input[k + i * input_channels] * kernel[k + j * input_channels];
		    reference_output[j + i * output_channels] += input[k + i * input_channels] * kernel[k * output_channels + j];
		  }
	  }
  }
  for (int i = 0; i < batch_size; ++i) {
	  for (int j = 0; j < output_channels; ++j) {
		  reference_output[j + i * output_channels] += bias[j];
	  }
  }
  //std::cout<<" input \n";
  //for (int i = 0; i < batch_size; ++i) {
  //        for (int j = 0; j < input_channels; ++j) {
  //      	  std::cout<< input[j + i * input_channels] << ", ";
  //        }
  //        std::cout<<std::endl;
  //}
  //std::cout<<" kernel \n";
  //for (int j = 0; j < input_channels; ++j) {
  //        for (int i = 0; i < output_channels; ++i) {
  //      	  std::cout<< kernel[i + j * output_channels] << ", ";
  //        }
  //        std::cout<<std::endl;
  //}
  //std::cout<<" bias \n";
  //for (int i = 0; i < output_channels; ++i) {
  //        std::cout<< bias[i] << ", ";
  //}

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, kernel_dims.size(), kernel_dims.data(), kernel.data(),
                          /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, bias_dims.size(), bias_dims.data(), bias.data(),
                          /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_fully_connected(subgraph, output_min, output_max, input_id, kernel_id, bias_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  for (int i = 0; i < subgraph_output.size(); ++i) {
	  float tolerance = std::max(1.0e-5f, std::abs(reference_output[i]) * 1.0e-6f);
	  EXPECT_NEAR(subgraph_output[i], reference_output[i], tolerance) << " III " << i;
//  std::cout<<" VAL " << subgraph_output[i] << " ref " << reference_output[i] << std::endl;
  }
  //for (int i = 0; i < subgraph_output.size(); ++i) {
  //        std::cout<< subgraph_output[i] << std::endl;
  //}
  //std::cout<<" ref \n";
  //for (int i = 0; i < reference_output.size(); ++i) {
  //        std::cout<< reference_output[i] << std::endl;
  //}
}

int main(int argc, char** argv) {
  @autoreleasepool {
	  run();
  }

  return 0;
}
