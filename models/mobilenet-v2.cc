// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include "models/models.h"

namespace models {

ExecutionPlan MobileNetV2(pthreadpool_t threadpool) {
  alignas(16) static float v0[150528];
  alignas(16) static float v1[401408];
  alignas(16) static float v2[401408];
  alignas(16) static float v3[200704];
  alignas(16) static float v4[1204224];
  alignas(16) static float v5[301056];
  alignas(16) static float v6[75264];
  alignas(16) static float v7[451584];
  alignas(16) static float v8[451584];
  alignas(16) static float v9[75264];
  alignas(16) static float v10[75264];
  alignas(16) static float v11[451584];
  alignas(16) static float v12[112896];
  alignas(16) static float v13[25088];
  alignas(16) static float v14[150528];
  alignas(16) static float v15[150528];
  alignas(16) static float v16[25088];
  alignas(16) static float v17[25088];
  alignas(16) static float v18[150528];
  alignas(16) static float v19[150528];
  alignas(16) static float v20[25088];
  alignas(16) static float v21[25088];
  alignas(16) static float v22[150528];
  alignas(16) static float v23[37632];
  alignas(16) static float v24[12544];
  alignas(16) static float v25[75264];
  alignas(16) static float v26[75264];
  alignas(16) static float v27[12544];
  alignas(16) static float v28[12544];
  alignas(16) static float v29[75264];
  alignas(16) static float v30[75264];
  alignas(16) static float v31[12544];
  alignas(16) static float v32[12544];
  alignas(16) static float v33[75264];
  alignas(16) static float v34[75264];
  alignas(16) static float v35[12544];
  alignas(16) static float v36[12544];
  alignas(16) static float v37[75264];
  alignas(16) static float v38[75264];
  alignas(16) static float v39[18816];
  alignas(16) static float v40[112896];
  alignas(16) static float v41[112896];
  alignas(16) static float v42[18816];
  alignas(16) static float v43[18816];
  alignas(16) static float v44[112896];
  alignas(16) static float v45[112896];
  alignas(16) static float v46[18816];
  alignas(16) static float v47[18816];
  alignas(16) static float v48[112896];
  alignas(16) static float v49[28224];
  alignas(16) static float v50[7840];
  alignas(16) static float v51[47040];
  alignas(16) static float v52[47040];
  alignas(16) static float v53[7840];
  alignas(16) static float v54[7840];
  alignas(16) static float v55[47040];
  alignas(16) static float v56[47040];
  alignas(16) static float v57[7840];
  alignas(16) static float v58[7840];
  alignas(16) static float v59[47040];
  alignas(16) static float v60[47040];
  alignas(16) static float v61[15680];
  alignas(16) static float v62[62720];
  alignas(16) static float v63[1280];
  alignas(16) static float v64[1001];
  alignas(16) static float w65[864];
  alignas(16) static float w66[32];
  alignas(16) static float w67[288];
  alignas(16) static float w68[32];
  alignas(16) static float w69[512];
  alignas(16) static float w70[16];
  alignas(16) static float w71[1536];
  alignas(16) static float w72[96];
  alignas(16) static float w73[864];
  alignas(16) static float w74[96];
  alignas(16) static float w75[2304];
  alignas(16) static float w76[24];
  alignas(16) static float w77[3456];
  alignas(16) static float w78[144];
  alignas(16) static float w79[1296];
  alignas(16) static float w80[144];
  alignas(16) static float w81[3456];
  alignas(16) static float w82[24];
  alignas(16) static float w83[3456];
  alignas(16) static float w84[144];
  alignas(16) static float w85[1296];
  alignas(16) static float w86[144];
  alignas(16) static float w87[4608];
  alignas(16) static float w88[32];
  alignas(16) static float w89[6144];
  alignas(16) static float w90[192];
  alignas(16) static float w91[1728];
  alignas(16) static float w92[192];
  alignas(16) static float w93[6144];
  alignas(16) static float w94[32];
  alignas(16) static float w95[6144];
  alignas(16) static float w96[192];
  alignas(16) static float w97[1728];
  alignas(16) static float w98[192];
  alignas(16) static float w99[6144];
  alignas(16) static float w100[32];
  alignas(16) static float w101[6144];
  alignas(16) static float w102[192];
  alignas(16) static float w103[1728];
  alignas(16) static float w104[192];
  alignas(16) static float w105[12288];
  alignas(16) static float w106[64];
  alignas(16) static float w107[24576];
  alignas(16) static float w108[384];
  alignas(16) static float w109[3456];
  alignas(16) static float w110[384];
  alignas(16) static float w111[24576];
  alignas(16) static float w112[64];
  alignas(16) static float w113[24576];
  alignas(16) static float w114[384];
  alignas(16) static float w115[3456];
  alignas(16) static float w116[384];
  alignas(16) static float w117[24576];
  alignas(16) static float w118[64];
  alignas(16) static float w119[24576];
  alignas(16) static float w120[384];
  alignas(16) static float w121[3456];
  alignas(16) static float w122[384];
  alignas(16) static float w123[24576];
  alignas(16) static float w124[64];
  alignas(16) static float w125[24576];
  alignas(16) static float w126[384];
  alignas(16) static float w127[3456];
  alignas(16) static float w128[384];
  alignas(16) static float w129[36864];
  alignas(16) static float w130[96];
  alignas(16) static float w131[55296];
  alignas(16) static float w132[576];
  alignas(16) static float w133[5184];
  alignas(16) static float w134[576];
  alignas(16) static float w135[55296];
  alignas(16) static float w136[96];
  alignas(16) static float w137[55296];
  alignas(16) static float w138[576];
  alignas(16) static float w139[5184];
  alignas(16) static float w140[576];
  alignas(16) static float w141[55296];
  alignas(16) static float w142[96];
  alignas(16) static float w143[55296];
  alignas(16) static float w144[576];
  alignas(16) static float w145[5184];
  alignas(16) static float w146[576];
  alignas(16) static float w147[92160];
  alignas(16) static float w148[160];
  alignas(16) static float w149[153600];
  alignas(16) static float w150[960];
  alignas(16) static float w151[8640];
  alignas(16) static float w152[960];
  alignas(16) static float w153[153600];
  alignas(16) static float w154[160];
  alignas(16) static float w155[153600];
  alignas(16) static float w156[960];
  alignas(16) static float w157[8640];
  alignas(16) static float w158[960];
  alignas(16) static float w159[153600];
  alignas(16) static float w160[160];
  alignas(16) static float w161[153600];
  alignas(16) static float w162[960];
  alignas(16) static float w163[8640];
  alignas(16) static float w164[960];
  alignas(16) static float w165[307200];
  alignas(16) static float w166[320];
  alignas(16) static float w167[409600];
  alignas(16) static float w168[1280];
  alignas(16) static float w169[1281280];
  alignas(16) static float w170[1001];

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), rng);
  std::generate(v0, v0 + 150528, std::ref(f32rng));
  std::generate(w65, w65 + 864, std::ref(f32rng));
  std::generate(w66, w66 + 32, std::ref(f32rng));
  std::generate(w67, w67 + 288, std::ref(f32rng));
  std::generate(w68, w68 + 32, std::ref(f32rng));
  std::generate(w69, w69 + 512, std::ref(f32rng));
  std::generate(w70, w70 + 16, std::ref(f32rng));
  std::generate(w71, w71 + 1536, std::ref(f32rng));
  std::generate(w72, w72 + 96, std::ref(f32rng));
  std::generate(w73, w73 + 864, std::ref(f32rng));
  std::generate(w74, w74 + 96, std::ref(f32rng));
  std::generate(w75, w75 + 2304, std::ref(f32rng));
  std::generate(w76, w76 + 24, std::ref(f32rng));
  std::generate(w77, w77 + 3456, std::ref(f32rng));
  std::generate(w78, w78 + 144, std::ref(f32rng));
  std::generate(w79, w79 + 1296, std::ref(f32rng));
  std::generate(w80, w80 + 144, std::ref(f32rng));
  std::generate(w81, w81 + 3456, std::ref(f32rng));
  std::generate(w82, w82 + 24, std::ref(f32rng));
  std::generate(w83, w83 + 3456, std::ref(f32rng));
  std::generate(w84, w84 + 144, std::ref(f32rng));
  std::generate(w85, w85 + 1296, std::ref(f32rng));
  std::generate(w86, w86 + 144, std::ref(f32rng));
  std::generate(w87, w87 + 4608, std::ref(f32rng));
  std::generate(w88, w88 + 32, std::ref(f32rng));
  std::generate(w89, w89 + 6144, std::ref(f32rng));
  std::generate(w90, w90 + 192, std::ref(f32rng));
  std::generate(w91, w91 + 1728, std::ref(f32rng));
  std::generate(w92, w92 + 192, std::ref(f32rng));
  std::generate(w93, w93 + 6144, std::ref(f32rng));
  std::generate(w94, w94 + 32, std::ref(f32rng));
  std::generate(w95, w95 + 6144, std::ref(f32rng));
  std::generate(w96, w96 + 192, std::ref(f32rng));
  std::generate(w97, w97 + 1728, std::ref(f32rng));
  std::generate(w98, w98 + 192, std::ref(f32rng));
  std::generate(w99, w99 + 6144, std::ref(f32rng));
  std::generate(w100, w100 + 32, std::ref(f32rng));
  std::generate(w101, w101 + 6144, std::ref(f32rng));
  std::generate(w102, w102 + 192, std::ref(f32rng));
  std::generate(w103, w103 + 1728, std::ref(f32rng));
  std::generate(w104, w104 + 192, std::ref(f32rng));
  std::generate(w105, w105 + 12288, std::ref(f32rng));
  std::generate(w106, w106 + 64, std::ref(f32rng));
  std::generate(w107, w107 + 24576, std::ref(f32rng));
  std::generate(w108, w108 + 384, std::ref(f32rng));
  std::generate(w109, w109 + 3456, std::ref(f32rng));
  std::generate(w110, w110 + 384, std::ref(f32rng));
  std::generate(w111, w111 + 24576, std::ref(f32rng));
  std::generate(w112, w112 + 64, std::ref(f32rng));
  std::generate(w113, w113 + 24576, std::ref(f32rng));
  std::generate(w114, w114 + 384, std::ref(f32rng));
  std::generate(w115, w115 + 3456, std::ref(f32rng));
  std::generate(w116, w116 + 384, std::ref(f32rng));
  std::generate(w117, w117 + 24576, std::ref(f32rng));
  std::generate(w118, w118 + 64, std::ref(f32rng));
  std::generate(w119, w119 + 24576, std::ref(f32rng));
  std::generate(w120, w120 + 384, std::ref(f32rng));
  std::generate(w121, w121 + 3456, std::ref(f32rng));
  std::generate(w122, w122 + 384, std::ref(f32rng));
  std::generate(w123, w123 + 24576, std::ref(f32rng));
  std::generate(w124, w124 + 64, std::ref(f32rng));
  std::generate(w125, w125 + 24576, std::ref(f32rng));
  std::generate(w126, w126 + 384, std::ref(f32rng));
  std::generate(w127, w127 + 3456, std::ref(f32rng));
  std::generate(w128, w128 + 384, std::ref(f32rng));
  std::generate(w129, w129 + 36864, std::ref(f32rng));
  std::generate(w130, w130 + 96, std::ref(f32rng));
  std::generate(w131, w131 + 55296, std::ref(f32rng));
  std::generate(w132, w132 + 576, std::ref(f32rng));
  std::generate(w133, w133 + 5184, std::ref(f32rng));
  std::generate(w134, w134 + 576, std::ref(f32rng));
  std::generate(w135, w135 + 55296, std::ref(f32rng));
  std::generate(w136, w136 + 96, std::ref(f32rng));
  std::generate(w137, w137 + 55296, std::ref(f32rng));
  std::generate(w138, w138 + 576, std::ref(f32rng));
  std::generate(w139, w139 + 5184, std::ref(f32rng));
  std::generate(w140, w140 + 576, std::ref(f32rng));
  std::generate(w141, w141 + 55296, std::ref(f32rng));
  std::generate(w142, w142 + 96, std::ref(f32rng));
  std::generate(w143, w143 + 55296, std::ref(f32rng));
  std::generate(w144, w144 + 576, std::ref(f32rng));
  std::generate(w145, w145 + 5184, std::ref(f32rng));
  std::generate(w146, w146 + 576, std::ref(f32rng));
  std::generate(w147, w147 + 92160, std::ref(f32rng));
  std::generate(w148, w148 + 160, std::ref(f32rng));
  std::generate(w149, w149 + 153600, std::ref(f32rng));
  std::generate(w150, w150 + 960, std::ref(f32rng));
  std::generate(w151, w151 + 8640, std::ref(f32rng));
  std::generate(w152, w152 + 960, std::ref(f32rng));
  std::generate(w153, w153 + 153600, std::ref(f32rng));
  std::generate(w154, w154 + 160, std::ref(f32rng));
  std::generate(w155, w155 + 153600, std::ref(f32rng));
  std::generate(w156, w156 + 960, std::ref(f32rng));
  std::generate(w157, w157 + 8640, std::ref(f32rng));
  std::generate(w158, w158 + 960, std::ref(f32rng));
  std::generate(w159, w159 + 153600, std::ref(f32rng));
  std::generate(w160, w160 + 160, std::ref(f32rng));
  std::generate(w161, w161 + 153600, std::ref(f32rng));
  std::generate(w162, w162 + 960, std::ref(f32rng));
  std::generate(w163, w163 + 8640, std::ref(f32rng));
  std::generate(w164, w164 + 960, std::ref(f32rng));
  std::generate(w165, w165 + 307200, std::ref(f32rng));
  std::generate(w166, w166 + 320, std::ref(f32rng));
  std::generate(w167, w167 + 409600, std::ref(f32rng));
  std::generate(w168, w168 + 1280, std::ref(f32rng));
  std::generate(w169, w169 + 1281280, std::ref(f32rng));
  std::generate(w170, w170 + 1001, std::ref(f32rng));

  ExecutionPlan operators;
  xnn_status status;

  xnn_operator_t op0 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    3 /* input channels per group */,
    32 /* output_channels_per_group */,
    3 /* input pixel stride */,
    32 /* output pixel stride */,
    w65, w66,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #0" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op0, xnn_delete_operator);

  xnn_operator_t op1 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    32 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    32 /* input pixel stride */,
    32 /* output pixel stride */,
    w67, w68,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #1" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op1, xnn_delete_operator);

  xnn_operator_t op2 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    32 /* input channels per group */,
    16 /* output_channels_per_group */,
    32 /* input pixel stride */,
    16 /* output pixel stride */,
    w69, w70,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #2" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op2, xnn_delete_operator);

  xnn_operator_t op3 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    16 /* input channels per group */,
    96 /* output_channels_per_group */,
    16 /* input pixel stride */,
    96 /* output pixel stride */,
    w71, w72,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #3" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op3, xnn_delete_operator);

  xnn_operator_t op4 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    96 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    96 /* input pixel stride */,
    96 /* output pixel stride */,
    w73, w74,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op4);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #4" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op4, xnn_delete_operator);

  xnn_operator_t op5 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    96 /* input channels per group */,
    24 /* output_channels_per_group */,
    96 /* input pixel stride */,
    24 /* output pixel stride */,
    w75, w76,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #5" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op5, xnn_delete_operator);

  xnn_operator_t op6 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    24 /* input channels per group */,
    144 /* output_channels_per_group */,
    24 /* input pixel stride */,
    144 /* output pixel stride */,
    w77, w78,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op6);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #6" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op6, xnn_delete_operator);

  xnn_operator_t op7 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    144 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    144 /* input pixel stride */,
    144 /* output pixel stride */,
    w79, w80,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op7);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #7" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op7, xnn_delete_operator);

  xnn_operator_t op8 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    144 /* input channels per group */,
    24 /* output_channels_per_group */,
    144 /* input pixel stride */,
    24 /* output pixel stride */,
    w81, w82,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #8" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op8, xnn_delete_operator);

  xnn_operator_t op9 = nullptr;
  status = xnn_create_add_nc_f32(
    24 /* channels */,
    24 /* a stride */,
    24 /* b stride */,
    24 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #9" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op9, xnn_delete_operator);

  xnn_operator_t op10 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    24 /* input channels per group */,
    144 /* output_channels_per_group */,
    24 /* input pixel stride */,
    144 /* output pixel stride */,
    w83, w84,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #10" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op10, xnn_delete_operator);

  xnn_operator_t op11 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    144 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    144 /* input pixel stride */,
    144 /* output pixel stride */,
    w85, w86,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #11" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op11, xnn_delete_operator);

  xnn_operator_t op12 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    144 /* input channels per group */,
    32 /* output_channels_per_group */,
    144 /* input pixel stride */,
    32 /* output pixel stride */,
    w87, w88,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #12" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op12, xnn_delete_operator);

  xnn_operator_t op13 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    32 /* input channels per group */,
    192 /* output_channels_per_group */,
    32 /* input pixel stride */,
    192 /* output pixel stride */,
    w89, w90,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #13" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op13, xnn_delete_operator);

  xnn_operator_t op14 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    192 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    192 /* input pixel stride */,
    192 /* output pixel stride */,
    w91, w92,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op14);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #14" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op14, xnn_delete_operator);

  xnn_operator_t op15 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    192 /* input channels per group */,
    32 /* output_channels_per_group */,
    192 /* input pixel stride */,
    32 /* output pixel stride */,
    w93, w94,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #15" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op15, xnn_delete_operator);

  xnn_operator_t op16 = nullptr;
  status = xnn_create_add_nc_f32(
    32 /* channels */,
    32 /* a stride */,
    32 /* b stride */,
    32 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #16" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op16, xnn_delete_operator);

  xnn_operator_t op17 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    32 /* input channels per group */,
    192 /* output_channels_per_group */,
    32 /* input pixel stride */,
    192 /* output pixel stride */,
    w95, w96,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #17" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op17, xnn_delete_operator);

  xnn_operator_t op18 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    192 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    192 /* input pixel stride */,
    192 /* output pixel stride */,
    w97, w98,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op18);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #18" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op18, xnn_delete_operator);

  xnn_operator_t op19 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    192 /* input channels per group */,
    32 /* output_channels_per_group */,
    192 /* input pixel stride */,
    32 /* output pixel stride */,
    w99, w100,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #19" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op19, xnn_delete_operator);

  xnn_operator_t op20 = nullptr;
  status = xnn_create_add_nc_f32(
    32 /* channels */,
    32 /* a stride */,
    32 /* b stride */,
    32 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op20);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #20" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op20, xnn_delete_operator);

  xnn_operator_t op21 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    32 /* input channels per group */,
    192 /* output_channels_per_group */,
    32 /* input pixel stride */,
    192 /* output pixel stride */,
    w101, w102,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op21);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #21" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op21, xnn_delete_operator);

  xnn_operator_t op22 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    192 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    192 /* input pixel stride */,
    192 /* output pixel stride */,
    w103, w104,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op22);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #22" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op22, xnn_delete_operator);

  xnn_operator_t op23 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    192 /* input channels per group */,
    64 /* output_channels_per_group */,
    192 /* input pixel stride */,
    64 /* output pixel stride */,
    w105, w106,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op23);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #23" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op23, xnn_delete_operator);

  xnn_operator_t op24 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    64 /* input channels per group */,
    384 /* output_channels_per_group */,
    64 /* input pixel stride */,
    384 /* output pixel stride */,
    w107, w108,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #24" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op24, xnn_delete_operator);

  xnn_operator_t op25 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    384 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    384 /* input pixel stride */,
    384 /* output pixel stride */,
    w109, w110,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #25" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op25, xnn_delete_operator);

  xnn_operator_t op26 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    384 /* input channels per group */,
    64 /* output_channels_per_group */,
    384 /* input pixel stride */,
    64 /* output pixel stride */,
    w111, w112,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #26" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op26, xnn_delete_operator);

  xnn_operator_t op27 = nullptr;
  status = xnn_create_add_nc_f32(
    64 /* channels */,
    64 /* a stride */,
    64 /* b stride */,
    64 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #27" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op27, xnn_delete_operator);

  xnn_operator_t op28 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    64 /* input channels per group */,
    384 /* output_channels_per_group */,
    64 /* input pixel stride */,
    384 /* output pixel stride */,
    w113, w114,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #28" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op28, xnn_delete_operator);

  xnn_operator_t op29 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    384 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    384 /* input pixel stride */,
    384 /* output pixel stride */,
    w115, w116,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op29);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #29" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op29, xnn_delete_operator);

  xnn_operator_t op30 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    384 /* input channels per group */,
    64 /* output_channels_per_group */,
    384 /* input pixel stride */,
    64 /* output pixel stride */,
    w117, w118,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op30);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #30" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op30, xnn_delete_operator);

  xnn_operator_t op31 = nullptr;
  status = xnn_create_add_nc_f32(
    64 /* channels */,
    64 /* a stride */,
    64 /* b stride */,
    64 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op31);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #31" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op31, xnn_delete_operator);

  xnn_operator_t op32 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    64 /* input channels per group */,
    384 /* output_channels_per_group */,
    64 /* input pixel stride */,
    384 /* output pixel stride */,
    w119, w120,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #32" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op32, xnn_delete_operator);

  xnn_operator_t op33 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    384 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    384 /* input pixel stride */,
    384 /* output pixel stride */,
    w121, w122,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op33);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #33" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op33, xnn_delete_operator);

  xnn_operator_t op34 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    384 /* input channels per group */,
    64 /* output_channels_per_group */,
    384 /* input pixel stride */,
    64 /* output pixel stride */,
    w123, w124,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op34);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #34" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op34, xnn_delete_operator);

  xnn_operator_t op35 = nullptr;
  status = xnn_create_add_nc_f32(
    64 /* channels */,
    64 /* a stride */,
    64 /* b stride */,
    64 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op35);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #35" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op35, xnn_delete_operator);

  xnn_operator_t op36 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    64 /* input channels per group */,
    384 /* output_channels_per_group */,
    64 /* input pixel stride */,
    384 /* output pixel stride */,
    w125, w126,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op36);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #36" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op36, xnn_delete_operator);

  xnn_operator_t op37 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    384 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    384 /* input pixel stride */,
    384 /* output pixel stride */,
    w127, w128,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op37);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #37" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op37, xnn_delete_operator);

  xnn_operator_t op38 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    384 /* input channels per group */,
    96 /* output_channels_per_group */,
    384 /* input pixel stride */,
    96 /* output pixel stride */,
    w129, w130,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op38);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #38" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op38, xnn_delete_operator);

  xnn_operator_t op39 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    96 /* input channels per group */,
    576 /* output_channels_per_group */,
    96 /* input pixel stride */,
    576 /* output pixel stride */,
    w131, w132,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op39);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #39" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op39, xnn_delete_operator);

  xnn_operator_t op40 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    576 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    576 /* input pixel stride */,
    576 /* output pixel stride */,
    w133, w134,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op40);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #40" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op40, xnn_delete_operator);

  xnn_operator_t op41 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    576 /* input channels per group */,
    96 /* output_channels_per_group */,
    576 /* input pixel stride */,
    96 /* output pixel stride */,
    w135, w136,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op41);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #41" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op41, xnn_delete_operator);

  xnn_operator_t op42 = nullptr;
  status = xnn_create_add_nc_f32(
    96 /* channels */,
    96 /* a stride */,
    96 /* b stride */,
    96 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op42);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #42" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op42, xnn_delete_operator);

  xnn_operator_t op43 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    96 /* input channels per group */,
    576 /* output_channels_per_group */,
    96 /* input pixel stride */,
    576 /* output pixel stride */,
    w137, w138,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op43);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #43" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op43, xnn_delete_operator);

  xnn_operator_t op44 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    576 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    576 /* input pixel stride */,
    576 /* output pixel stride */,
    w139, w140,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op44);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #44" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op44, xnn_delete_operator);

  xnn_operator_t op45 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    576 /* input channels per group */,
    96 /* output_channels_per_group */,
    576 /* input pixel stride */,
    96 /* output pixel stride */,
    w141, w142,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op45);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #45" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op45, xnn_delete_operator);

  xnn_operator_t op46 = nullptr;
  status = xnn_create_add_nc_f32(
    96 /* channels */,
    96 /* a stride */,
    96 /* b stride */,
    96 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op46);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #46" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op46, xnn_delete_operator);

  xnn_operator_t op47 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    96 /* input channels per group */,
    576 /* output_channels_per_group */,
    96 /* input pixel stride */,
    576 /* output pixel stride */,
    w143, w144,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op47);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #47" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op47, xnn_delete_operator);

  xnn_operator_t op48 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    576 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    576 /* input pixel stride */,
    576 /* output pixel stride */,
    w145, w146,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op48);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #48" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op48, xnn_delete_operator);

  xnn_operator_t op49 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    576 /* input channels per group */,
    160 /* output_channels_per_group */,
    576 /* input pixel stride */,
    160 /* output pixel stride */,
    w147, w148,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op49);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #49" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op49, xnn_delete_operator);

  xnn_operator_t op50 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    160 /* input channels per group */,
    960 /* output_channels_per_group */,
    160 /* input pixel stride */,
    960 /* output pixel stride */,
    w149, w150,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op50);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #50" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op50, xnn_delete_operator);

  xnn_operator_t op51 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    960 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    960 /* input pixel stride */,
    960 /* output pixel stride */,
    w151, w152,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op51);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #51" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op51, xnn_delete_operator);

  xnn_operator_t op52 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    960 /* input channels per group */,
    160 /* output_channels_per_group */,
    960 /* input pixel stride */,
    160 /* output pixel stride */,
    w153, w154,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op52);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #52" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op52, xnn_delete_operator);

  xnn_operator_t op53 = nullptr;
  status = xnn_create_add_nc_f32(
    160 /* channels */,
    160 /* a stride */,
    160 /* b stride */,
    160 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op53);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #53" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op53, xnn_delete_operator);

  xnn_operator_t op54 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    160 /* input channels per group */,
    960 /* output_channels_per_group */,
    160 /* input pixel stride */,
    960 /* output pixel stride */,
    w155, w156,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op54);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #54" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op54, xnn_delete_operator);

  xnn_operator_t op55 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    960 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    960 /* input pixel stride */,
    960 /* output pixel stride */,
    w157, w158,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op55);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #55" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op55, xnn_delete_operator);

  xnn_operator_t op56 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    960 /* input channels per group */,
    160 /* output_channels_per_group */,
    960 /* input pixel stride */,
    160 /* output pixel stride */,
    w159, w160,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op56);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #56" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op56, xnn_delete_operator);

  xnn_operator_t op57 = nullptr;
  status = xnn_create_add_nc_f32(
    160 /* channels */,
    160 /* a stride */,
    160 /* b stride */,
    160 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op57);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #57" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op57, xnn_delete_operator);

  xnn_operator_t op58 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    160 /* input channels per group */,
    960 /* output_channels_per_group */,
    160 /* input pixel stride */,
    960 /* output pixel stride */,
    w161, w162,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op58);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #58" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op58, xnn_delete_operator);

  xnn_operator_t op59 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    960 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    960 /* input pixel stride */,
    960 /* output pixel stride */,
    w163, w164,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op59);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #59" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op59, xnn_delete_operator);

  xnn_operator_t op60 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    960 /* input channels per group */,
    320 /* output_channels_per_group */,
    960 /* input pixel stride */,
    320 /* output pixel stride */,
    w165, w166,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op60);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #60" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op60, xnn_delete_operator);

  xnn_operator_t op61 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    320 /* input channels per group */,
    1280 /* output_channels_per_group */,
    320 /* input pixel stride */,
    1280 /* output pixel stride */,
    w167, w168,
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &op61);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #61" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op61, xnn_delete_operator);

  xnn_operator_t op62 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    1280 /* channels */, 1280 /* input stride */, 1280 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op62);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #62" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op62, xnn_delete_operator);

  xnn_operator_t op63 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    1280 /* input channels per group */,
    1001 /* output_channels_per_group */,
    1280 /* input pixel stride */,
    1001 /* output pixel stride */,
    w169, w170,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op63);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #63" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op63, xnn_delete_operator);



  status = xnn_setup_convolution2d_nhwc_f32(
    op0,
    1 /* batch size */, 224 /* input height */, 224 /* input width */,
    &v0[0] /* input */, &v1[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op1,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    &v1[0] /* input */, &v2[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #1" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op2,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    &v2[0] /* input */, &v3[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op3,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    &v3[0] /* input */, &v4[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #3" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op4,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    &v4[0] /* input */, &v5[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op5,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    &v5[0] /* input */, &v6[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #5" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op6,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    &v6[0] /* input */, &v7[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op7,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    &v7[0] /* input */, &v8[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op8,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    &v8[0] /* input */, &v9[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #8" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nc_f32(
    op9,
    3136 /* batch size */,
    &v9[0] /* a */, &v6[0] /* b */, &v10[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op10,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    &v10[0] /* input */, &v11[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #10" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op11,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    &v11[0] /* input */, &v12[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op12,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    &v12[0] /* input */, &v13[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op13,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    &v13[0] /* input */, &v14[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #13" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op14,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    &v14[0] /* input */, &v15[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op15,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    &v15[0] /* input */, &v16[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #15" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nc_f32(
    op16,
    784 /* batch size */,
    &v16[0] /* a */, &v13[0] /* b */, &v17[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #16" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op17,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    &v17[0] /* input */, &v18[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op18,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    &v18[0] /* input */, &v19[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op19,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    &v19[0] /* input */, &v20[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #19" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nc_f32(
    op20,
    784 /* batch size */,
    &v20[0] /* a */, &v17[0] /* b */, &v21[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op21,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    &v21[0] /* input */, &v22[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #21" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op22,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    &v22[0] /* input */, &v23[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op23,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v23[0] /* input */, &v24[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #23" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op24,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v24[0] /* input */, &v25[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op25,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v25[0] /* input */, &v26[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #25" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op26,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v26[0] /* input */, &v27[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #26" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nc_f32(
    op27,
    196 /* batch size */,
    &v27[0] /* a */, &v24[0] /* b */, &v28[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op28,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v28[0] /* input */, &v29[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #28" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op29,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v29[0] /* input */, &v30[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #29" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op30,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v30[0] /* input */, &v31[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #30" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nc_f32(
    op31,
    196 /* batch size */,
    &v31[0] /* a */, &v28[0] /* b */, &v32[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #31" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op32,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v32[0] /* input */, &v33[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #32" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op33,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v33[0] /* input */, &v34[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #33" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op34,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v34[0] /* input */, &v35[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #34" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nc_f32(
    op35,
    196 /* batch size */,
    &v35[0] /* a */, &v32[0] /* b */, &v36[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #35" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op36,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v36[0] /* input */, &v37[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #36" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op37,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v37[0] /* input */, &v38[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #37" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op38,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v38[0] /* input */, &v39[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #38" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op39,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v39[0] /* input */, &v40[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #39" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op40,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v40[0] /* input */, &v41[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #40" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op41,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v41[0] /* input */, &v42[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #41" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nc_f32(
    op42,
    196 /* batch size */,
    &v42[0] /* a */, &v39[0] /* b */, &v43[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #42" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op43,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v43[0] /* input */, &v44[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #43" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op44,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v44[0] /* input */, &v45[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #44" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op45,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v45[0] /* input */, &v46[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #45" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nc_f32(
    op46,
    196 /* batch size */,
    &v46[0] /* a */, &v43[0] /* b */, &v47[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #46" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op47,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v47[0] /* input */, &v48[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #47" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op48,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v48[0] /* input */, &v49[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #48" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op49,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v49[0] /* input */, &v50[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #49" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op50,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v50[0] /* input */, &v51[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #50" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op51,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v51[0] /* input */, &v52[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #51" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op52,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v52[0] /* input */, &v53[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #52" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nc_f32(
    op53,
    49 /* batch size */,
    &v53[0] /* a */, &v50[0] /* b */, &v54[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #53" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op54,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v54[0] /* input */, &v55[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #54" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op55,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v55[0] /* input */, &v56[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #55" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op56,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v56[0] /* input */, &v57[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #56" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nc_f32(
    op57,
    49 /* batch size */,
    &v57[0] /* a */, &v54[0] /* b */, &v58[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #57" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op58,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v58[0] /* input */, &v59[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #58" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op59,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v59[0] /* input */, &v60[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #59" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op60,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v60[0] /* input */, &v61[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #60" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op61,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v61[0] /* input */, &v62[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #61" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op62,
    1 /* batch size */, 49 /* width */,
    &v62[0] /* input */, &v63[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #62" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op63,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v63[0] /* input */, &v64[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #63" << std::endl;
    return ExecutionPlan();
  }

  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wpessimizing-move"
  return operators;
  #pragma clang diagnostic pop
}

}  // namespace models
