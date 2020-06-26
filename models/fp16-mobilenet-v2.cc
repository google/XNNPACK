// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include <fp16/fp16.h>

#include "models/models.h"

namespace models {

ExecutionPlan FP16MobileNetV2(pthreadpool_t threadpool) {
  alignas(16) static uint16_t v0[150528];
  alignas(16) static uint16_t v1[401408];
  alignas(16) static uint16_t v2[401408];
  alignas(16) static uint16_t v3[200704];
  alignas(16) static uint16_t v4[1204224];
  alignas(16) static uint16_t v5[301056];
  alignas(16) static uint16_t v6[75264];
  alignas(16) static uint16_t v7[451584];
  alignas(16) static uint16_t v8[451584];
  alignas(16) static uint16_t v9[75264];
  alignas(16) static uint16_t v10[75264];
  alignas(16) static uint16_t v11[451584];
  alignas(16) static uint16_t v12[112896];
  alignas(16) static uint16_t v13[25088];
  alignas(16) static uint16_t v14[150528];
  alignas(16) static uint16_t v15[150528];
  alignas(16) static uint16_t v16[25088];
  alignas(16) static uint16_t v17[25088];
  alignas(16) static uint16_t v18[150528];
  alignas(16) static uint16_t v19[150528];
  alignas(16) static uint16_t v20[25088];
  alignas(16) static uint16_t v21[25088];
  alignas(16) static uint16_t v22[150528];
  alignas(16) static uint16_t v23[37632];
  alignas(16) static uint16_t v24[12544];
  alignas(16) static uint16_t v25[75264];
  alignas(16) static uint16_t v26[75264];
  alignas(16) static uint16_t v27[12544];
  alignas(16) static uint16_t v28[12544];
  alignas(16) static uint16_t v29[75264];
  alignas(16) static uint16_t v30[75264];
  alignas(16) static uint16_t v31[12544];
  alignas(16) static uint16_t v32[12544];
  alignas(16) static uint16_t v33[75264];
  alignas(16) static uint16_t v34[75264];
  alignas(16) static uint16_t v35[12544];
  alignas(16) static uint16_t v36[12544];
  alignas(16) static uint16_t v37[75264];
  alignas(16) static uint16_t v38[75264];
  alignas(16) static uint16_t v39[18816];
  alignas(16) static uint16_t v40[112896];
  alignas(16) static uint16_t v41[112896];
  alignas(16) static uint16_t v42[18816];
  alignas(16) static uint16_t v43[18816];
  alignas(16) static uint16_t v44[112896];
  alignas(16) static uint16_t v45[112896];
  alignas(16) static uint16_t v46[18816];
  alignas(16) static uint16_t v47[18816];
  alignas(16) static uint16_t v48[112896];
  alignas(16) static uint16_t v49[28224];
  alignas(16) static uint16_t v50[7840];
  alignas(16) static uint16_t v51[47040];
  alignas(16) static uint16_t v52[47040];
  alignas(16) static uint16_t v53[7840];
  alignas(16) static uint16_t v54[7840];
  alignas(16) static uint16_t v55[47040];
  alignas(16) static uint16_t v56[47040];
  alignas(16) static uint16_t v57[7840];
  alignas(16) static uint16_t v58[7840];
  alignas(16) static uint16_t v59[47040];
  alignas(16) static uint16_t v60[47040];
  alignas(16) static uint16_t v61[15680];
  alignas(16) static uint16_t v62[62720];
  alignas(16) static uint16_t v63[1280];
  alignas(16) static uint16_t v64[1001];
  alignas(16) static uint16_t w65[864];
  alignas(16) static uint16_t w66[32];
  alignas(16) static uint16_t w67[288];
  alignas(16) static uint16_t w68[32];
  alignas(16) static uint16_t w69[512];
  alignas(16) static uint16_t w70[16];
  alignas(16) static uint16_t w71[1536];
  alignas(16) static uint16_t w72[96];
  alignas(16) static uint16_t w73[864];
  alignas(16) static uint16_t w74[96];
  alignas(16) static uint16_t w75[2304];
  alignas(16) static uint16_t w76[24];
  alignas(16) static uint16_t w77[3456];
  alignas(16) static uint16_t w78[144];
  alignas(16) static uint16_t w79[1296];
  alignas(16) static uint16_t w80[144];
  alignas(16) static uint16_t w81[3456];
  alignas(16) static uint16_t w82[24];
  alignas(16) static uint16_t w83[3456];
  alignas(16) static uint16_t w84[144];
  alignas(16) static uint16_t w85[1296];
  alignas(16) static uint16_t w86[144];
  alignas(16) static uint16_t w87[4608];
  alignas(16) static uint16_t w88[32];
  alignas(16) static uint16_t w89[6144];
  alignas(16) static uint16_t w90[192];
  alignas(16) static uint16_t w91[1728];
  alignas(16) static uint16_t w92[192];
  alignas(16) static uint16_t w93[6144];
  alignas(16) static uint16_t w94[32];
  alignas(16) static uint16_t w95[6144];
  alignas(16) static uint16_t w96[192];
  alignas(16) static uint16_t w97[1728];
  alignas(16) static uint16_t w98[192];
  alignas(16) static uint16_t w99[6144];
  alignas(16) static uint16_t w100[32];
  alignas(16) static uint16_t w101[6144];
  alignas(16) static uint16_t w102[192];
  alignas(16) static uint16_t w103[1728];
  alignas(16) static uint16_t w104[192];
  alignas(16) static uint16_t w105[12288];
  alignas(16) static uint16_t w106[64];
  alignas(16) static uint16_t w107[24576];
  alignas(16) static uint16_t w108[384];
  alignas(16) static uint16_t w109[3456];
  alignas(16) static uint16_t w110[384];
  alignas(16) static uint16_t w111[24576];
  alignas(16) static uint16_t w112[64];
  alignas(16) static uint16_t w113[24576];
  alignas(16) static uint16_t w114[384];
  alignas(16) static uint16_t w115[3456];
  alignas(16) static uint16_t w116[384];
  alignas(16) static uint16_t w117[24576];
  alignas(16) static uint16_t w118[64];
  alignas(16) static uint16_t w119[24576];
  alignas(16) static uint16_t w120[384];
  alignas(16) static uint16_t w121[3456];
  alignas(16) static uint16_t w122[384];
  alignas(16) static uint16_t w123[24576];
  alignas(16) static uint16_t w124[64];
  alignas(16) static uint16_t w125[24576];
  alignas(16) static uint16_t w126[384];
  alignas(16) static uint16_t w127[3456];
  alignas(16) static uint16_t w128[384];
  alignas(16) static uint16_t w129[36864];
  alignas(16) static uint16_t w130[96];
  alignas(16) static uint16_t w131[55296];
  alignas(16) static uint16_t w132[576];
  alignas(16) static uint16_t w133[5184];
  alignas(16) static uint16_t w134[576];
  alignas(16) static uint16_t w135[55296];
  alignas(16) static uint16_t w136[96];
  alignas(16) static uint16_t w137[55296];
  alignas(16) static uint16_t w138[576];
  alignas(16) static uint16_t w139[5184];
  alignas(16) static uint16_t w140[576];
  alignas(16) static uint16_t w141[55296];
  alignas(16) static uint16_t w142[96];
  alignas(16) static uint16_t w143[55296];
  alignas(16) static uint16_t w144[576];
  alignas(16) static uint16_t w145[5184];
  alignas(16) static uint16_t w146[576];
  alignas(16) static uint16_t w147[92160];
  alignas(16) static uint16_t w148[160];
  alignas(16) static uint16_t w149[153600];
  alignas(16) static uint16_t w150[960];
  alignas(16) static uint16_t w151[8640];
  alignas(16) static uint16_t w152[960];
  alignas(16) static uint16_t w153[153600];
  alignas(16) static uint16_t w154[160];
  alignas(16) static uint16_t w155[153600];
  alignas(16) static uint16_t w156[960];
  alignas(16) static uint16_t w157[8640];
  alignas(16) static uint16_t w158[960];
  alignas(16) static uint16_t w159[153600];
  alignas(16) static uint16_t w160[160];
  alignas(16) static uint16_t w161[153600];
  alignas(16) static uint16_t w162[960];
  alignas(16) static uint16_t w163[8640];
  alignas(16) static uint16_t w164[960];
  alignas(16) static uint16_t w165[307200];
  alignas(16) static uint16_t w166[320];
  alignas(16) static uint16_t w167[409600];
  alignas(16) static uint16_t w168[1280];
  alignas(16) static uint16_t w169[1281280];
  alignas(16) static uint16_t w170[1001];

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);
  std::generate(v0, v0 + 150528, std::ref(f16rng));
  std::generate(v1, v1 + 401408, std::ref(f16rng));
  std::generate(v2, v2 + 401408, std::ref(f16rng));
  std::generate(v3, v3 + 200704, std::ref(f16rng));
  std::generate(v4, v4 + 1204224, std::ref(f16rng));
  std::generate(v5, v5 + 301056, std::ref(f16rng));
  std::generate(v6, v6 + 75264, std::ref(f16rng));
  std::generate(v7, v7 + 451584, std::ref(f16rng));
  std::generate(v8, v8 + 451584, std::ref(f16rng));
  std::generate(v9, v9 + 75264, std::ref(f16rng));
  std::generate(v10, v10 + 75264, std::ref(f16rng));
  std::generate(v11, v11 + 451584, std::ref(f16rng));
  std::generate(v12, v12 + 112896, std::ref(f16rng));
  std::generate(v13, v13 + 25088, std::ref(f16rng));
  std::generate(v14, v14 + 150528, std::ref(f16rng));
  std::generate(v15, v15 + 150528, std::ref(f16rng));
  std::generate(v16, v16 + 25088, std::ref(f16rng));
  std::generate(v17, v17 + 25088, std::ref(f16rng));
  std::generate(v18, v18 + 150528, std::ref(f16rng));
  std::generate(v19, v19 + 150528, std::ref(f16rng));
  std::generate(v20, v20 + 25088, std::ref(f16rng));
  std::generate(v21, v21 + 25088, std::ref(f16rng));
  std::generate(v22, v22 + 150528, std::ref(f16rng));
  std::generate(v23, v23 + 37632, std::ref(f16rng));
  std::generate(v24, v24 + 12544, std::ref(f16rng));
  std::generate(v25, v25 + 75264, std::ref(f16rng));
  std::generate(v26, v26 + 75264, std::ref(f16rng));
  std::generate(v27, v27 + 12544, std::ref(f16rng));
  std::generate(v28, v28 + 12544, std::ref(f16rng));
  std::generate(v29, v29 + 75264, std::ref(f16rng));
  std::generate(v30, v30 + 75264, std::ref(f16rng));
  std::generate(v31, v31 + 12544, std::ref(f16rng));
  std::generate(v32, v32 + 12544, std::ref(f16rng));
  std::generate(v33, v33 + 75264, std::ref(f16rng));
  std::generate(v34, v34 + 75264, std::ref(f16rng));
  std::generate(v35, v35 + 12544, std::ref(f16rng));
  std::generate(v36, v36 + 12544, std::ref(f16rng));
  std::generate(v37, v37 + 75264, std::ref(f16rng));
  std::generate(v38, v38 + 75264, std::ref(f16rng));
  std::generate(v39, v39 + 18816, std::ref(f16rng));
  std::generate(v40, v40 + 112896, std::ref(f16rng));
  std::generate(v41, v41 + 112896, std::ref(f16rng));
  std::generate(v42, v42 + 18816, std::ref(f16rng));
  std::generate(v43, v43 + 18816, std::ref(f16rng));
  std::generate(v44, v44 + 112896, std::ref(f16rng));
  std::generate(v45, v45 + 112896, std::ref(f16rng));
  std::generate(v46, v46 + 18816, std::ref(f16rng));
  std::generate(v47, v47 + 18816, std::ref(f16rng));
  std::generate(v48, v48 + 112896, std::ref(f16rng));
  std::generate(v49, v49 + 28224, std::ref(f16rng));
  std::generate(v50, v50 + 7840, std::ref(f16rng));
  std::generate(v51, v51 + 47040, std::ref(f16rng));
  std::generate(v52, v52 + 47040, std::ref(f16rng));
  std::generate(v53, v53 + 7840, std::ref(f16rng));
  std::generate(v54, v54 + 7840, std::ref(f16rng));
  std::generate(v55, v55 + 47040, std::ref(f16rng));
  std::generate(v56, v56 + 47040, std::ref(f16rng));
  std::generate(v57, v57 + 7840, std::ref(f16rng));
  std::generate(v58, v58 + 7840, std::ref(f16rng));
  std::generate(v59, v59 + 47040, std::ref(f16rng));
  std::generate(v60, v60 + 47040, std::ref(f16rng));
  std::generate(v61, v61 + 15680, std::ref(f16rng));
  std::generate(v62, v62 + 62720, std::ref(f16rng));
  std::generate(v63, v63 + 1280, std::ref(f16rng));
  std::generate(v64, v64 + 1001, std::ref(f16rng));
  std::generate(w65, w65 + 864, std::ref(f16rng));
  std::generate(w66, w66 + 32, std::ref(f16rng));
  std::generate(w67, w67 + 288, std::ref(f16rng));
  std::generate(w68, w68 + 32, std::ref(f16rng));
  std::generate(w69, w69 + 512, std::ref(f16rng));
  std::generate(w70, w70 + 16, std::ref(f16rng));
  std::generate(w71, w71 + 1536, std::ref(f16rng));
  std::generate(w72, w72 + 96, std::ref(f16rng));
  std::generate(w73, w73 + 864, std::ref(f16rng));
  std::generate(w74, w74 + 96, std::ref(f16rng));
  std::generate(w75, w75 + 2304, std::ref(f16rng));
  std::generate(w76, w76 + 24, std::ref(f16rng));
  std::generate(w77, w77 + 3456, std::ref(f16rng));
  std::generate(w78, w78 + 144, std::ref(f16rng));
  std::generate(w79, w79 + 1296, std::ref(f16rng));
  std::generate(w80, w80 + 144, std::ref(f16rng));
  std::generate(w81, w81 + 3456, std::ref(f16rng));
  std::generate(w82, w82 + 24, std::ref(f16rng));
  std::generate(w83, w83 + 3456, std::ref(f16rng));
  std::generate(w84, w84 + 144, std::ref(f16rng));
  std::generate(w85, w85 + 1296, std::ref(f16rng));
  std::generate(w86, w86 + 144, std::ref(f16rng));
  std::generate(w87, w87 + 4608, std::ref(f16rng));
  std::generate(w88, w88 + 32, std::ref(f16rng));
  std::generate(w89, w89 + 6144, std::ref(f16rng));
  std::generate(w90, w90 + 192, std::ref(f16rng));
  std::generate(w91, w91 + 1728, std::ref(f16rng));
  std::generate(w92, w92 + 192, std::ref(f16rng));
  std::generate(w93, w93 + 6144, std::ref(f16rng));
  std::generate(w94, w94 + 32, std::ref(f16rng));
  std::generate(w95, w95 + 6144, std::ref(f16rng));
  std::generate(w96, w96 + 192, std::ref(f16rng));
  std::generate(w97, w97 + 1728, std::ref(f16rng));
  std::generate(w98, w98 + 192, std::ref(f16rng));
  std::generate(w99, w99 + 6144, std::ref(f16rng));
  std::generate(w100, w100 + 32, std::ref(f16rng));
  std::generate(w101, w101 + 6144, std::ref(f16rng));
  std::generate(w102, w102 + 192, std::ref(f16rng));
  std::generate(w103, w103 + 1728, std::ref(f16rng));
  std::generate(w104, w104 + 192, std::ref(f16rng));
  std::generate(w105, w105 + 12288, std::ref(f16rng));
  std::generate(w106, w106 + 64, std::ref(f16rng));
  std::generate(w107, w107 + 24576, std::ref(f16rng));
  std::generate(w108, w108 + 384, std::ref(f16rng));
  std::generate(w109, w109 + 3456, std::ref(f16rng));
  std::generate(w110, w110 + 384, std::ref(f16rng));
  std::generate(w111, w111 + 24576, std::ref(f16rng));
  std::generate(w112, w112 + 64, std::ref(f16rng));
  std::generate(w113, w113 + 24576, std::ref(f16rng));
  std::generate(w114, w114 + 384, std::ref(f16rng));
  std::generate(w115, w115 + 3456, std::ref(f16rng));
  std::generate(w116, w116 + 384, std::ref(f16rng));
  std::generate(w117, w117 + 24576, std::ref(f16rng));
  std::generate(w118, w118 + 64, std::ref(f16rng));
  std::generate(w119, w119 + 24576, std::ref(f16rng));
  std::generate(w120, w120 + 384, std::ref(f16rng));
  std::generate(w121, w121 + 3456, std::ref(f16rng));
  std::generate(w122, w122 + 384, std::ref(f16rng));
  std::generate(w123, w123 + 24576, std::ref(f16rng));
  std::generate(w124, w124 + 64, std::ref(f16rng));
  std::generate(w125, w125 + 24576, std::ref(f16rng));
  std::generate(w126, w126 + 384, std::ref(f16rng));
  std::generate(w127, w127 + 3456, std::ref(f16rng));
  std::generate(w128, w128 + 384, std::ref(f16rng));
  std::generate(w129, w129 + 36864, std::ref(f16rng));
  std::generate(w130, w130 + 96, std::ref(f16rng));
  std::generate(w131, w131 + 55296, std::ref(f16rng));
  std::generate(w132, w132 + 576, std::ref(f16rng));
  std::generate(w133, w133 + 5184, std::ref(f16rng));
  std::generate(w134, w134 + 576, std::ref(f16rng));
  std::generate(w135, w135 + 55296, std::ref(f16rng));
  std::generate(w136, w136 + 96, std::ref(f16rng));
  std::generate(w137, w137 + 55296, std::ref(f16rng));
  std::generate(w138, w138 + 576, std::ref(f16rng));
  std::generate(w139, w139 + 5184, std::ref(f16rng));
  std::generate(w140, w140 + 576, std::ref(f16rng));
  std::generate(w141, w141 + 55296, std::ref(f16rng));
  std::generate(w142, w142 + 96, std::ref(f16rng));
  std::generate(w143, w143 + 55296, std::ref(f16rng));
  std::generate(w144, w144 + 576, std::ref(f16rng));
  std::generate(w145, w145 + 5184, std::ref(f16rng));
  std::generate(w146, w146 + 576, std::ref(f16rng));
  std::generate(w147, w147 + 92160, std::ref(f16rng));
  std::generate(w148, w148 + 160, std::ref(f16rng));
  std::generate(w149, w149 + 153600, std::ref(f16rng));
  std::generate(w150, w150 + 960, std::ref(f16rng));
  std::generate(w151, w151 + 8640, std::ref(f16rng));
  std::generate(w152, w152 + 960, std::ref(f16rng));
  std::generate(w153, w153 + 153600, std::ref(f16rng));
  std::generate(w154, w154 + 160, std::ref(f16rng));
  std::generate(w155, w155 + 153600, std::ref(f16rng));
  std::generate(w156, w156 + 960, std::ref(f16rng));
  std::generate(w157, w157 + 8640, std::ref(f16rng));
  std::generate(w158, w158 + 960, std::ref(f16rng));
  std::generate(w159, w159 + 153600, std::ref(f16rng));
  std::generate(w160, w160 + 160, std::ref(f16rng));
  std::generate(w161, w161 + 153600, std::ref(f16rng));
  std::generate(w162, w162 + 960, std::ref(f16rng));
  std::generate(w163, w163 + 8640, std::ref(f16rng));
  std::generate(w164, w164 + 960, std::ref(f16rng));
  std::generate(w165, w165 + 307200, std::ref(f16rng));
  std::generate(w166, w166 + 320, std::ref(f16rng));
  std::generate(w167, w167 + 409600, std::ref(f16rng));
  std::generate(w168, w168 + 1280, std::ref(f16rng));
  std::generate(w169, w169 + 1281280, std::ref(f16rng));
  std::generate(w170, w170 + 1001, std::ref(f16rng));

  ExecutionPlan operators;
  xnn_status status;

  xnn_operator_t op0 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #9" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op9, xnn_delete_operator);

  xnn_operator_t op10 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #16" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op16, xnn_delete_operator);

  xnn_operator_t op17 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op20);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #20" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op20, xnn_delete_operator);

  xnn_operator_t op21 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #27" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op27, xnn_delete_operator);

  xnn_operator_t op28 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op31);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #31" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op31, xnn_delete_operator);

  xnn_operator_t op32 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op35);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #35" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op35, xnn_delete_operator);

  xnn_operator_t op36 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op42);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #42" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op42, xnn_delete_operator);

  xnn_operator_t op43 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op46);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #46" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op46, xnn_delete_operator);

  xnn_operator_t op47 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op53);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #53" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op53, xnn_delete_operator);

  xnn_operator_t op54 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op57);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #57" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op57, xnn_delete_operator);

  xnn_operator_t op58 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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
  status = xnn_create_global_average_pooling_nwc_f16(
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
  status = xnn_create_convolution2d_nhwc_f16(
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



  status = xnn_setup_convolution2d_nhwc_f16(
    op0,
    1 /* batch size */, 224 /* input height */, 224 /* input width */,
    v0 /* input */, v1 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op1,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v1 /* input */, v2 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #1" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op2,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v2 /* input */, v3 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op3,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v3 /* input */, v4 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #3" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op4,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v4 /* input */, v5 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op5,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v5 /* input */, v6 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #5" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op6,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v6 /* input */, v7 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op7,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v7 /* input */, v8 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op8,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v8 /* input */, v9 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #8" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 56, 56, 24 };
    const size_t b_shape[] = { 1, 56, 56, 24 };
    status = xnn_setup_add_nd_f16(
      op9,
      4, a_shape, 4, b_shape,
      v9 /* a */, v6 /* b */, v10 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op10,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v10 /* input */, v11 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #10" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op11,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v11 /* input */, v12 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op12,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v12 /* input */, v13 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op13,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v13 /* input */, v14 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #13" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op14,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v14 /* input */, v15 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op15,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v15 /* input */, v16 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #15" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 28, 28, 32 };
    const size_t b_shape[] = { 1, 28, 28, 32 };
    status = xnn_setup_add_nd_f16(
      op16,
      4, a_shape, 4, b_shape,
      v16 /* a */, v13 /* b */, v17 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #16" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op17,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v17 /* input */, v18 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op18,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v18 /* input */, v19 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op19,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v19 /* input */, v20 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #19" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 28, 28, 32 };
    const size_t b_shape[] = { 1, 28, 28, 32 };
    status = xnn_setup_add_nd_f16(
      op20,
      4, a_shape, 4, b_shape,
      v20 /* a */, v17 /* b */, v21 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op21,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v21 /* input */, v22 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #21" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op22,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v22 /* input */, v23 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op23,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v23 /* input */, v24 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #23" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op24,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v24 /* input */, v25 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op25,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v25 /* input */, v26 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #25" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op26,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v26 /* input */, v27 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #26" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 64 };
    const size_t b_shape[] = { 1, 14, 14, 64 };
    status = xnn_setup_add_nd_f16(
      op27,
      4, a_shape, 4, b_shape,
      v27 /* a */, v24 /* b */, v28 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op28,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v28 /* input */, v29 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #28" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op29,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v29 /* input */, v30 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #29" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op30,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v30 /* input */, v31 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #30" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 64 };
    const size_t b_shape[] = { 1, 14, 14, 64 };
    status = xnn_setup_add_nd_f16(
      op31,
      4, a_shape, 4, b_shape,
      v31 /* a */, v28 /* b */, v32 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #31" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op32,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v32 /* input */, v33 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #32" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op33,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v33 /* input */, v34 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #33" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op34,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v34 /* input */, v35 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #34" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 64 };
    const size_t b_shape[] = { 1, 14, 14, 64 };
    status = xnn_setup_add_nd_f16(
      op35,
      4, a_shape, 4, b_shape,
      v35 /* a */, v32 /* b */, v36 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #35" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op36,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v36 /* input */, v37 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #36" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op37,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v37 /* input */, v38 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #37" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op38,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v38 /* input */, v39 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #38" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op39,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v39 /* input */, v40 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #39" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op40,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v40 /* input */, v41 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #40" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op41,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v41 /* input */, v42 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #41" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 96 };
    const size_t b_shape[] = { 1, 14, 14, 96 };
    status = xnn_setup_add_nd_f16(
      op42,
      4, a_shape, 4, b_shape,
      v42 /* a */, v39 /* b */, v43 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #42" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op43,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v43 /* input */, v44 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #43" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op44,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v44 /* input */, v45 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #44" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op45,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v45 /* input */, v46 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #45" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 96 };
    const size_t b_shape[] = { 1, 14, 14, 96 };
    status = xnn_setup_add_nd_f16(
      op46,
      4, a_shape, 4, b_shape,
      v46 /* a */, v43 /* b */, v47 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #46" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op47,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v47 /* input */, v48 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #47" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op48,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v48 /* input */, v49 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #48" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op49,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v49 /* input */, v50 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #49" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op50,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v50 /* input */, v51 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #50" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op51,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v51 /* input */, v52 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #51" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op52,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v52 /* input */, v53 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #52" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 160 };
    const size_t b_shape[] = { 1, 7, 7, 160 };
    status = xnn_setup_add_nd_f16(
      op53,
      4, a_shape, 4, b_shape,
      v53 /* a */, v50 /* b */, v54 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #53" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op54,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v54 /* input */, v55 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #54" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op55,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v55 /* input */, v56 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #55" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op56,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v56 /* input */, v57 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #56" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 160 };
    const size_t b_shape[] = { 1, 7, 7, 160 };
    status = xnn_setup_add_nd_f16(
      op57,
      4, a_shape, 4, b_shape,
      v57 /* a */, v54 /* b */, v58 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #57" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op58,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v58 /* input */, v59 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #58" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op59,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v59 /* input */, v60 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #59" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op60,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v60 /* input */, v61 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #60" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op61,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v61 /* input */, v62 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #61" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f16(
    op62,
    1 /* batch size */, 49 /* width */,
    v62 /* input */, v63 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #62" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op63,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v63 /* input */, v64 /* output */,
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
