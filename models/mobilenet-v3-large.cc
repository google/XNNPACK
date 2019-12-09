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

ExecutionPlan MobileNetV3Large(pthreadpool_t threadpool) {
  alignas(16) static float v0[150528];
  alignas(16) static float v1[200704];
  alignas(16) static float v2[200704];
  alignas(16) static float v3[200704];
  alignas(16) static float v4[200704];
  alignas(16) static float v5[200704];
  alignas(16) static float v6[802816];
  alignas(16) static float v7[200704];
  alignas(16) static float v8[75264];
  alignas(16) static float v9[225792];
  alignas(16) static float v10[225792];
  alignas(16) static float v11[75264];
  alignas(16) static float v12[75264];
  alignas(16) static float v13[225792];
  alignas(16) static float v14[56448];
  alignas(16) static float v15[72];
  alignas(16) static float v16[24];
  alignas(16) static float v17[72];
  alignas(16) static float v18[56448];
  alignas(16) static float v19[31360];
  alignas(16) static float v20[94080];
  alignas(16) static float v21[94080];
  alignas(16) static float v22[120];
  alignas(16) static float v23[32];
  alignas(16) static float v24[120];
  alignas(16) static float v25[94080];
  alignas(16) static float v26[31360];
  alignas(16) static float v27[31360];
  alignas(16) static float v28[94080];
  alignas(16) static float v29[94080];
  alignas(16) static float v30[120];
  alignas(16) static float v31[32];
  alignas(16) static float v32[120];
  alignas(16) static float v33[94080];
  alignas(16) static float v34[31360];
  alignas(16) static float v35[31360];
  alignas(16) static float v36[188160];
  alignas(16) static float v37[188160];
  alignas(16) static float v38[47040];
  alignas(16) static float v39[47040];
  alignas(16) static float v40[15680];
  alignas(16) static float v41[39200];
  alignas(16) static float v42[39200];
  alignas(16) static float v43[39200];
  alignas(16) static float v44[39200];
  alignas(16) static float v45[15680];
  alignas(16) static float v46[15680];
  alignas(16) static float v47[36064];
  alignas(16) static float v48[36064];
  alignas(16) static float v49[36064];
  alignas(16) static float v50[36064];
  alignas(16) static float v51[15680];
  alignas(16) static float v52[15680];
  alignas(16) static float v53[36064];
  alignas(16) static float v54[36064];
  alignas(16) static float v55[36064];
  alignas(16) static float v56[36064];
  alignas(16) static float v57[15680];
  alignas(16) static float v58[15680];
  alignas(16) static float v59[94080];
  alignas(16) static float v60[94080];
  alignas(16) static float v61[94080];
  alignas(16) static float v62[94080];
  alignas(16) static float v63[480];
  alignas(16) static float v64[120];
  alignas(16) static float v65[480];
  alignas(16) static float v66[94080];
  alignas(16) static float v67[21952];
  alignas(16) static float v68[131712];
  alignas(16) static float v69[131712];
  alignas(16) static float v70[131712];
  alignas(16) static float v71[131712];
  alignas(16) static float v72[672];
  alignas(16) static float v73[168];
  alignas(16) static float v74[672];
  alignas(16) static float v75[131712];
  alignas(16) static float v76[21952];
  alignas(16) static float v77[21952];
  alignas(16) static float v78[131712];
  alignas(16) static float v79[131712];
  alignas(16) static float v80[32928];
  alignas(16) static float v81[32928];
  alignas(16) static float v82[672];
  alignas(16) static float v83[168];
  alignas(16) static float v84[672];
  alignas(16) static float v85[32928];
  alignas(16) static float v86[7840];
  alignas(16) static float v87[47040];
  alignas(16) static float v88[47040];
  alignas(16) static float v89[47040];
  alignas(16) static float v90[47040];
  alignas(16) static float v91[960];
  alignas(16) static float v92[240];
  alignas(16) static float v93[960];
  alignas(16) static float v94[47040];
  alignas(16) static float v95[7840];
  alignas(16) static float v96[7840];
  alignas(16) static float v97[47040];
  alignas(16) static float v98[47040];
  alignas(16) static float v99[47040];
  alignas(16) static float v100[47040];
  alignas(16) static float v101[960];
  alignas(16) static float v102[240];
  alignas(16) static float v103[960];
  alignas(16) static float v104[47040];
  alignas(16) static float v105[7840];
  alignas(16) static float v106[7840];
  alignas(16) static float v107[47040];
  alignas(16) static float v108[47040];
  alignas(16) static float v109[960];
  alignas(16) static float v110[1280];
  alignas(16) static float v111[1280];
  alignas(16) static float v112[1280];
  alignas(16) static float v113[1001];
  alignas(16) static float w114[432];
  alignas(16) static float w115[16];
  alignas(16) static float w116[144];
  alignas(16) static float w117[16];
  alignas(16) static float w118[256];
  alignas(16) static float w119[16];
  alignas(16) static float w120[1024];
  alignas(16) static float w121[64];
  alignas(16) static float w122[576];
  alignas(16) static float w123[64];
  alignas(16) static float w124[1536];
  alignas(16) static float w125[24];
  alignas(16) static float w126[1728];
  alignas(16) static float w127[72];
  alignas(16) static float w128[648];
  alignas(16) static float w129[72];
  alignas(16) static float w130[1728];
  alignas(16) static float w131[24];
  alignas(16) static float w132[1728];
  alignas(16) static float w133[72];
  alignas(16) static float w134[1800];
  alignas(16) static float w135[72];
  alignas(16) static float w136[1728];
  alignas(16) static float w137[24];
  alignas(16) static float w138[1728];
  alignas(16) static float w139[72];
  alignas(16) static float w140[2880];
  alignas(16) static float w141[40];
  alignas(16) static float w142[4800];
  alignas(16) static float w143[120];
  alignas(16) static float w144[3000];
  alignas(16) static float w145[120];
  alignas(16) static float w146[3840];
  alignas(16) static float w147[32];
  alignas(16) static float w148[3840];
  alignas(16) static float w149[120];
  alignas(16) static float w150[4800];
  alignas(16) static float w151[40];
  alignas(16) static float w152[4800];
  alignas(16) static float w153[120];
  alignas(16) static float w154[3000];
  alignas(16) static float w155[120];
  alignas(16) static float w156[3840];
  alignas(16) static float w157[32];
  alignas(16) static float w158[3840];
  alignas(16) static float w159[120];
  alignas(16) static float w160[4800];
  alignas(16) static float w161[40];
  alignas(16) static float w162[9600];
  alignas(16) static float w163[240];
  alignas(16) static float w164[2160];
  alignas(16) static float w165[240];
  alignas(16) static float w166[19200];
  alignas(16) static float w167[80];
  alignas(16) static float w168[16000];
  alignas(16) static float w169[200];
  alignas(16) static float w170[1800];
  alignas(16) static float w171[200];
  alignas(16) static float w172[16000];
  alignas(16) static float w173[80];
  alignas(16) static float w174[14720];
  alignas(16) static float w175[184];
  alignas(16) static float w176[1656];
  alignas(16) static float w177[184];
  alignas(16) static float w178[14720];
  alignas(16) static float w179[80];
  alignas(16) static float w180[14720];
  alignas(16) static float w181[184];
  alignas(16) static float w182[1656];
  alignas(16) static float w183[184];
  alignas(16) static float w184[14720];
  alignas(16) static float w185[80];
  alignas(16) static float w186[38400];
  alignas(16) static float w187[480];
  alignas(16) static float w188[4320];
  alignas(16) static float w189[480];
  alignas(16) static float w190[57600];
  alignas(16) static float w191[120];
  alignas(16) static float w192[57600];
  alignas(16) static float w193[480];
  alignas(16) static float w194[53760];
  alignas(16) static float w195[112];
  alignas(16) static float w196[75264];
  alignas(16) static float w197[672];
  alignas(16) static float w198[6048];
  alignas(16) static float w199[672];
  alignas(16) static float w200[112896];
  alignas(16) static float w201[168];
  alignas(16) static float w202[112896];
  alignas(16) static float w203[672];
  alignas(16) static float w204[75264];
  alignas(16) static float w205[112];
  alignas(16) static float w206[75264];
  alignas(16) static float w207[672];
  alignas(16) static float w208[16800];
  alignas(16) static float w209[672];
  alignas(16) static float w210[112896];
  alignas(16) static float w211[168];
  alignas(16) static float w212[112896];
  alignas(16) static float w213[672];
  alignas(16) static float w214[107520];
  alignas(16) static float w215[160];
  alignas(16) static float w216[153600];
  alignas(16) static float w217[960];
  alignas(16) static float w218[24000];
  alignas(16) static float w219[960];
  alignas(16) static float w220[230400];
  alignas(16) static float w221[240];
  alignas(16) static float w222[230400];
  alignas(16) static float w223[960];
  alignas(16) static float w224[153600];
  alignas(16) static float w225[160];
  alignas(16) static float w226[153600];
  alignas(16) static float w227[960];
  alignas(16) static float w228[24000];
  alignas(16) static float w229[960];
  alignas(16) static float w230[230400];
  alignas(16) static float w231[240];
  alignas(16) static float w232[230400];
  alignas(16) static float w233[960];
  alignas(16) static float w234[153600];
  alignas(16) static float w235[160];
  alignas(16) static float w236[153600];
  alignas(16) static float w237[960];
  alignas(16) static float w238[1228800];
  alignas(16) static float w239[1280];
  alignas(16) static float w240[1281280];
  alignas(16) static float w241[1001];

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), rng);
  std::generate(v0, v0 + 150528, std::ref(f32rng));
  std::generate(w114, w114 + 432, std::ref(f32rng));
  std::generate(w115, w115 + 16, std::ref(f32rng));
  std::generate(w116, w116 + 144, std::ref(f32rng));
  std::generate(w117, w117 + 16, std::ref(f32rng));
  std::generate(w118, w118 + 256, std::ref(f32rng));
  std::generate(w119, w119 + 16, std::ref(f32rng));
  std::generate(w120, w120 + 1024, std::ref(f32rng));
  std::generate(w121, w121 + 64, std::ref(f32rng));
  std::generate(w122, w122 + 576, std::ref(f32rng));
  std::generate(w123, w123 + 64, std::ref(f32rng));
  std::generate(w124, w124 + 1536, std::ref(f32rng));
  std::generate(w125, w125 + 24, std::ref(f32rng));
  std::generate(w126, w126 + 1728, std::ref(f32rng));
  std::generate(w127, w127 + 72, std::ref(f32rng));
  std::generate(w128, w128 + 648, std::ref(f32rng));
  std::generate(w129, w129 + 72, std::ref(f32rng));
  std::generate(w130, w130 + 1728, std::ref(f32rng));
  std::generate(w131, w131 + 24, std::ref(f32rng));
  std::generate(w132, w132 + 1728, std::ref(f32rng));
  std::generate(w133, w133 + 72, std::ref(f32rng));
  std::generate(w134, w134 + 1800, std::ref(f32rng));
  std::generate(w135, w135 + 72, std::ref(f32rng));
  std::generate(w136, w136 + 1728, std::ref(f32rng));
  std::generate(w137, w137 + 24, std::ref(f32rng));
  std::generate(w138, w138 + 1728, std::ref(f32rng));
  std::generate(w139, w139 + 72, std::ref(f32rng));
  std::generate(w140, w140 + 2880, std::ref(f32rng));
  std::generate(w141, w141 + 40, std::ref(f32rng));
  std::generate(w142, w142 + 4800, std::ref(f32rng));
  std::generate(w143, w143 + 120, std::ref(f32rng));
  std::generate(w144, w144 + 3000, std::ref(f32rng));
  std::generate(w145, w145 + 120, std::ref(f32rng));
  std::generate(w146, w146 + 3840, std::ref(f32rng));
  std::generate(w147, w147 + 32, std::ref(f32rng));
  std::generate(w148, w148 + 3840, std::ref(f32rng));
  std::generate(w149, w149 + 120, std::ref(f32rng));
  std::generate(w150, w150 + 4800, std::ref(f32rng));
  std::generate(w151, w151 + 40, std::ref(f32rng));
  std::generate(w152, w152 + 4800, std::ref(f32rng));
  std::generate(w153, w153 + 120, std::ref(f32rng));
  std::generate(w154, w154 + 3000, std::ref(f32rng));
  std::generate(w155, w155 + 120, std::ref(f32rng));
  std::generate(w156, w156 + 3840, std::ref(f32rng));
  std::generate(w157, w157 + 32, std::ref(f32rng));
  std::generate(w158, w158 + 3840, std::ref(f32rng));
  std::generate(w159, w159 + 120, std::ref(f32rng));
  std::generate(w160, w160 + 4800, std::ref(f32rng));
  std::generate(w161, w161 + 40, std::ref(f32rng));
  std::generate(w162, w162 + 9600, std::ref(f32rng));
  std::generate(w163, w163 + 240, std::ref(f32rng));
  std::generate(w164, w164 + 2160, std::ref(f32rng));
  std::generate(w165, w165 + 240, std::ref(f32rng));
  std::generate(w166, w166 + 19200, std::ref(f32rng));
  std::generate(w167, w167 + 80, std::ref(f32rng));
  std::generate(w168, w168 + 16000, std::ref(f32rng));
  std::generate(w169, w169 + 200, std::ref(f32rng));
  std::generate(w170, w170 + 1800, std::ref(f32rng));
  std::generate(w171, w171 + 200, std::ref(f32rng));
  std::generate(w172, w172 + 16000, std::ref(f32rng));
  std::generate(w173, w173 + 80, std::ref(f32rng));
  std::generate(w174, w174 + 14720, std::ref(f32rng));
  std::generate(w175, w175 + 184, std::ref(f32rng));
  std::generate(w176, w176 + 1656, std::ref(f32rng));
  std::generate(w177, w177 + 184, std::ref(f32rng));
  std::generate(w178, w178 + 14720, std::ref(f32rng));
  std::generate(w179, w179 + 80, std::ref(f32rng));
  std::generate(w180, w180 + 14720, std::ref(f32rng));
  std::generate(w181, w181 + 184, std::ref(f32rng));
  std::generate(w182, w182 + 1656, std::ref(f32rng));
  std::generate(w183, w183 + 184, std::ref(f32rng));
  std::generate(w184, w184 + 14720, std::ref(f32rng));
  std::generate(w185, w185 + 80, std::ref(f32rng));
  std::generate(w186, w186 + 38400, std::ref(f32rng));
  std::generate(w187, w187 + 480, std::ref(f32rng));
  std::generate(w188, w188 + 4320, std::ref(f32rng));
  std::generate(w189, w189 + 480, std::ref(f32rng));
  std::generate(w190, w190 + 57600, std::ref(f32rng));
  std::generate(w191, w191 + 120, std::ref(f32rng));
  std::generate(w192, w192 + 57600, std::ref(f32rng));
  std::generate(w193, w193 + 480, std::ref(f32rng));
  std::generate(w194, w194 + 53760, std::ref(f32rng));
  std::generate(w195, w195 + 112, std::ref(f32rng));
  std::generate(w196, w196 + 75264, std::ref(f32rng));
  std::generate(w197, w197 + 672, std::ref(f32rng));
  std::generate(w198, w198 + 6048, std::ref(f32rng));
  std::generate(w199, w199 + 672, std::ref(f32rng));
  std::generate(w200, w200 + 112896, std::ref(f32rng));
  std::generate(w201, w201 + 168, std::ref(f32rng));
  std::generate(w202, w202 + 112896, std::ref(f32rng));
  std::generate(w203, w203 + 672, std::ref(f32rng));
  std::generate(w204, w204 + 75264, std::ref(f32rng));
  std::generate(w205, w205 + 112, std::ref(f32rng));
  std::generate(w206, w206 + 75264, std::ref(f32rng));
  std::generate(w207, w207 + 672, std::ref(f32rng));
  std::generate(w208, w208 + 16800, std::ref(f32rng));
  std::generate(w209, w209 + 672, std::ref(f32rng));
  std::generate(w210, w210 + 112896, std::ref(f32rng));
  std::generate(w211, w211 + 168, std::ref(f32rng));
  std::generate(w212, w212 + 112896, std::ref(f32rng));
  std::generate(w213, w213 + 672, std::ref(f32rng));
  std::generate(w214, w214 + 107520, std::ref(f32rng));
  std::generate(w215, w215 + 160, std::ref(f32rng));
  std::generate(w216, w216 + 153600, std::ref(f32rng));
  std::generate(w217, w217 + 960, std::ref(f32rng));
  std::generate(w218, w218 + 24000, std::ref(f32rng));
  std::generate(w219, w219 + 960, std::ref(f32rng));
  std::generate(w220, w220 + 230400, std::ref(f32rng));
  std::generate(w221, w221 + 240, std::ref(f32rng));
  std::generate(w222, w222 + 230400, std::ref(f32rng));
  std::generate(w223, w223 + 960, std::ref(f32rng));
  std::generate(w224, w224 + 153600, std::ref(f32rng));
  std::generate(w225, w225 + 160, std::ref(f32rng));
  std::generate(w226, w226 + 153600, std::ref(f32rng));
  std::generate(w227, w227 + 960, std::ref(f32rng));
  std::generate(w228, w228 + 24000, std::ref(f32rng));
  std::generate(w229, w229 + 960, std::ref(f32rng));
  std::generate(w230, w230 + 230400, std::ref(f32rng));
  std::generate(w231, w231 + 240, std::ref(f32rng));
  std::generate(w232, w232 + 230400, std::ref(f32rng));
  std::generate(w233, w233 + 960, std::ref(f32rng));
  std::generate(w234, w234 + 153600, std::ref(f32rng));
  std::generate(w235, w235 + 160, std::ref(f32rng));
  std::generate(w236, w236 + 153600, std::ref(f32rng));
  std::generate(w237, w237 + 960, std::ref(f32rng));
  std::generate(w238, w238 + 1228800, std::ref(f32rng));
  std::generate(w239, w239 + 1280, std::ref(f32rng));
  std::generate(w240, w240 + 1281280, std::ref(f32rng));
  std::generate(w241, w241 + 1001, std::ref(f32rng));

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
    16 /* output_channels_per_group */,
    3 /* input pixel stride */,
    16 /* output pixel stride */,
    w114, w115,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #0" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op0, xnn_delete_operator);

  xnn_operator_t op1 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    16 /* channels */,
    16 /* input stride */,
    16 /* output stride */,
    0 /* flags */,
    &op1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #1" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op1, xnn_delete_operator);

  xnn_operator_t op2 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    16 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    16 /* input pixel stride */,
    16 /* output pixel stride */,
    w116, w117,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    16 /* output_channels_per_group */,
    16 /* input pixel stride */,
    16 /* output pixel stride */,
    w118, w119,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #3" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op3, xnn_delete_operator);

  xnn_operator_t op4 = nullptr;
  status = xnn_create_add_nc_f32(
    16 /* channels */,
    16 /* a stride */,
    16 /* b stride */,
    16 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    16 /* input channels per group */,
    64 /* output_channels_per_group */,
    16 /* input pixel stride */,
    64 /* output pixel stride */,
    w120, w121,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #5" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op5, xnn_delete_operator);

  xnn_operator_t op6 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    64 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    64 /* input pixel stride */,
    64 /* output pixel stride */,
    w122, w123,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op6);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #6" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op6, xnn_delete_operator);

  xnn_operator_t op7 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    64 /* input channels per group */,
    24 /* output_channels_per_group */,
    64 /* input pixel stride */,
    24 /* output pixel stride */,
    w124, w125,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    24 /* input channels per group */,
    72 /* output_channels_per_group */,
    24 /* input pixel stride */,
    72 /* output pixel stride */,
    w126, w127,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #8" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op8, xnn_delete_operator);

  xnn_operator_t op9 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    72 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    72 /* input pixel stride */,
    72 /* output pixel stride */,
    w128, w129,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    72 /* input channels per group */,
    24 /* output_channels_per_group */,
    72 /* input pixel stride */,
    24 /* output pixel stride */,
    w130, w131,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #10" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op10, xnn_delete_operator);

  xnn_operator_t op11 = nullptr;
  status = xnn_create_add_nc_f32(
    24 /* channels */,
    24 /* a stride */,
    24 /* b stride */,
    24 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    24 /* input channels per group */,
    72 /* output_channels_per_group */,
    24 /* input pixel stride */,
    72 /* output pixel stride */,
    w132, w133,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #12" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op12, xnn_delete_operator);

  xnn_operator_t op13 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 1 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    72 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    72 /* input pixel stride */,
    72 /* output pixel stride */,
    w134, w135,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #13" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op13, xnn_delete_operator);

  xnn_operator_t op14 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    72 /* channels */, 72 /* input stride */, 72 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
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
    72 /* input channels per group */,
    24 /* output_channels_per_group */,
    72 /* input pixel stride */,
    24 /* output pixel stride */,
    w136, w137,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #15" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op15, xnn_delete_operator);

  xnn_operator_t op16 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    24 /* input channels per group */,
    72 /* output_channels_per_group */,
    24 /* input pixel stride */,
    72 /* output pixel stride */,
    w138, w139,
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &op16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #16" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op16, xnn_delete_operator);

  xnn_operator_t op17 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #17" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op17, xnn_delete_operator);

  xnn_operator_t op18 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    72 /* input channels per group */,
    40 /* output_channels_per_group */,
    72 /* input pixel stride */,
    40 /* output pixel stride */,
    w140, w141,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    40 /* input channels per group */,
    120 /* output_channels_per_group */,
    40 /* input pixel stride */,
    120 /* output pixel stride */,
    w142, w143,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #19" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op19, xnn_delete_operator);

  xnn_operator_t op20 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    120 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    120 /* input pixel stride */,
    120 /* output pixel stride */,
    w144, w145,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op20);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #20" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op20, xnn_delete_operator);

  xnn_operator_t op21 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    120 /* channels */, 120 /* input stride */, 120 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op21);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #21" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op21, xnn_delete_operator);

  xnn_operator_t op22 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    120 /* input channels per group */,
    32 /* output_channels_per_group */,
    120 /* input pixel stride */,
    32 /* output pixel stride */,
    w146, w147,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    32 /* input channels per group */,
    120 /* output_channels_per_group */,
    32 /* input pixel stride */,
    120 /* output pixel stride */,
    w148, w149,
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &op23);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #23" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op23, xnn_delete_operator);

  xnn_operator_t op24 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #24" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op24, xnn_delete_operator);

  xnn_operator_t op25 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    120 /* input channels per group */,
    40 /* output_channels_per_group */,
    120 /* input pixel stride */,
    40 /* output pixel stride */,
    w150, w151,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #25" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op25, xnn_delete_operator);

  xnn_operator_t op26 = nullptr;
  status = xnn_create_add_nc_f32(
    40 /* channels */,
    40 /* a stride */,
    40 /* b stride */,
    40 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #26" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op26, xnn_delete_operator);

  xnn_operator_t op27 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    40 /* input channels per group */,
    120 /* output_channels_per_group */,
    40 /* input pixel stride */,
    120 /* output pixel stride */,
    w152, w153,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #27" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op27, xnn_delete_operator);

  xnn_operator_t op28 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    120 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    120 /* input pixel stride */,
    120 /* output pixel stride */,
    w154, w155,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #28" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op28, xnn_delete_operator);

  xnn_operator_t op29 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    120 /* channels */, 120 /* input stride */, 120 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
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
    120 /* input channels per group */,
    32 /* output_channels_per_group */,
    120 /* input pixel stride */,
    32 /* output pixel stride */,
    w156, w157,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op30);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #30" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op30, xnn_delete_operator);

  xnn_operator_t op31 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    32 /* input channels per group */,
    120 /* output_channels_per_group */,
    32 /* input pixel stride */,
    120 /* output pixel stride */,
    w158, w159,
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &op31);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #31" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op31, xnn_delete_operator);

  xnn_operator_t op32 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #32" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op32, xnn_delete_operator);

  xnn_operator_t op33 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    120 /* input channels per group */,
    40 /* output_channels_per_group */,
    120 /* input pixel stride */,
    40 /* output pixel stride */,
    w160, w161,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op33);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #33" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op33, xnn_delete_operator);

  xnn_operator_t op34 = nullptr;
  status = xnn_create_add_nc_f32(
    40 /* channels */,
    40 /* a stride */,
    40 /* b stride */,
    40 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op34);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #34" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op34, xnn_delete_operator);

  xnn_operator_t op35 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    40 /* input channels per group */,
    240 /* output_channels_per_group */,
    40 /* input pixel stride */,
    240 /* output pixel stride */,
    w162, w163,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op35);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #35" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op35, xnn_delete_operator);

  xnn_operator_t op36 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
    0 /* flags */,
    &op36);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #36" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op36, xnn_delete_operator);

  xnn_operator_t op37 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    240 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    240 /* input pixel stride */,
    240 /* output pixel stride */,
    w164, w165,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op37);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #37" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op37, xnn_delete_operator);

  xnn_operator_t op38 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
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
    240 /* input channels per group */,
    80 /* output_channels_per_group */,
    240 /* input pixel stride */,
    80 /* output pixel stride */,
    w166, w167,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op39);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #39" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op39, xnn_delete_operator);

  xnn_operator_t op40 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    80 /* input channels per group */,
    200 /* output_channels_per_group */,
    80 /* input pixel stride */,
    200 /* output pixel stride */,
    w168, w169,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op40);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #40" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op40, xnn_delete_operator);

  xnn_operator_t op41 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    200 /* channels */,
    200 /* input stride */,
    200 /* output stride */,
    0 /* flags */,
    &op41);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #41" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op41, xnn_delete_operator);

  xnn_operator_t op42 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    200 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    200 /* input pixel stride */,
    200 /* output pixel stride */,
    w170, w171,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op42);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #42" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op42, xnn_delete_operator);

  xnn_operator_t op43 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    200 /* channels */,
    200 /* input stride */,
    200 /* output stride */,
    0 /* flags */,
    &op43);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #43" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op43, xnn_delete_operator);

  xnn_operator_t op44 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    200 /* input channels per group */,
    80 /* output_channels_per_group */,
    200 /* input pixel stride */,
    80 /* output pixel stride */,
    w172, w173,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op44);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #44" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op44, xnn_delete_operator);

  xnn_operator_t op45 = nullptr;
  status = xnn_create_add_nc_f32(
    80 /* channels */,
    80 /* a stride */,
    80 /* b stride */,
    80 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op45);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #45" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op45, xnn_delete_operator);

  xnn_operator_t op46 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    80 /* input channels per group */,
    184 /* output_channels_per_group */,
    80 /* input pixel stride */,
    184 /* output pixel stride */,
    w174, w175,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op46);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #46" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op46, xnn_delete_operator);

  xnn_operator_t op47 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    184 /* channels */,
    184 /* input stride */,
    184 /* output stride */,
    0 /* flags */,
    &op47);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #47" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op47, xnn_delete_operator);

  xnn_operator_t op48 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    184 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    184 /* input pixel stride */,
    184 /* output pixel stride */,
    w176, w177,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op48);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #48" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op48, xnn_delete_operator);

  xnn_operator_t op49 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    184 /* channels */,
    184 /* input stride */,
    184 /* output stride */,
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
    184 /* input channels per group */,
    80 /* output_channels_per_group */,
    184 /* input pixel stride */,
    80 /* output pixel stride */,
    w178, w179,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op50);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #50" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op50, xnn_delete_operator);

  xnn_operator_t op51 = nullptr;
  status = xnn_create_add_nc_f32(
    80 /* channels */,
    80 /* a stride */,
    80 /* b stride */,
    80 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    80 /* input channels per group */,
    184 /* output_channels_per_group */,
    80 /* input pixel stride */,
    184 /* output pixel stride */,
    w180, w181,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op52);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #52" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op52, xnn_delete_operator);

  xnn_operator_t op53 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    184 /* channels */,
    184 /* input stride */,
    184 /* output stride */,
    0 /* flags */,
    &op53);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #53" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op53, xnn_delete_operator);

  xnn_operator_t op54 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    184 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    184 /* input pixel stride */,
    184 /* output pixel stride */,
    w182, w183,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op54);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #54" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op54, xnn_delete_operator);

  xnn_operator_t op55 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    184 /* channels */,
    184 /* input stride */,
    184 /* output stride */,
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
    184 /* input channels per group */,
    80 /* output_channels_per_group */,
    184 /* input pixel stride */,
    80 /* output pixel stride */,
    w184, w185,
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
    80 /* channels */,
    80 /* a stride */,
    80 /* b stride */,
    80 /* c stride */,
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
    80 /* input channels per group */,
    480 /* output_channels_per_group */,
    80 /* input pixel stride */,
    480 /* output pixel stride */,
    w186, w187,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op58);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #58" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op58, xnn_delete_operator);

  xnn_operator_t op59 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    480 /* channels */,
    480 /* input stride */,
    480 /* output stride */,
    0 /* flags */,
    &op59);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #59" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op59, xnn_delete_operator);

  xnn_operator_t op60 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    480 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    480 /* input pixel stride */,
    480 /* output pixel stride */,
    w188, w189,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op60);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #60" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op60, xnn_delete_operator);

  xnn_operator_t op61 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    480 /* channels */,
    480 /* input stride */,
    480 /* output stride */,
    0 /* flags */,
    &op61);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #61" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op61, xnn_delete_operator);

  xnn_operator_t op62 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    480 /* channels */, 480 /* input stride */, 480 /* output stride */,
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
    480 /* input channels per group */,
    120 /* output_channels_per_group */,
    480 /* input pixel stride */,
    120 /* output pixel stride */,
    w190, w191,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op63);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #63" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op63, xnn_delete_operator);

  xnn_operator_t op64 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    120 /* input channels per group */,
    480 /* output_channels_per_group */,
    120 /* input pixel stride */,
    480 /* output pixel stride */,
    w192, w193,
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &op64);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #64" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op64, xnn_delete_operator);

  xnn_operator_t op65 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op65);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #65" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op65, xnn_delete_operator);

  xnn_operator_t op66 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    480 /* input channels per group */,
    112 /* output_channels_per_group */,
    480 /* input pixel stride */,
    112 /* output pixel stride */,
    w194, w195,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op66);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #66" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op66, xnn_delete_operator);

  xnn_operator_t op67 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    112 /* input channels per group */,
    672 /* output_channels_per_group */,
    112 /* input pixel stride */,
    672 /* output pixel stride */,
    w196, w197,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op67);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #67" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op67, xnn_delete_operator);

  xnn_operator_t op68 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    672 /* channels */,
    672 /* input stride */,
    672 /* output stride */,
    0 /* flags */,
    &op68);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #68" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op68, xnn_delete_operator);

  xnn_operator_t op69 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    672 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    672 /* input pixel stride */,
    672 /* output pixel stride */,
    w198, w199,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op69);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #69" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op69, xnn_delete_operator);

  xnn_operator_t op70 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    672 /* channels */,
    672 /* input stride */,
    672 /* output stride */,
    0 /* flags */,
    &op70);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #70" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op70, xnn_delete_operator);

  xnn_operator_t op71 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    672 /* channels */, 672 /* input stride */, 672 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op71);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #71" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op71, xnn_delete_operator);

  xnn_operator_t op72 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    672 /* input channels per group */,
    168 /* output_channels_per_group */,
    672 /* input pixel stride */,
    168 /* output pixel stride */,
    w200, w201,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op72);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #72" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op72, xnn_delete_operator);

  xnn_operator_t op73 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    168 /* input channels per group */,
    672 /* output_channels_per_group */,
    168 /* input pixel stride */,
    672 /* output pixel stride */,
    w202, w203,
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &op73);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #73" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op73, xnn_delete_operator);

  xnn_operator_t op74 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op74);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #74" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op74, xnn_delete_operator);

  xnn_operator_t op75 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    672 /* input channels per group */,
    112 /* output_channels_per_group */,
    672 /* input pixel stride */,
    112 /* output pixel stride */,
    w204, w205,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op75);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #75" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op75, xnn_delete_operator);

  xnn_operator_t op76 = nullptr;
  status = xnn_create_add_nc_f32(
    112 /* channels */,
    112 /* a stride */,
    112 /* b stride */,
    112 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op76);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #76" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op76, xnn_delete_operator);

  xnn_operator_t op77 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    112 /* input channels per group */,
    672 /* output_channels_per_group */,
    112 /* input pixel stride */,
    672 /* output pixel stride */,
    w206, w207,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op77);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #77" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op77, xnn_delete_operator);

  xnn_operator_t op78 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    672 /* channels */,
    672 /* input stride */,
    672 /* output stride */,
    0 /* flags */,
    &op78);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #78" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op78, xnn_delete_operator);

  xnn_operator_t op79 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 1 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    672 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    672 /* input pixel stride */,
    672 /* output pixel stride */,
    w208, w209,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op79);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #79" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op79, xnn_delete_operator);

  xnn_operator_t op80 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    672 /* channels */,
    672 /* input stride */,
    672 /* output stride */,
    0 /* flags */,
    &op80);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #80" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op80, xnn_delete_operator);

  xnn_operator_t op81 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    672 /* channels */, 672 /* input stride */, 672 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op81);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #81" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op81, xnn_delete_operator);

  xnn_operator_t op82 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    672 /* input channels per group */,
    168 /* output_channels_per_group */,
    672 /* input pixel stride */,
    168 /* output pixel stride */,
    w210, w211,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op82);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #82" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op82, xnn_delete_operator);

  xnn_operator_t op83 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    168 /* input channels per group */,
    672 /* output_channels_per_group */,
    168 /* input pixel stride */,
    672 /* output pixel stride */,
    w212, w213,
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &op83);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #83" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op83, xnn_delete_operator);

  xnn_operator_t op84 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op84);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #84" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op84, xnn_delete_operator);

  xnn_operator_t op85 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    672 /* input channels per group */,
    160 /* output_channels_per_group */,
    672 /* input pixel stride */,
    160 /* output pixel stride */,
    w214, w215,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op85);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #85" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op85, xnn_delete_operator);

  xnn_operator_t op86 = nullptr;
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
    w216, w217,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op86);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #86" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op86, xnn_delete_operator);

  xnn_operator_t op87 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    960 /* channels */,
    960 /* input stride */,
    960 /* output stride */,
    0 /* flags */,
    &op87);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #87" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op87, xnn_delete_operator);

  xnn_operator_t op88 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    960 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    960 /* input pixel stride */,
    960 /* output pixel stride */,
    w218, w219,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op88);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #88" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op88, xnn_delete_operator);

  xnn_operator_t op89 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    960 /* channels */,
    960 /* input stride */,
    960 /* output stride */,
    0 /* flags */,
    &op89);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #89" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op89, xnn_delete_operator);

  xnn_operator_t op90 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    960 /* channels */, 960 /* input stride */, 960 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op90);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #90" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op90, xnn_delete_operator);

  xnn_operator_t op91 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    960 /* input channels per group */,
    240 /* output_channels_per_group */,
    960 /* input pixel stride */,
    240 /* output pixel stride */,
    w220, w221,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op91);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #91" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op91, xnn_delete_operator);

  xnn_operator_t op92 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    240 /* input channels per group */,
    960 /* output_channels_per_group */,
    240 /* input pixel stride */,
    960 /* output pixel stride */,
    w222, w223,
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &op92);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #92" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op92, xnn_delete_operator);

  xnn_operator_t op93 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op93);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #93" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op93, xnn_delete_operator);

  xnn_operator_t op94 = nullptr;
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
    w224, w225,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op94);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #94" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op94, xnn_delete_operator);

  xnn_operator_t op95 = nullptr;
  status = xnn_create_add_nc_f32(
    160 /* channels */,
    160 /* a stride */,
    160 /* b stride */,
    160 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op95);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #95" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op95, xnn_delete_operator);

  xnn_operator_t op96 = nullptr;
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
    w226, w227,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op96);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #96" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op96, xnn_delete_operator);

  xnn_operator_t op97 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    960 /* channels */,
    960 /* input stride */,
    960 /* output stride */,
    0 /* flags */,
    &op97);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #97" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op97, xnn_delete_operator);

  xnn_operator_t op98 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    960 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    960 /* input pixel stride */,
    960 /* output pixel stride */,
    w228, w229,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op98);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #98" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op98, xnn_delete_operator);

  xnn_operator_t op99 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    960 /* channels */,
    960 /* input stride */,
    960 /* output stride */,
    0 /* flags */,
    &op99);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #99" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op99, xnn_delete_operator);

  xnn_operator_t op100 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    960 /* channels */, 960 /* input stride */, 960 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op100);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #100" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op100, xnn_delete_operator);

  xnn_operator_t op101 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    960 /* input channels per group */,
    240 /* output_channels_per_group */,
    960 /* input pixel stride */,
    240 /* output pixel stride */,
    w230, w231,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op101);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #101" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op101, xnn_delete_operator);

  xnn_operator_t op102 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    240 /* input channels per group */,
    960 /* output_channels_per_group */,
    240 /* input pixel stride */,
    960 /* output pixel stride */,
    w232, w233,
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &op102);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #102" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op102, xnn_delete_operator);

  xnn_operator_t op103 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op103);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #103" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op103, xnn_delete_operator);

  xnn_operator_t op104 = nullptr;
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
    w234, w235,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op104);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #104" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op104, xnn_delete_operator);

  xnn_operator_t op105 = nullptr;
  status = xnn_create_add_nc_f32(
    160 /* channels */,
    160 /* a stride */,
    160 /* b stride */,
    160 /* c stride */,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op105);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #105" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op105, xnn_delete_operator);

  xnn_operator_t op106 = nullptr;
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
    w236, w237,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op106);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #106" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op106, xnn_delete_operator);

  xnn_operator_t op107 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    960 /* channels */,
    960 /* input stride */,
    960 /* output stride */,
    0 /* flags */,
    &op107);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #107" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op107, xnn_delete_operator);

  xnn_operator_t op108 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    960 /* channels */, 960 /* input stride */, 960 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op108);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #108" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op108, xnn_delete_operator);

  xnn_operator_t op109 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    960 /* input channels per group */,
    1280 /* output_channels_per_group */,
    960 /* input pixel stride */,
    1280 /* output pixel stride */,
    w238, w239,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op109);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #109" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op109, xnn_delete_operator);

  xnn_operator_t op110 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    1280 /* channels */,
    1280 /* input stride */,
    1280 /* output stride */,
    0 /* flags */,
    &op110);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #110" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op110, xnn_delete_operator);

  xnn_operator_t op111 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    1280 /* channels */, 1280 /* input stride */, 1280 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op111);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #111" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op111, xnn_delete_operator);

  xnn_operator_t op112 = nullptr;
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
    w240, w241,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op112);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #112" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op112, xnn_delete_operator);



  status = xnn_setup_convolution2d_nhwc_f32(
    op0,
    1 /* batch size */, 224 /* input height */, 224 /* input width */,
    &v0[0] /* input */, &v1[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op1,
    12544 /* batch size */,
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

  status = xnn_setup_add_nc_f32(
    op4,
    12544 /* batch size */,
    &v4[0] /* a */, &v2[0] /* b */, &v5[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op5,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    &v5[0] /* input */, &v6[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #5" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op6,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
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

  status = xnn_setup_convolution2d_nhwc_f32(
    op9,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    &v9[0] /* input */, &v10[0] /* output */,
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

  status = xnn_setup_add_nc_f32(
    op11,
    3136 /* batch size */,
    &v11[0] /* a */, &v8[0] /* b */, &v12[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op12,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    &v12[0] /* input */, &v13[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op13,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    &v13[0] /* input */, &v14[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #13" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op14,
    1 /* batch size */, 784 /* width */,
    &v14[0] /* input */, &v15[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op15,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v15[0] /* input */, &v16[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #15" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op16,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v16[0] /* input */, &v17[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #16" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 28, 28, 72 };
    const size_t b_shape[] = { 1, 1, 1, 72 };
    status = xnn_setup_multiply_nd_f32(
      op17,
      4, a_shape, 4, b_shape,
      &v14[0] /* a */, &v17[0] /* b */, &v18[0] /* output */,
      threadpool /* threadpool */);
  }
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

  status = xnn_setup_convolution2d_nhwc_f32(
    op20,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    &v20[0] /* input */, &v21[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op21,
    1 /* batch size */, 784 /* width */,
    &v21[0] /* input */, &v22[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #21" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op22,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v22[0] /* input */, &v23[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op23,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v23[0] /* input */, &v24[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #23" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 28, 28, 120 };
    const size_t b_shape[] = { 1, 1, 1, 120 };
    status = xnn_setup_multiply_nd_f32(
      op24,
      4, a_shape, 4, b_shape,
      &v21[0] /* a */, &v24[0] /* b */, &v25[0] /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op25,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    &v25[0] /* input */, &v26[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #25" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nc_f32(
    op26,
    784 /* batch size */,
    &v26[0] /* a */, &v19[0] /* b */, &v27[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #26" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op27,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    &v27[0] /* input */, &v28[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op28,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    &v28[0] /* input */, &v29[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #28" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op29,
    1 /* batch size */, 784 /* width */,
    &v29[0] /* input */, &v30[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #29" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op30,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v30[0] /* input */, &v31[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #30" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op31,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v31[0] /* input */, &v32[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #31" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 28, 28, 120 };
    const size_t b_shape[] = { 1, 1, 1, 120 };
    status = xnn_setup_multiply_nd_f32(
      op32,
      4, a_shape, 4, b_shape,
      &v29[0] /* a */, &v32[0] /* b */, &v33[0] /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #32" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op33,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    &v33[0] /* input */, &v34[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #33" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nc_f32(
    op34,
    784 /* batch size */,
    &v34[0] /* a */, &v27[0] /* b */, &v35[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #34" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op35,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    &v35[0] /* input */, &v36[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #35" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op36,
    784 /* batch size */,
    &v36[0] /* input */, &v37[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #36" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op37,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    &v37[0] /* input */, &v38[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #37" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op38,
    196 /* batch size */,
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

  status = xnn_setup_hardswish_nc_f32(
    op41,
    196 /* batch size */,
    &v41[0] /* input */, &v42[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #41" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op42,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v42[0] /* input */, &v43[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #42" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op43,
    196 /* batch size */,
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

  status = xnn_setup_add_nc_f32(
    op45,
    196 /* batch size */,
    &v45[0] /* a */, &v40[0] /* b */, &v46[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #45" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op46,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v46[0] /* input */, &v47[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #46" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op47,
    196 /* batch size */,
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

  status = xnn_setup_hardswish_nc_f32(
    op49,
    196 /* batch size */,
    &v49[0] /* input */, &v50[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #49" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op50,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v50[0] /* input */, &v51[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #50" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nc_f32(
    op51,
    196 /* batch size */,
    &v51[0] /* a */, &v46[0] /* b */, &v52[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #51" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op52,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v52[0] /* input */, &v53[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #52" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op53,
    196 /* batch size */,
    &v53[0] /* input */, &v54[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #53" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op54,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v54[0] /* input */, &v55[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #54" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op55,
    196 /* batch size */,
    &v55[0] /* input */, &v56[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #55" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op56,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v56[0] /* input */, &v57[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #56" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nc_f32(
    op57,
    196 /* batch size */,
    &v57[0] /* a */, &v52[0] /* b */, &v58[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #57" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op58,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v58[0] /* input */, &v59[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #58" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op59,
    196 /* batch size */,
    &v59[0] /* input */, &v60[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #59" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op60,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v60[0] /* input */, &v61[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #60" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op61,
    196 /* batch size */,
    &v61[0] /* input */, &v62[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #61" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op62,
    1 /* batch size */, 196 /* width */,
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

  status = xnn_setup_convolution2d_nhwc_f32(
    op64,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v64[0] /* input */, &v65[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #64" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 480 };
    const size_t b_shape[] = { 1, 1, 1, 480 };
    status = xnn_setup_multiply_nd_f32(
      op65,
      4, a_shape, 4, b_shape,
      &v62[0] /* a */, &v65[0] /* b */, &v66[0] /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #65" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op66,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v66[0] /* input */, &v67[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #66" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op67,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v67[0] /* input */, &v68[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #67" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op68,
    196 /* batch size */,
    &v68[0] /* input */, &v69[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #68" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op69,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v69[0] /* input */, &v70[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #69" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op70,
    196 /* batch size */,
    &v70[0] /* input */, &v71[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #70" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op71,
    1 /* batch size */, 196 /* width */,
    &v71[0] /* input */, &v72[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #71" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op72,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v72[0] /* input */, &v73[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #72" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op73,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v73[0] /* input */, &v74[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #73" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 672 };
    const size_t b_shape[] = { 1, 1, 1, 672 };
    status = xnn_setup_multiply_nd_f32(
      op74,
      4, a_shape, 4, b_shape,
      &v71[0] /* a */, &v74[0] /* b */, &v75[0] /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #74" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op75,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v75[0] /* input */, &v76[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #75" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nc_f32(
    op76,
    196 /* batch size */,
    &v76[0] /* a */, &v67[0] /* b */, &v77[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #76" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op77,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v77[0] /* input */, &v78[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #77" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op78,
    196 /* batch size */,
    &v78[0] /* input */, &v79[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #78" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op79,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    &v79[0] /* input */, &v80[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #79" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op80,
    49 /* batch size */,
    &v80[0] /* input */, &v81[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #80" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op81,
    1 /* batch size */, 49 /* width */,
    &v81[0] /* input */, &v82[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #81" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op82,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v82[0] /* input */, &v83[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #82" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op83,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v83[0] /* input */, &v84[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #83" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 672 };
    const size_t b_shape[] = { 1, 1, 1, 672 };
    status = xnn_setup_multiply_nd_f32(
      op84,
      4, a_shape, 4, b_shape,
      &v81[0] /* a */, &v84[0] /* b */, &v85[0] /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #84" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op85,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v85[0] /* input */, &v86[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #85" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op86,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v86[0] /* input */, &v87[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #86" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op87,
    49 /* batch size */,
    &v87[0] /* input */, &v88[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #87" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op88,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v88[0] /* input */, &v89[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #88" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op89,
    49 /* batch size */,
    &v89[0] /* input */, &v90[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #89" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op90,
    1 /* batch size */, 49 /* width */,
    &v90[0] /* input */, &v91[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #90" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op91,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v91[0] /* input */, &v92[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #91" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op92,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v92[0] /* input */, &v93[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #92" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 960 };
    const size_t b_shape[] = { 1, 1, 1, 960 };
    status = xnn_setup_multiply_nd_f32(
      op93,
      4, a_shape, 4, b_shape,
      &v90[0] /* a */, &v93[0] /* b */, &v94[0] /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #93" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op94,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v94[0] /* input */, &v95[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #94" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nc_f32(
    op95,
    49 /* batch size */,
    &v95[0] /* a */, &v86[0] /* b */, &v96[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #95" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op96,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v96[0] /* input */, &v97[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #96" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op97,
    49 /* batch size */,
    &v97[0] /* input */, &v98[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #97" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op98,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v98[0] /* input */, &v99[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #98" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op99,
    49 /* batch size */,
    &v99[0] /* input */, &v100[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #99" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op100,
    1 /* batch size */, 49 /* width */,
    &v100[0] /* input */, &v101[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #100" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op101,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v101[0] /* input */, &v102[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #101" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op102,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v102[0] /* input */, &v103[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #102" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 960 };
    const size_t b_shape[] = { 1, 1, 1, 960 };
    status = xnn_setup_multiply_nd_f32(
      op103,
      4, a_shape, 4, b_shape,
      &v100[0] /* a */, &v103[0] /* b */, &v104[0] /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #103" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op104,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v104[0] /* input */, &v105[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #104" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nc_f32(
    op105,
    49 /* batch size */,
    &v105[0] /* a */, &v96[0] /* b */, &v106[0] /* sum */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #105" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op106,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    &v106[0] /* input */, &v107[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #106" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op107,
    49 /* batch size */,
    &v107[0] /* input */, &v108[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #107" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op108,
    1 /* batch size */, 49 /* width */,
    &v108[0] /* input */, &v109[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #108" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op109,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v109[0] /* input */, &v110[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #109" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op110,
    1 /* batch size */,
    &v110[0] /* input */, &v111[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #110" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op111,
    1 /* batch size */, 1 /* width */,
    &v111[0] /* input */, &v112[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #111" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op112,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    &v112[0] /* input */, &v113[0] /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #112" << std::endl;
    return ExecutionPlan();
  }

  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wpessimizing-move"
  return operators;
  #pragma clang diagnostic pop
}

}  // namespace models
