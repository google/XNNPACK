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

#include "models/models.h"

namespace models {

ExecutionPlan FP32MobileNetV3Small(pthreadpool_t threadpool) {
  alignas(16) static float v0[150528];
  alignas(16) static float v1[200704];
  alignas(16) static float v2[200704];
  alignas(16) static float v3[50176];
  alignas(16) static float v4[16];
  alignas(16) static float v5[8];
  alignas(16) static float v6[16];
  alignas(16) static float v7[50176];
  alignas(16) static float v8[50176];
  alignas(16) static float v9[225792];
  alignas(16) static float v10[56448];
  alignas(16) static float v11[18816];
  alignas(16) static float v12[68992];
  alignas(16) static float v13[68992];
  alignas(16) static float v14[18816];
  alignas(16) static float v15[18816];
  alignas(16) static float v16[75264];
  alignas(16) static float v17[75264];
  alignas(16) static float v18[18816];
  alignas(16) static float v19[18816];
  alignas(16) static float v20[96];
  alignas(16) static float v21[24];
  alignas(16) static float v22[96];
  alignas(16) static float v23[18816];
  alignas(16) static float v24[7840];
  alignas(16) static float v25[47040];
  alignas(16) static float v26[47040];
  alignas(16) static float v27[47040];
  alignas(16) static float v28[47040];
  alignas(16) static float v29[240];
  alignas(16) static float v30[64];
  alignas(16) static float v31[240];
  alignas(16) static float v32[47040];
  alignas(16) static float v33[7840];
  alignas(16) static float v34[7840];
  alignas(16) static float v35[47040];
  alignas(16) static float v36[47040];
  alignas(16) static float v37[47040];
  alignas(16) static float v38[47040];
  alignas(16) static float v39[240];
  alignas(16) static float v40[64];
  alignas(16) static float v41[240];
  alignas(16) static float v42[47040];
  alignas(16) static float v43[7840];
  alignas(16) static float v44[7840];
  alignas(16) static float v45[23520];
  alignas(16) static float v46[23520];
  alignas(16) static float v47[23520];
  alignas(16) static float v48[23520];
  alignas(16) static float v49[120];
  alignas(16) static float v50[32];
  alignas(16) static float v51[120];
  alignas(16) static float v52[23520];
  alignas(16) static float v53[9408];
  alignas(16) static float v54[28224];
  alignas(16) static float v55[28224];
  alignas(16) static float v56[28224];
  alignas(16) static float v57[28224];
  alignas(16) static float v58[144];
  alignas(16) static float v59[40];
  alignas(16) static float v60[144];
  alignas(16) static float v61[28224];
  alignas(16) static float v62[9408];
  alignas(16) static float v63[9408];
  alignas(16) static float v64[56448];
  alignas(16) static float v65[56448];
  alignas(16) static float v66[14112];
  alignas(16) static float v67[14112];
  alignas(16) static float v68[288];
  alignas(16) static float v69[72];
  alignas(16) static float v70[288];
  alignas(16) static float v71[14112];
  alignas(16) static float v72[4704];
  alignas(16) static float v73[28224];
  alignas(16) static float v74[28224];
  alignas(16) static float v75[28224];
  alignas(16) static float v76[28224];
  alignas(16) static float v77[576];
  alignas(16) static float v78[144];
  alignas(16) static float v79[576];
  alignas(16) static float v80[28224];
  alignas(16) static float v81[4704];
  alignas(16) static float v82[4704];
  alignas(16) static float v83[28224];
  alignas(16) static float v84[28224];
  alignas(16) static float v85[28224];
  alignas(16) static float v86[28224];
  alignas(16) static float v87[576];
  alignas(16) static float v88[144];
  alignas(16) static float v89[576];
  alignas(16) static float v90[28224];
  alignas(16) static float v91[4704];
  alignas(16) static float v92[4704];
  alignas(16) static float v93[28224];
  alignas(16) static float v94[28224];
  alignas(16) static float v95[576];
  alignas(16) static float v96[1024];
  alignas(16) static float v97[1024];
  alignas(16) static float v98[1024];
  alignas(16) static float v99[1001];
  alignas(16) static float w100[432];
  alignas(16) static float w101[16];
  alignas(16) static float w102[144];
  alignas(16) static float w103[16];
  alignas(16) static float w104[128];
  alignas(16) static float w105[8];
  alignas(16) static float w106[128];
  alignas(16) static float w107[16];
  alignas(16) static float w108[256];
  alignas(16) static float w109[16];
  alignas(16) static float w110[1152];
  alignas(16) static float w111[72];
  alignas(16) static float w112[648];
  alignas(16) static float w113[72];
  alignas(16) static float w114[1728];
  alignas(16) static float w115[24];
  alignas(16) static float w116[2112];
  alignas(16) static float w117[88];
  alignas(16) static float w118[792];
  alignas(16) static float w119[88];
  alignas(16) static float w120[2112];
  alignas(16) static float w121[24];
  alignas(16) static float w122[2304];
  alignas(16) static float w123[96];
  alignas(16) static float w124[2400];
  alignas(16) static float w125[96];
  alignas(16) static float w126[2304];
  alignas(16) static float w127[24];
  alignas(16) static float w128[2304];
  alignas(16) static float w129[96];
  alignas(16) static float w130[3840];
  alignas(16) static float w131[40];
  alignas(16) static float w132[9600];
  alignas(16) static float w133[240];
  alignas(16) static float w134[6000];
  alignas(16) static float w135[240];
  alignas(16) static float w136[15360];
  alignas(16) static float w137[64];
  alignas(16) static float w138[15360];
  alignas(16) static float w139[240];
  alignas(16) static float w140[9600];
  alignas(16) static float w141[40];
  alignas(16) static float w142[9600];
  alignas(16) static float w143[240];
  alignas(16) static float w144[6000];
  alignas(16) static float w145[240];
  alignas(16) static float w146[15360];
  alignas(16) static float w147[64];
  alignas(16) static float w148[15360];
  alignas(16) static float w149[240];
  alignas(16) static float w150[9600];
  alignas(16) static float w151[40];
  alignas(16) static float w152[4800];
  alignas(16) static float w153[120];
  alignas(16) static float w154[3000];
  alignas(16) static float w155[120];
  alignas(16) static float w156[3840];
  alignas(16) static float w157[32];
  alignas(16) static float w158[3840];
  alignas(16) static float w159[120];
  alignas(16) static float w160[5760];
  alignas(16) static float w161[48];
  alignas(16) static float w162[6912];
  alignas(16) static float w163[144];
  alignas(16) static float w164[3600];
  alignas(16) static float w165[144];
  alignas(16) static float w166[5760];
  alignas(16) static float w167[40];
  alignas(16) static float w168[5760];
  alignas(16) static float w169[144];
  alignas(16) static float w170[6912];
  alignas(16) static float w171[48];
  alignas(16) static float w172[13824];
  alignas(16) static float w173[288];
  alignas(16) static float w174[7200];
  alignas(16) static float w175[288];
  alignas(16) static float w176[20736];
  alignas(16) static float w177[72];
  alignas(16) static float w178[20736];
  alignas(16) static float w179[288];
  alignas(16) static float w180[27648];
  alignas(16) static float w181[96];
  alignas(16) static float w182[55296];
  alignas(16) static float w183[576];
  alignas(16) static float w184[14400];
  alignas(16) static float w185[576];
  alignas(16) static float w186[82944];
  alignas(16) static float w187[144];
  alignas(16) static float w188[82944];
  alignas(16) static float w189[576];
  alignas(16) static float w190[55296];
  alignas(16) static float w191[96];
  alignas(16) static float w192[55296];
  alignas(16) static float w193[576];
  alignas(16) static float w194[14400];
  alignas(16) static float w195[576];
  alignas(16) static float w196[82944];
  alignas(16) static float w197[144];
  alignas(16) static float w198[82944];
  alignas(16) static float w199[576];
  alignas(16) static float w200[55296];
  alignas(16) static float w201[96];
  alignas(16) static float w202[55296];
  alignas(16) static float w203[576];
  alignas(16) static float w204[589824];
  alignas(16) static float w205[1024];
  alignas(16) static float w206[1025024];
  alignas(16) static float w207[1001];

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), std::ref(rng));
  std::generate(v0, v0 + 150528, std::ref(f32rng));
  std::generate(v1, v1 + 200704, std::ref(f32rng));
  std::generate(v2, v2 + 200704, std::ref(f32rng));
  std::generate(v3, v3 + 50176, std::ref(f32rng));
  std::generate(v4, v4 + 16, std::ref(f32rng));
  std::generate(v5, v5 + 8, std::ref(f32rng));
  std::generate(v6, v6 + 16, std::ref(f32rng));
  std::generate(v7, v7 + 50176, std::ref(f32rng));
  std::generate(v8, v8 + 50176, std::ref(f32rng));
  std::generate(v9, v9 + 225792, std::ref(f32rng));
  std::generate(v10, v10 + 56448, std::ref(f32rng));
  std::generate(v11, v11 + 18816, std::ref(f32rng));
  std::generate(v12, v12 + 68992, std::ref(f32rng));
  std::generate(v13, v13 + 68992, std::ref(f32rng));
  std::generate(v14, v14 + 18816, std::ref(f32rng));
  std::generate(v15, v15 + 18816, std::ref(f32rng));
  std::generate(v16, v16 + 75264, std::ref(f32rng));
  std::generate(v17, v17 + 75264, std::ref(f32rng));
  std::generate(v18, v18 + 18816, std::ref(f32rng));
  std::generate(v19, v19 + 18816, std::ref(f32rng));
  std::generate(v20, v20 + 96, std::ref(f32rng));
  std::generate(v21, v21 + 24, std::ref(f32rng));
  std::generate(v22, v22 + 96, std::ref(f32rng));
  std::generate(v23, v23 + 18816, std::ref(f32rng));
  std::generate(v24, v24 + 7840, std::ref(f32rng));
  std::generate(v25, v25 + 47040, std::ref(f32rng));
  std::generate(v26, v26 + 47040, std::ref(f32rng));
  std::generate(v27, v27 + 47040, std::ref(f32rng));
  std::generate(v28, v28 + 47040, std::ref(f32rng));
  std::generate(v29, v29 + 240, std::ref(f32rng));
  std::generate(v30, v30 + 64, std::ref(f32rng));
  std::generate(v31, v31 + 240, std::ref(f32rng));
  std::generate(v32, v32 + 47040, std::ref(f32rng));
  std::generate(v33, v33 + 7840, std::ref(f32rng));
  std::generate(v34, v34 + 7840, std::ref(f32rng));
  std::generate(v35, v35 + 47040, std::ref(f32rng));
  std::generate(v36, v36 + 47040, std::ref(f32rng));
  std::generate(v37, v37 + 47040, std::ref(f32rng));
  std::generate(v38, v38 + 47040, std::ref(f32rng));
  std::generate(v39, v39 + 240, std::ref(f32rng));
  std::generate(v40, v40 + 64, std::ref(f32rng));
  std::generate(v41, v41 + 240, std::ref(f32rng));
  std::generate(v42, v42 + 47040, std::ref(f32rng));
  std::generate(v43, v43 + 7840, std::ref(f32rng));
  std::generate(v44, v44 + 7840, std::ref(f32rng));
  std::generate(v45, v45 + 23520, std::ref(f32rng));
  std::generate(v46, v46 + 23520, std::ref(f32rng));
  std::generate(v47, v47 + 23520, std::ref(f32rng));
  std::generate(v48, v48 + 23520, std::ref(f32rng));
  std::generate(v49, v49 + 120, std::ref(f32rng));
  std::generate(v50, v50 + 32, std::ref(f32rng));
  std::generate(v51, v51 + 120, std::ref(f32rng));
  std::generate(v52, v52 + 23520, std::ref(f32rng));
  std::generate(v53, v53 + 9408, std::ref(f32rng));
  std::generate(v54, v54 + 28224, std::ref(f32rng));
  std::generate(v55, v55 + 28224, std::ref(f32rng));
  std::generate(v56, v56 + 28224, std::ref(f32rng));
  std::generate(v57, v57 + 28224, std::ref(f32rng));
  std::generate(v58, v58 + 144, std::ref(f32rng));
  std::generate(v59, v59 + 40, std::ref(f32rng));
  std::generate(v60, v60 + 144, std::ref(f32rng));
  std::generate(v61, v61 + 28224, std::ref(f32rng));
  std::generate(v62, v62 + 9408, std::ref(f32rng));
  std::generate(v63, v63 + 9408, std::ref(f32rng));
  std::generate(v64, v64 + 56448, std::ref(f32rng));
  std::generate(v65, v65 + 56448, std::ref(f32rng));
  std::generate(v66, v66 + 14112, std::ref(f32rng));
  std::generate(v67, v67 + 14112, std::ref(f32rng));
  std::generate(v68, v68 + 288, std::ref(f32rng));
  std::generate(v69, v69 + 72, std::ref(f32rng));
  std::generate(v70, v70 + 288, std::ref(f32rng));
  std::generate(v71, v71 + 14112, std::ref(f32rng));
  std::generate(v72, v72 + 4704, std::ref(f32rng));
  std::generate(v73, v73 + 28224, std::ref(f32rng));
  std::generate(v74, v74 + 28224, std::ref(f32rng));
  std::generate(v75, v75 + 28224, std::ref(f32rng));
  std::generate(v76, v76 + 28224, std::ref(f32rng));
  std::generate(v77, v77 + 576, std::ref(f32rng));
  std::generate(v78, v78 + 144, std::ref(f32rng));
  std::generate(v79, v79 + 576, std::ref(f32rng));
  std::generate(v80, v80 + 28224, std::ref(f32rng));
  std::generate(v81, v81 + 4704, std::ref(f32rng));
  std::generate(v82, v82 + 4704, std::ref(f32rng));
  std::generate(v83, v83 + 28224, std::ref(f32rng));
  std::generate(v84, v84 + 28224, std::ref(f32rng));
  std::generate(v85, v85 + 28224, std::ref(f32rng));
  std::generate(v86, v86 + 28224, std::ref(f32rng));
  std::generate(v87, v87 + 576, std::ref(f32rng));
  std::generate(v88, v88 + 144, std::ref(f32rng));
  std::generate(v89, v89 + 576, std::ref(f32rng));
  std::generate(v90, v90 + 28224, std::ref(f32rng));
  std::generate(v91, v91 + 4704, std::ref(f32rng));
  std::generate(v92, v92 + 4704, std::ref(f32rng));
  std::generate(v93, v93 + 28224, std::ref(f32rng));
  std::generate(v94, v94 + 28224, std::ref(f32rng));
  std::generate(v95, v95 + 576, std::ref(f32rng));
  std::generate(v96, v96 + 1024, std::ref(f32rng));
  std::generate(v97, v97 + 1024, std::ref(f32rng));
  std::generate(v98, v98 + 1024, std::ref(f32rng));
  std::generate(v99, v99 + 1001, std::ref(f32rng));
  std::generate(w100, w100 + 432, std::ref(f32rng));
  std::generate(w101, w101 + 16, std::ref(f32rng));
  std::generate(w102, w102 + 144, std::ref(f32rng));
  std::generate(w103, w103 + 16, std::ref(f32rng));
  std::generate(w104, w104 + 128, std::ref(f32rng));
  std::generate(w105, w105 + 8, std::ref(f32rng));
  std::generate(w106, w106 + 128, std::ref(f32rng));
  std::generate(w107, w107 + 16, std::ref(f32rng));
  std::generate(w108, w108 + 256, std::ref(f32rng));
  std::generate(w109, w109 + 16, std::ref(f32rng));
  std::generate(w110, w110 + 1152, std::ref(f32rng));
  std::generate(w111, w111 + 72, std::ref(f32rng));
  std::generate(w112, w112 + 648, std::ref(f32rng));
  std::generate(w113, w113 + 72, std::ref(f32rng));
  std::generate(w114, w114 + 1728, std::ref(f32rng));
  std::generate(w115, w115 + 24, std::ref(f32rng));
  std::generate(w116, w116 + 2112, std::ref(f32rng));
  std::generate(w117, w117 + 88, std::ref(f32rng));
  std::generate(w118, w118 + 792, std::ref(f32rng));
  std::generate(w119, w119 + 88, std::ref(f32rng));
  std::generate(w120, w120 + 2112, std::ref(f32rng));
  std::generate(w121, w121 + 24, std::ref(f32rng));
  std::generate(w122, w122 + 2304, std::ref(f32rng));
  std::generate(w123, w123 + 96, std::ref(f32rng));
  std::generate(w124, w124 + 2400, std::ref(f32rng));
  std::generate(w125, w125 + 96, std::ref(f32rng));
  std::generate(w126, w126 + 2304, std::ref(f32rng));
  std::generate(w127, w127 + 24, std::ref(f32rng));
  std::generate(w128, w128 + 2304, std::ref(f32rng));
  std::generate(w129, w129 + 96, std::ref(f32rng));
  std::generate(w130, w130 + 3840, std::ref(f32rng));
  std::generate(w131, w131 + 40, std::ref(f32rng));
  std::generate(w132, w132 + 9600, std::ref(f32rng));
  std::generate(w133, w133 + 240, std::ref(f32rng));
  std::generate(w134, w134 + 6000, std::ref(f32rng));
  std::generate(w135, w135 + 240, std::ref(f32rng));
  std::generate(w136, w136 + 15360, std::ref(f32rng));
  std::generate(w137, w137 + 64, std::ref(f32rng));
  std::generate(w138, w138 + 15360, std::ref(f32rng));
  std::generate(w139, w139 + 240, std::ref(f32rng));
  std::generate(w140, w140 + 9600, std::ref(f32rng));
  std::generate(w141, w141 + 40, std::ref(f32rng));
  std::generate(w142, w142 + 9600, std::ref(f32rng));
  std::generate(w143, w143 + 240, std::ref(f32rng));
  std::generate(w144, w144 + 6000, std::ref(f32rng));
  std::generate(w145, w145 + 240, std::ref(f32rng));
  std::generate(w146, w146 + 15360, std::ref(f32rng));
  std::generate(w147, w147 + 64, std::ref(f32rng));
  std::generate(w148, w148 + 15360, std::ref(f32rng));
  std::generate(w149, w149 + 240, std::ref(f32rng));
  std::generate(w150, w150 + 9600, std::ref(f32rng));
  std::generate(w151, w151 + 40, std::ref(f32rng));
  std::generate(w152, w152 + 4800, std::ref(f32rng));
  std::generate(w153, w153 + 120, std::ref(f32rng));
  std::generate(w154, w154 + 3000, std::ref(f32rng));
  std::generate(w155, w155 + 120, std::ref(f32rng));
  std::generate(w156, w156 + 3840, std::ref(f32rng));
  std::generate(w157, w157 + 32, std::ref(f32rng));
  std::generate(w158, w158 + 3840, std::ref(f32rng));
  std::generate(w159, w159 + 120, std::ref(f32rng));
  std::generate(w160, w160 + 5760, std::ref(f32rng));
  std::generate(w161, w161 + 48, std::ref(f32rng));
  std::generate(w162, w162 + 6912, std::ref(f32rng));
  std::generate(w163, w163 + 144, std::ref(f32rng));
  std::generate(w164, w164 + 3600, std::ref(f32rng));
  std::generate(w165, w165 + 144, std::ref(f32rng));
  std::generate(w166, w166 + 5760, std::ref(f32rng));
  std::generate(w167, w167 + 40, std::ref(f32rng));
  std::generate(w168, w168 + 5760, std::ref(f32rng));
  std::generate(w169, w169 + 144, std::ref(f32rng));
  std::generate(w170, w170 + 6912, std::ref(f32rng));
  std::generate(w171, w171 + 48, std::ref(f32rng));
  std::generate(w172, w172 + 13824, std::ref(f32rng));
  std::generate(w173, w173 + 288, std::ref(f32rng));
  std::generate(w174, w174 + 7200, std::ref(f32rng));
  std::generate(w175, w175 + 288, std::ref(f32rng));
  std::generate(w176, w176 + 20736, std::ref(f32rng));
  std::generate(w177, w177 + 72, std::ref(f32rng));
  std::generate(w178, w178 + 20736, std::ref(f32rng));
  std::generate(w179, w179 + 288, std::ref(f32rng));
  std::generate(w180, w180 + 27648, std::ref(f32rng));
  std::generate(w181, w181 + 96, std::ref(f32rng));
  std::generate(w182, w182 + 55296, std::ref(f32rng));
  std::generate(w183, w183 + 576, std::ref(f32rng));
  std::generate(w184, w184 + 14400, std::ref(f32rng));
  std::generate(w185, w185 + 576, std::ref(f32rng));
  std::generate(w186, w186 + 82944, std::ref(f32rng));
  std::generate(w187, w187 + 144, std::ref(f32rng));
  std::generate(w188, w188 + 82944, std::ref(f32rng));
  std::generate(w189, w189 + 576, std::ref(f32rng));
  std::generate(w190, w190 + 55296, std::ref(f32rng));
  std::generate(w191, w191 + 96, std::ref(f32rng));
  std::generate(w192, w192 + 55296, std::ref(f32rng));
  std::generate(w193, w193 + 576, std::ref(f32rng));
  std::generate(w194, w194 + 14400, std::ref(f32rng));
  std::generate(w195, w195 + 576, std::ref(f32rng));
  std::generate(w196, w196 + 82944, std::ref(f32rng));
  std::generate(w197, w197 + 144, std::ref(f32rng));
  std::generate(w198, w198 + 82944, std::ref(f32rng));
  std::generate(w199, w199 + 576, std::ref(f32rng));
  std::generate(w200, w200 + 55296, std::ref(f32rng));
  std::generate(w201, w201 + 96, std::ref(f32rng));
  std::generate(w202, w202 + 55296, std::ref(f32rng));
  std::generate(w203, w203 + 576, std::ref(f32rng));
  std::generate(w204, w204 + 589824, std::ref(f32rng));
  std::generate(w205, w205 + 1024, std::ref(f32rng));
  std::generate(w206, w206 + 1025024, std::ref(f32rng));
  std::generate(w207, w207 + 1001, std::ref(f32rng));

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
    w100, w101,
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
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    16 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    16 /* input pixel stride */,
    16 /* output pixel stride */,
    w102, w103,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #2" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op2, xnn_delete_operator);

  xnn_operator_t op3 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    16 /* channels */, 16 /* input stride */, 16 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #3" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op3, xnn_delete_operator);

  xnn_operator_t op4 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    16 /* input channels per group */,
    8 /* output_channels_per_group */,
    16 /* input pixel stride */,
    8 /* output pixel stride */,
    w104, w105,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    8 /* input channels per group */,
    16 /* output_channels_per_group */,
    8 /* input pixel stride */,
    16 /* output pixel stride */,
    w106, w107,
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &op5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #5" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op5, xnn_delete_operator);

  xnn_operator_t op6 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    16 /* input channels per group */,
    16 /* output_channels_per_group */,
    16 /* input pixel stride */,
    16 /* output pixel stride */,
    w108, w109,
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
    16 /* input channels per group */,
    72 /* output_channels_per_group */,
    16 /* input pixel stride */,
    72 /* output pixel stride */,
    w110, w111,
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
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    72 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    72 /* input pixel stride */,
    72 /* output pixel stride */,
    w112, w113,
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
    w114, w115,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #10" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op10, xnn_delete_operator);

  xnn_operator_t op11 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    24 /* input channels per group */,
    88 /* output_channels_per_group */,
    24 /* input pixel stride */,
    88 /* output pixel stride */,
    w116, w117,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #11" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op11, xnn_delete_operator);

  xnn_operator_t op12 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    88 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    88 /* input pixel stride */,
    88 /* output pixel stride */,
    w118, w119,
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
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    88 /* input channels per group */,
    24 /* output_channels_per_group */,
    88 /* input pixel stride */,
    24 /* output pixel stride */,
    w120, w121,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #13" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op13, xnn_delete_operator);

  xnn_operator_t op14 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    24 /* input channels per group */,
    96 /* output_channels_per_group */,
    24 /* input pixel stride */,
    96 /* output pixel stride */,
    w122, w123,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #15" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op15, xnn_delete_operator);

  xnn_operator_t op16 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    96 /* channels */,
    96 /* input stride */,
    96 /* output stride */,
    0 /* flags */,
    &op16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #16" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op16, xnn_delete_operator);

  xnn_operator_t op17 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 1 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    96 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    96 /* input pixel stride */,
    96 /* output pixel stride */,
    w124, w125,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #17" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op17, xnn_delete_operator);

  xnn_operator_t op18 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    96 /* channels */,
    96 /* input stride */,
    96 /* output stride */,
    0 /* flags */,
    &op18);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #18" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op18, xnn_delete_operator);

  xnn_operator_t op19 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    96 /* channels */, 96 /* input stride */, 96 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #19" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op19, xnn_delete_operator);

  xnn_operator_t op20 = nullptr;
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
    w126, w127,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    24 /* input channels per group */,
    96 /* output_channels_per_group */,
    24 /* input pixel stride */,
    96 /* output pixel stride */,
    w128, w129,
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &op21);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #21" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op21, xnn_delete_operator);

  xnn_operator_t op22 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    96 /* input channels per group */,
    40 /* output_channels_per_group */,
    96 /* input pixel stride */,
    40 /* output pixel stride */,
    w130, w131,
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
    40 /* input channels per group */,
    240 /* output_channels_per_group */,
    40 /* input pixel stride */,
    240 /* output pixel stride */,
    w132, w133,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #24" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op24, xnn_delete_operator);

  xnn_operator_t op25 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
    0 /* flags */,
    &op25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #25" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op25, xnn_delete_operator);

  xnn_operator_t op26 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    240 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    240 /* input pixel stride */,
    240 /* output pixel stride */,
    w134, w135,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #26" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op26, xnn_delete_operator);

  xnn_operator_t op27 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
    0 /* flags */,
    &op27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #27" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op27, xnn_delete_operator);

  xnn_operator_t op28 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    240 /* channels */, 240 /* input stride */, 240 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #28" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op28, xnn_delete_operator);

  xnn_operator_t op29 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    240 /* input channels per group */,
    64 /* output_channels_per_group */,
    240 /* input pixel stride */,
    64 /* output pixel stride */,
    w136, w137,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    64 /* input channels per group */,
    240 /* output_channels_per_group */,
    64 /* input pixel stride */,
    240 /* output pixel stride */,
    w138, w139,
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &op30);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #30" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op30, xnn_delete_operator);

  xnn_operator_t op31 = nullptr;
  status = xnn_create_multiply_nd_f32(
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
    240 /* input channels per group */,
    40 /* output_channels_per_group */,
    240 /* input pixel stride */,
    40 /* output pixel stride */,
    w140, w141,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #32" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op32, xnn_delete_operator);

  xnn_operator_t op33 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    40 /* input channels per group */,
    240 /* output_channels_per_group */,
    40 /* input pixel stride */,
    240 /* output pixel stride */,
    w142, w143,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op34);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #34" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op34, xnn_delete_operator);

  xnn_operator_t op35 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
    0 /* flags */,
    &op35);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #35" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op35, xnn_delete_operator);

  xnn_operator_t op36 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    240 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    240 /* input pixel stride */,
    240 /* output pixel stride */,
    w144, w145,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op36);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #36" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op36, xnn_delete_operator);

  xnn_operator_t op37 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
    0 /* flags */,
    &op37);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #37" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op37, xnn_delete_operator);

  xnn_operator_t op38 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    240 /* channels */, 240 /* input stride */, 240 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
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
    64 /* output_channels_per_group */,
    240 /* input pixel stride */,
    64 /* output pixel stride */,
    w146, w147,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    64 /* input channels per group */,
    240 /* output_channels_per_group */,
    64 /* input pixel stride */,
    240 /* output pixel stride */,
    w148, w149,
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &op40);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #40" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op40, xnn_delete_operator);

  xnn_operator_t op41 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op41);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #41" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op41, xnn_delete_operator);

  xnn_operator_t op42 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    240 /* input channels per group */,
    40 /* output_channels_per_group */,
    240 /* input pixel stride */,
    40 /* output pixel stride */,
    w150, w151,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op42);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #42" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op42, xnn_delete_operator);

  xnn_operator_t op43 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    40 /* input channels per group */,
    120 /* output_channels_per_group */,
    40 /* input pixel stride */,
    120 /* output pixel stride */,
    w152, w153,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op44);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #44" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op44, xnn_delete_operator);

  xnn_operator_t op45 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    120 /* channels */,
    120 /* input stride */,
    120 /* output stride */,
    0 /* flags */,
    &op45);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #45" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op45, xnn_delete_operator);

  xnn_operator_t op46 = nullptr;
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
    120 /* channels */,
    120 /* input stride */,
    120 /* output stride */,
    0 /* flags */,
    &op47);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #47" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op47, xnn_delete_operator);

  xnn_operator_t op48 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    120 /* channels */, 120 /* input stride */, 120 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
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
    120 /* input channels per group */,
    32 /* output_channels_per_group */,
    120 /* input pixel stride */,
    32 /* output pixel stride */,
    w156, w157,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    32 /* input channels per group */,
    120 /* output_channels_per_group */,
    32 /* input pixel stride */,
    120 /* output pixel stride */,
    w158, w159,
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &op50);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #50" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op50, xnn_delete_operator);

  xnn_operator_t op51 = nullptr;
  status = xnn_create_multiply_nd_f32(
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
    120 /* input channels per group */,
    48 /* output_channels_per_group */,
    120 /* input pixel stride */,
    48 /* output pixel stride */,
    w160, w161,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op52);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #52" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op52, xnn_delete_operator);

  xnn_operator_t op53 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    48 /* input channels per group */,
    144 /* output_channels_per_group */,
    48 /* input pixel stride */,
    144 /* output pixel stride */,
    w162, w163,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op53);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #53" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op53, xnn_delete_operator);

  xnn_operator_t op54 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    144 /* channels */,
    144 /* input stride */,
    144 /* output stride */,
    0 /* flags */,
    &op54);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #54" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op54, xnn_delete_operator);

  xnn_operator_t op55 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    144 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    144 /* input pixel stride */,
    144 /* output pixel stride */,
    w164, w165,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op55);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #55" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op55, xnn_delete_operator);

  xnn_operator_t op56 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    144 /* channels */,
    144 /* input stride */,
    144 /* output stride */,
    0 /* flags */,
    &op56);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #56" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op56, xnn_delete_operator);

  xnn_operator_t op57 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    144 /* channels */, 144 /* input stride */, 144 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
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
    144 /* input channels per group */,
    40 /* output_channels_per_group */,
    144 /* input pixel stride */,
    40 /* output pixel stride */,
    w166, w167,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op58);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #58" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op58, xnn_delete_operator);

  xnn_operator_t op59 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    40 /* input channels per group */,
    144 /* output_channels_per_group */,
    40 /* input pixel stride */,
    144 /* output pixel stride */,
    w168, w169,
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &op59);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #59" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op59, xnn_delete_operator);

  xnn_operator_t op60 = nullptr;
  status = xnn_create_multiply_nd_f32(
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
    144 /* input channels per group */,
    48 /* output_channels_per_group */,
    144 /* input pixel stride */,
    48 /* output pixel stride */,
    w170, w171,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op61);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #61" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op61, xnn_delete_operator);

  xnn_operator_t op62 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    48 /* input channels per group */,
    288 /* output_channels_per_group */,
    48 /* input pixel stride */,
    288 /* output pixel stride */,
    w172, w173,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op63);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #63" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op63, xnn_delete_operator);

  xnn_operator_t op64 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    288 /* channels */,
    288 /* input stride */,
    288 /* output stride */,
    0 /* flags */,
    &op64);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #64" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op64, xnn_delete_operator);

  xnn_operator_t op65 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 1 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    288 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    288 /* input pixel stride */,
    288 /* output pixel stride */,
    w174, w175,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op65);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #65" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op65, xnn_delete_operator);

  xnn_operator_t op66 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    288 /* channels */,
    288 /* input stride */,
    288 /* output stride */,
    0 /* flags */,
    &op66);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #66" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op66, xnn_delete_operator);

  xnn_operator_t op67 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    288 /* channels */, 288 /* input stride */, 288 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op67);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #67" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op67, xnn_delete_operator);

  xnn_operator_t op68 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    288 /* input channels per group */,
    72 /* output_channels_per_group */,
    288 /* input pixel stride */,
    72 /* output pixel stride */,
    w176, w177,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op68);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #68" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op68, xnn_delete_operator);

  xnn_operator_t op69 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    72 /* input channels per group */,
    288 /* output_channels_per_group */,
    72 /* input pixel stride */,
    288 /* output pixel stride */,
    w178, w179,
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &op69);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #69" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op69, xnn_delete_operator);

  xnn_operator_t op70 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op70);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #70" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op70, xnn_delete_operator);

  xnn_operator_t op71 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    288 /* input channels per group */,
    96 /* output_channels_per_group */,
    288 /* input pixel stride */,
    96 /* output pixel stride */,
    w180, w181,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    96 /* input channels per group */,
    576 /* output_channels_per_group */,
    96 /* input pixel stride */,
    576 /* output pixel stride */,
    w182, w183,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op72);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #72" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op72, xnn_delete_operator);

  xnn_operator_t op73 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    576 /* channels */,
    576 /* input stride */,
    576 /* output stride */,
    0 /* flags */,
    &op73);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #73" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op73, xnn_delete_operator);

  xnn_operator_t op74 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    576 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    576 /* input pixel stride */,
    576 /* output pixel stride */,
    w184, w185,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op74);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #74" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op74, xnn_delete_operator);

  xnn_operator_t op75 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    576 /* channels */,
    576 /* input stride */,
    576 /* output stride */,
    0 /* flags */,
    &op75);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #75" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op75, xnn_delete_operator);

  xnn_operator_t op76 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    576 /* channels */, 576 /* input stride */, 576 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
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
    576 /* input channels per group */,
    144 /* output_channels_per_group */,
    576 /* input pixel stride */,
    144 /* output pixel stride */,
    w186, w187,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op77);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #77" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op77, xnn_delete_operator);

  xnn_operator_t op78 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    144 /* input channels per group */,
    576 /* output_channels_per_group */,
    144 /* input pixel stride */,
    576 /* output pixel stride */,
    w188, w189,
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &op78);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #78" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op78, xnn_delete_operator);

  xnn_operator_t op79 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op79);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #79" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op79, xnn_delete_operator);

  xnn_operator_t op80 = nullptr;
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
    w190, w191,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op80);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #80" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op80, xnn_delete_operator);

  xnn_operator_t op81 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    96 /* input channels per group */,
    576 /* output_channels_per_group */,
    96 /* input pixel stride */,
    576 /* output pixel stride */,
    w192, w193,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op82);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #82" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op82, xnn_delete_operator);

  xnn_operator_t op83 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    576 /* channels */,
    576 /* input stride */,
    576 /* output stride */,
    0 /* flags */,
    &op83);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #83" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op83, xnn_delete_operator);

  xnn_operator_t op84 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    576 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    576 /* input pixel stride */,
    576 /* output pixel stride */,
    w194, w195,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op84);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #84" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op84, xnn_delete_operator);

  xnn_operator_t op85 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    576 /* channels */,
    576 /* input stride */,
    576 /* output stride */,
    0 /* flags */,
    &op85);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #85" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op85, xnn_delete_operator);

  xnn_operator_t op86 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    576 /* channels */, 576 /* input stride */, 576 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op86);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #86" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op86, xnn_delete_operator);

  xnn_operator_t op87 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    576 /* input channels per group */,
    144 /* output_channels_per_group */,
    576 /* input pixel stride */,
    144 /* output pixel stride */,
    w196, w197,
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op87);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #87" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op87, xnn_delete_operator);

  xnn_operator_t op88 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    144 /* input channels per group */,
    576 /* output_channels_per_group */,
    144 /* input pixel stride */,
    576 /* output pixel stride */,
    w198, w199,
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &op88);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #88" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op88, xnn_delete_operator);

  xnn_operator_t op89 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op89);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #89" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op89, xnn_delete_operator);

  xnn_operator_t op90 = nullptr;
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
    w200, w201,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op90);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #90" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op90, xnn_delete_operator);

  xnn_operator_t op91 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    96 /* input channels per group */,
    576 /* output_channels_per_group */,
    96 /* input pixel stride */,
    576 /* output pixel stride */,
    w202, w203,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op92);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #92" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op92, xnn_delete_operator);

  xnn_operator_t op93 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    576 /* channels */,
    576 /* input stride */,
    576 /* output stride */,
    0 /* flags */,
    &op93);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #93" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op93, xnn_delete_operator);

  xnn_operator_t op94 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    576 /* channels */, 576 /* input stride */, 576 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op94);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #94" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op94, xnn_delete_operator);

  xnn_operator_t op95 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    576 /* input channels per group */,
    1024 /* output_channels_per_group */,
    576 /* input pixel stride */,
    1024 /* output pixel stride */,
    w204, w205,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op95);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #95" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op95, xnn_delete_operator);

  xnn_operator_t op96 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    1024 /* channels */,
    1024 /* input stride */,
    1024 /* output stride */,
    0 /* flags */,
    &op96);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #96" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op96, xnn_delete_operator);

  xnn_operator_t op97 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    1024 /* channels */, 1024 /* input stride */, 1024 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op97);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #97" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op97, xnn_delete_operator);

  xnn_operator_t op98 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    1024 /* input channels per group */,
    1001 /* output_channels_per_group */,
    1024 /* input pixel stride */,
    1001 /* output pixel stride */,
    w206, w207,
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op98);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #98" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op98, xnn_delete_operator);



  status = xnn_setup_convolution2d_nhwc_f32(
    op0,
    1 /* batch size */, 224 /* input height */, 224 /* input width */,
    v0 /* input */, v1 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op1,
    12544 /* batch size */,
    v1 /* input */, v2 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #1" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op2,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v2 /* input */, v3 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op3,
    1 /* batch size */, 3136 /* width */,
    v3 /* input */, v4 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #3" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op4,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v4 /* input */, v5 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op5,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v5 /* input */, v6 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #5" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 56, 56, 16 };
    const size_t b_shape[] = { 1, 1, 1, 16 };
    status = xnn_setup_multiply_nd_f32(
      op6,
      4, a_shape, 4, b_shape,
      v3 /* a */, v6 /* b */, v7 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op7,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v7 /* input */, v8 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op8,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v8 /* input */, v9 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #8" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op9,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v9 /* input */, v10 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op10,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v10 /* input */, v11 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #10" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op11,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v11 /* input */, v12 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op12,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v12 /* input */, v13 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op13,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v13 /* input */, v14 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #13" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 28, 28, 24 };
    const size_t b_shape[] = { 1, 28, 28, 24 };
    status = xnn_setup_add_nd_f32(
      op14,
      4, a_shape, 4, b_shape,
      v14 /* a */, v11 /* b */, v15 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op15,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v15 /* input */, v16 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #15" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op16,
    784 /* batch size */,
    v16 /* input */, v17 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #16" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op17,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v17 /* input */, v18 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op18,
    196 /* batch size */,
    v18 /* input */, v19 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op19,
    1 /* batch size */, 196 /* width */,
    v19 /* input */, v20 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #19" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op20,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v20 /* input */, v21 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op21,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v21 /* input */, v22 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #21" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 96 };
    const size_t b_shape[] = { 1, 1, 1, 96 };
    status = xnn_setup_multiply_nd_f32(
      op22,
      4, a_shape, 4, b_shape,
      v19 /* a */, v22 /* b */, v23 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op23,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v23 /* input */, v24 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #23" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op24,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v24 /* input */, v25 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op25,
    196 /* batch size */,
    v25 /* input */, v26 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #25" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op26,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v26 /* input */, v27 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #26" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op27,
    196 /* batch size */,
    v27 /* input */, v28 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op28,
    1 /* batch size */, 196 /* width */,
    v28 /* input */, v29 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #28" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op29,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v29 /* input */, v30 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #29" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op30,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v30 /* input */, v31 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #30" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 240 };
    const size_t b_shape[] = { 1, 1, 1, 240 };
    status = xnn_setup_multiply_nd_f32(
      op31,
      4, a_shape, 4, b_shape,
      v28 /* a */, v31 /* b */, v32 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #31" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op32,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v32 /* input */, v33 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #32" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 40 };
    const size_t b_shape[] = { 1, 14, 14, 40 };
    status = xnn_setup_add_nd_f32(
      op33,
      4, a_shape, 4, b_shape,
      v33 /* a */, v24 /* b */, v34 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #33" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op34,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v34 /* input */, v35 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #34" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op35,
    196 /* batch size */,
    v35 /* input */, v36 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #35" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op36,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v36 /* input */, v37 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #36" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op37,
    196 /* batch size */,
    v37 /* input */, v38 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #37" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op38,
    1 /* batch size */, 196 /* width */,
    v38 /* input */, v39 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #38" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op39,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v39 /* input */, v40 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #39" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op40,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v40 /* input */, v41 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #40" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 240 };
    const size_t b_shape[] = { 1, 1, 1, 240 };
    status = xnn_setup_multiply_nd_f32(
      op41,
      4, a_shape, 4, b_shape,
      v38 /* a */, v41 /* b */, v42 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #41" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op42,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v42 /* input */, v43 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #42" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 40 };
    const size_t b_shape[] = { 1, 14, 14, 40 };
    status = xnn_setup_add_nd_f32(
      op43,
      4, a_shape, 4, b_shape,
      v43 /* a */, v34 /* b */, v44 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #43" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op44,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v44 /* input */, v45 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #44" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op45,
    196 /* batch size */,
    v45 /* input */, v46 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #45" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op46,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v46 /* input */, v47 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #46" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op47,
    196 /* batch size */,
    v47 /* input */, v48 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #47" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op48,
    1 /* batch size */, 196 /* width */,
    v48 /* input */, v49 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #48" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op49,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v49 /* input */, v50 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #49" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op50,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v50 /* input */, v51 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #50" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 120 };
    const size_t b_shape[] = { 1, 1, 1, 120 };
    status = xnn_setup_multiply_nd_f32(
      op51,
      4, a_shape, 4, b_shape,
      v48 /* a */, v51 /* b */, v52 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #51" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op52,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v52 /* input */, v53 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #52" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op53,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v53 /* input */, v54 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #53" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op54,
    196 /* batch size */,
    v54 /* input */, v55 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #54" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op55,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v55 /* input */, v56 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #55" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op56,
    196 /* batch size */,
    v56 /* input */, v57 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #56" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op57,
    1 /* batch size */, 196 /* width */,
    v57 /* input */, v58 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #57" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op58,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v58 /* input */, v59 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #58" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op59,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v59 /* input */, v60 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #59" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 144 };
    const size_t b_shape[] = { 1, 1, 1, 144 };
    status = xnn_setup_multiply_nd_f32(
      op60,
      4, a_shape, 4, b_shape,
      v57 /* a */, v60 /* b */, v61 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #60" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op61,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v61 /* input */, v62 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #61" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 48 };
    const size_t b_shape[] = { 1, 14, 14, 48 };
    status = xnn_setup_add_nd_f32(
      op62,
      4, a_shape, 4, b_shape,
      v62 /* a */, v53 /* b */, v63 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #62" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op63,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v63 /* input */, v64 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #63" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op64,
    196 /* batch size */,
    v64 /* input */, v65 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #64" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op65,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v65 /* input */, v66 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #65" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op66,
    49 /* batch size */,
    v66 /* input */, v67 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #66" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op67,
    1 /* batch size */, 49 /* width */,
    v67 /* input */, v68 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #67" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op68,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v68 /* input */, v69 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #68" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op69,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v69 /* input */, v70 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #69" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 288 };
    const size_t b_shape[] = { 1, 1, 1, 288 };
    status = xnn_setup_multiply_nd_f32(
      op70,
      4, a_shape, 4, b_shape,
      v67 /* a */, v70 /* b */, v71 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #70" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op71,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v71 /* input */, v72 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #71" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op72,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v72 /* input */, v73 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #72" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op73,
    49 /* batch size */,
    v73 /* input */, v74 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #73" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op74,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v74 /* input */, v75 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #74" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op75,
    49 /* batch size */,
    v75 /* input */, v76 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #75" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op76,
    1 /* batch size */, 49 /* width */,
    v76 /* input */, v77 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #76" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op77,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v77 /* input */, v78 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #77" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op78,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v78 /* input */, v79 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #78" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 576 };
    const size_t b_shape[] = { 1, 1, 1, 576 };
    status = xnn_setup_multiply_nd_f32(
      op79,
      4, a_shape, 4, b_shape,
      v76 /* a */, v79 /* b */, v80 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #79" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op80,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v80 /* input */, v81 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #80" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 96 };
    const size_t b_shape[] = { 1, 7, 7, 96 };
    status = xnn_setup_add_nd_f32(
      op81,
      4, a_shape, 4, b_shape,
      v81 /* a */, v72 /* b */, v82 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #81" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op82,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v82 /* input */, v83 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #82" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op83,
    49 /* batch size */,
    v83 /* input */, v84 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #83" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op84,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v84 /* input */, v85 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #84" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op85,
    49 /* batch size */,
    v85 /* input */, v86 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #85" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op86,
    1 /* batch size */, 49 /* width */,
    v86 /* input */, v87 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #86" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op87,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v87 /* input */, v88 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #87" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op88,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v88 /* input */, v89 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #88" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 576 };
    const size_t b_shape[] = { 1, 1, 1, 576 };
    status = xnn_setup_multiply_nd_f32(
      op89,
      4, a_shape, 4, b_shape,
      v86 /* a */, v89 /* b */, v90 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #89" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op90,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v90 /* input */, v91 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #90" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 96 };
    const size_t b_shape[] = { 1, 7, 7, 96 };
    status = xnn_setup_add_nd_f32(
      op91,
      4, a_shape, 4, b_shape,
      v91 /* a */, v82 /* b */, v92 /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #91" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op92,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v92 /* input */, v93 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #92" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op93,
    49 /* batch size */,
    v93 /* input */, v94 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #93" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op94,
    1 /* batch size */, 49 /* width */,
    v94 /* input */, v95 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #94" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op95,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v95 /* input */, v96 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #95" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op96,
    1 /* batch size */,
    v96 /* input */, v97 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #96" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op97,
    1 /* batch size */, 1 /* width */,
    v97 /* input */, v98 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #97" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op98,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v98 /* input */, v99 /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #98" << std::endl;
    return ExecutionPlan();
  }

  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wpessimizing-move"
  return operators;
  #pragma clang diagnostic pop
}

}  // namespace models
