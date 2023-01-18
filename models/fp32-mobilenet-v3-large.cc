// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include <array>
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include <xnnpack/cache.h>
#include <xnnpack/common.h>

#include "models/models.h"

namespace models {

ExecutionPlan FP32MobileNetV3Large(pthreadpool_t threadpool) {
  alignas(16) static std::array<float, 150528 + XNN_EXTRA_BYTES / sizeof(float)> v0;
  alignas(16) static std::array<float, 200704 + XNN_EXTRA_BYTES / sizeof(float)> v1;
  alignas(16) static std::array<float, 200704 + XNN_EXTRA_BYTES / sizeof(float)> v2;
  alignas(16) static std::array<float, 200704 + XNN_EXTRA_BYTES / sizeof(float)> v3;
  alignas(16) static std::array<float, 200704 + XNN_EXTRA_BYTES / sizeof(float)> v4;
  alignas(16) static std::array<float, 200704 + XNN_EXTRA_BYTES / sizeof(float)> v5;
  alignas(16) static std::array<float, 802816 + XNN_EXTRA_BYTES / sizeof(float)> v6;
  alignas(16) static std::array<float, 200704 + XNN_EXTRA_BYTES / sizeof(float)> v7;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> v8;
  alignas(16) static std::array<float, 225792 + XNN_EXTRA_BYTES / sizeof(float)> v9;
  alignas(16) static std::array<float, 225792 + XNN_EXTRA_BYTES / sizeof(float)> v10;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> v11;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> v12;
  alignas(16) static std::array<float, 225792 + XNN_EXTRA_BYTES / sizeof(float)> v13;
  alignas(16) static std::array<float, 56448 + XNN_EXTRA_BYTES / sizeof(float)> v14;
  alignas(16) static std::array<float, 72 + XNN_EXTRA_BYTES / sizeof(float)> v15;
  alignas(16) static std::array<float, 24 + XNN_EXTRA_BYTES / sizeof(float)> v16;
  alignas(16) static std::array<float, 72 + XNN_EXTRA_BYTES / sizeof(float)> v17;
  alignas(16) static std::array<float, 56448 + XNN_EXTRA_BYTES / sizeof(float)> v18;
  alignas(16) static std::array<float, 31360 + XNN_EXTRA_BYTES / sizeof(float)> v19;
  alignas(16) static std::array<float, 94080 + XNN_EXTRA_BYTES / sizeof(float)> v20;
  alignas(16) static std::array<float, 94080 + XNN_EXTRA_BYTES / sizeof(float)> v21;
  alignas(16) static std::array<float, 120 + XNN_EXTRA_BYTES / sizeof(float)> v22;
  alignas(16) static std::array<float, 32 + XNN_EXTRA_BYTES / sizeof(float)> v23;
  alignas(16) static std::array<float, 120 + XNN_EXTRA_BYTES / sizeof(float)> v24;
  alignas(16) static std::array<float, 94080 + XNN_EXTRA_BYTES / sizeof(float)> v25;
  alignas(16) static std::array<float, 31360 + XNN_EXTRA_BYTES / sizeof(float)> v26;
  alignas(16) static std::array<float, 31360 + XNN_EXTRA_BYTES / sizeof(float)> v27;
  alignas(16) static std::array<float, 94080 + XNN_EXTRA_BYTES / sizeof(float)> v28;
  alignas(16) static std::array<float, 94080 + XNN_EXTRA_BYTES / sizeof(float)> v29;
  alignas(16) static std::array<float, 120 + XNN_EXTRA_BYTES / sizeof(float)> v30;
  alignas(16) static std::array<float, 32 + XNN_EXTRA_BYTES / sizeof(float)> v31;
  alignas(16) static std::array<float, 120 + XNN_EXTRA_BYTES / sizeof(float)> v32;
  alignas(16) static std::array<float, 94080 + XNN_EXTRA_BYTES / sizeof(float)> v33;
  alignas(16) static std::array<float, 31360 + XNN_EXTRA_BYTES / sizeof(float)> v34;
  alignas(16) static std::array<float, 31360 + XNN_EXTRA_BYTES / sizeof(float)> v35;
  alignas(16) static std::array<float, 188160 + XNN_EXTRA_BYTES / sizeof(float)> v36;
  alignas(16) static std::array<float, 188160 + XNN_EXTRA_BYTES / sizeof(float)> v37;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v38;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v39;
  alignas(16) static std::array<float, 15680 + XNN_EXTRA_BYTES / sizeof(float)> v40;
  alignas(16) static std::array<float, 39200 + XNN_EXTRA_BYTES / sizeof(float)> v41;
  alignas(16) static std::array<float, 39200 + XNN_EXTRA_BYTES / sizeof(float)> v42;
  alignas(16) static std::array<float, 39200 + XNN_EXTRA_BYTES / sizeof(float)> v43;
  alignas(16) static std::array<float, 39200 + XNN_EXTRA_BYTES / sizeof(float)> v44;
  alignas(16) static std::array<float, 15680 + XNN_EXTRA_BYTES / sizeof(float)> v45;
  alignas(16) static std::array<float, 15680 + XNN_EXTRA_BYTES / sizeof(float)> v46;
  alignas(16) static std::array<float, 36064 + XNN_EXTRA_BYTES / sizeof(float)> v47;
  alignas(16) static std::array<float, 36064 + XNN_EXTRA_BYTES / sizeof(float)> v48;
  alignas(16) static std::array<float, 36064 + XNN_EXTRA_BYTES / sizeof(float)> v49;
  alignas(16) static std::array<float, 36064 + XNN_EXTRA_BYTES / sizeof(float)> v50;
  alignas(16) static std::array<float, 15680 + XNN_EXTRA_BYTES / sizeof(float)> v51;
  alignas(16) static std::array<float, 15680 + XNN_EXTRA_BYTES / sizeof(float)> v52;
  alignas(16) static std::array<float, 36064 + XNN_EXTRA_BYTES / sizeof(float)> v53;
  alignas(16) static std::array<float, 36064 + XNN_EXTRA_BYTES / sizeof(float)> v54;
  alignas(16) static std::array<float, 36064 + XNN_EXTRA_BYTES / sizeof(float)> v55;
  alignas(16) static std::array<float, 36064 + XNN_EXTRA_BYTES / sizeof(float)> v56;
  alignas(16) static std::array<float, 15680 + XNN_EXTRA_BYTES / sizeof(float)> v57;
  alignas(16) static std::array<float, 15680 + XNN_EXTRA_BYTES / sizeof(float)> v58;
  alignas(16) static std::array<float, 94080 + XNN_EXTRA_BYTES / sizeof(float)> v59;
  alignas(16) static std::array<float, 94080 + XNN_EXTRA_BYTES / sizeof(float)> v60;
  alignas(16) static std::array<float, 94080 + XNN_EXTRA_BYTES / sizeof(float)> v61;
  alignas(16) static std::array<float, 94080 + XNN_EXTRA_BYTES / sizeof(float)> v62;
  alignas(16) static std::array<float, 480 + XNN_EXTRA_BYTES / sizeof(float)> v63;
  alignas(16) static std::array<float, 120 + XNN_EXTRA_BYTES / sizeof(float)> v64;
  alignas(16) static std::array<float, 480 + XNN_EXTRA_BYTES / sizeof(float)> v65;
  alignas(16) static std::array<float, 94080 + XNN_EXTRA_BYTES / sizeof(float)> v66;
  alignas(16) static std::array<float, 21952 + XNN_EXTRA_BYTES / sizeof(float)> v67;
  alignas(16) static std::array<float, 131712 + XNN_EXTRA_BYTES / sizeof(float)> v68;
  alignas(16) static std::array<float, 131712 + XNN_EXTRA_BYTES / sizeof(float)> v69;
  alignas(16) static std::array<float, 131712 + XNN_EXTRA_BYTES / sizeof(float)> v70;
  alignas(16) static std::array<float, 131712 + XNN_EXTRA_BYTES / sizeof(float)> v71;
  alignas(16) static std::array<float, 672 + XNN_EXTRA_BYTES / sizeof(float)> v72;
  alignas(16) static std::array<float, 168 + XNN_EXTRA_BYTES / sizeof(float)> v73;
  alignas(16) static std::array<float, 672 + XNN_EXTRA_BYTES / sizeof(float)> v74;
  alignas(16) static std::array<float, 131712 + XNN_EXTRA_BYTES / sizeof(float)> v75;
  alignas(16) static std::array<float, 21952 + XNN_EXTRA_BYTES / sizeof(float)> v76;
  alignas(16) static std::array<float, 21952 + XNN_EXTRA_BYTES / sizeof(float)> v77;
  alignas(16) static std::array<float, 131712 + XNN_EXTRA_BYTES / sizeof(float)> v78;
  alignas(16) static std::array<float, 131712 + XNN_EXTRA_BYTES / sizeof(float)> v79;
  alignas(16) static std::array<float, 32928 + XNN_EXTRA_BYTES / sizeof(float)> v80;
  alignas(16) static std::array<float, 32928 + XNN_EXTRA_BYTES / sizeof(float)> v81;
  alignas(16) static std::array<float, 672 + XNN_EXTRA_BYTES / sizeof(float)> v82;
  alignas(16) static std::array<float, 168 + XNN_EXTRA_BYTES / sizeof(float)> v83;
  alignas(16) static std::array<float, 672 + XNN_EXTRA_BYTES / sizeof(float)> v84;
  alignas(16) static std::array<float, 32928 + XNN_EXTRA_BYTES / sizeof(float)> v85;
  alignas(16) static std::array<float, 7840 + XNN_EXTRA_BYTES / sizeof(float)> v86;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v87;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v88;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v89;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v90;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> v91;
  alignas(16) static std::array<float, 240 + XNN_EXTRA_BYTES / sizeof(float)> v92;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> v93;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v94;
  alignas(16) static std::array<float, 7840 + XNN_EXTRA_BYTES / sizeof(float)> v95;
  alignas(16) static std::array<float, 7840 + XNN_EXTRA_BYTES / sizeof(float)> v96;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v97;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v98;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v99;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v100;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> v101;
  alignas(16) static std::array<float, 240 + XNN_EXTRA_BYTES / sizeof(float)> v102;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> v103;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v104;
  alignas(16) static std::array<float, 7840 + XNN_EXTRA_BYTES / sizeof(float)> v105;
  alignas(16) static std::array<float, 7840 + XNN_EXTRA_BYTES / sizeof(float)> v106;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v107;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v108;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> v109;
  alignas(16) static std::array<float, 1280 + XNN_EXTRA_BYTES / sizeof(float)> v110;
  alignas(16) static std::array<float, 1280 + XNN_EXTRA_BYTES / sizeof(float)> v111;
  alignas(16) static std::array<float, 1280 + XNN_EXTRA_BYTES / sizeof(float)> v112;
  alignas(16) static std::array<float, 1001 + XNN_EXTRA_BYTES / sizeof(float)> v113;
  alignas(16) static std::array<float, 432 + XNN_EXTRA_BYTES / sizeof(float)> w114;
  alignas(16) static std::array<float, 16 + XNN_EXTRA_BYTES / sizeof(float)> w115;
  alignas(16) static std::array<float, 144 + XNN_EXTRA_BYTES / sizeof(float)> w116;
  alignas(16) static std::array<float, 16 + XNN_EXTRA_BYTES / sizeof(float)> w117;
  alignas(16) static std::array<float, 256 + XNN_EXTRA_BYTES / sizeof(float)> w118;
  alignas(16) static std::array<float, 16 + XNN_EXTRA_BYTES / sizeof(float)> w119;
  alignas(16) static std::array<float, 1024 + XNN_EXTRA_BYTES / sizeof(float)> w120;
  alignas(16) static std::array<float, 64 + XNN_EXTRA_BYTES / sizeof(float)> w121;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> w122;
  alignas(16) static std::array<float, 64 + XNN_EXTRA_BYTES / sizeof(float)> w123;
  alignas(16) static std::array<float, 1536 + XNN_EXTRA_BYTES / sizeof(float)> w124;
  alignas(16) static std::array<float, 24 + XNN_EXTRA_BYTES / sizeof(float)> w125;
  alignas(16) static std::array<float, 1728 + XNN_EXTRA_BYTES / sizeof(float)> w126;
  alignas(16) static std::array<float, 72 + XNN_EXTRA_BYTES / sizeof(float)> w127;
  alignas(16) static std::array<float, 648 + XNN_EXTRA_BYTES / sizeof(float)> w128;
  alignas(16) static std::array<float, 72 + XNN_EXTRA_BYTES / sizeof(float)> w129;
  alignas(16) static std::array<float, 1728 + XNN_EXTRA_BYTES / sizeof(float)> w130;
  alignas(16) static std::array<float, 24 + XNN_EXTRA_BYTES / sizeof(float)> w131;
  alignas(16) static std::array<float, 1728 + XNN_EXTRA_BYTES / sizeof(float)> w132;
  alignas(16) static std::array<float, 72 + XNN_EXTRA_BYTES / sizeof(float)> w133;
  alignas(16) static std::array<float, 1800 + XNN_EXTRA_BYTES / sizeof(float)> w134;
  alignas(16) static std::array<float, 72 + XNN_EXTRA_BYTES / sizeof(float)> w135;
  alignas(16) static std::array<float, 1728 + XNN_EXTRA_BYTES / sizeof(float)> w136;
  alignas(16) static std::array<float, 24 + XNN_EXTRA_BYTES / sizeof(float)> w137;
  alignas(16) static std::array<float, 1728 + XNN_EXTRA_BYTES / sizeof(float)> w138;
  alignas(16) static std::array<float, 72 + XNN_EXTRA_BYTES / sizeof(float)> w139;
  alignas(16) static std::array<float, 2880 + XNN_EXTRA_BYTES / sizeof(float)> w140;
  alignas(16) static std::array<float, 40 + XNN_EXTRA_BYTES / sizeof(float)> w141;
  alignas(16) static std::array<float, 4800 + XNN_EXTRA_BYTES / sizeof(float)> w142;
  alignas(16) static std::array<float, 120 + XNN_EXTRA_BYTES / sizeof(float)> w143;
  alignas(16) static std::array<float, 3000 + XNN_EXTRA_BYTES / sizeof(float)> w144;
  alignas(16) static std::array<float, 120 + XNN_EXTRA_BYTES / sizeof(float)> w145;
  alignas(16) static std::array<float, 3840 + XNN_EXTRA_BYTES / sizeof(float)> w146;
  alignas(16) static std::array<float, 32 + XNN_EXTRA_BYTES / sizeof(float)> w147;
  alignas(16) static std::array<float, 3840 + XNN_EXTRA_BYTES / sizeof(float)> w148;
  alignas(16) static std::array<float, 120 + XNN_EXTRA_BYTES / sizeof(float)> w149;
  alignas(16) static std::array<float, 4800 + XNN_EXTRA_BYTES / sizeof(float)> w150;
  alignas(16) static std::array<float, 40 + XNN_EXTRA_BYTES / sizeof(float)> w151;
  alignas(16) static std::array<float, 4800 + XNN_EXTRA_BYTES / sizeof(float)> w152;
  alignas(16) static std::array<float, 120 + XNN_EXTRA_BYTES / sizeof(float)> w153;
  alignas(16) static std::array<float, 3000 + XNN_EXTRA_BYTES / sizeof(float)> w154;
  alignas(16) static std::array<float, 120 + XNN_EXTRA_BYTES / sizeof(float)> w155;
  alignas(16) static std::array<float, 3840 + XNN_EXTRA_BYTES / sizeof(float)> w156;
  alignas(16) static std::array<float, 32 + XNN_EXTRA_BYTES / sizeof(float)> w157;
  alignas(16) static std::array<float, 3840 + XNN_EXTRA_BYTES / sizeof(float)> w158;
  alignas(16) static std::array<float, 120 + XNN_EXTRA_BYTES / sizeof(float)> w159;
  alignas(16) static std::array<float, 4800 + XNN_EXTRA_BYTES / sizeof(float)> w160;
  alignas(16) static std::array<float, 40 + XNN_EXTRA_BYTES / sizeof(float)> w161;
  alignas(16) static std::array<float, 9600 + XNN_EXTRA_BYTES / sizeof(float)> w162;
  alignas(16) static std::array<float, 240 + XNN_EXTRA_BYTES / sizeof(float)> w163;
  alignas(16) static std::array<float, 2160 + XNN_EXTRA_BYTES / sizeof(float)> w164;
  alignas(16) static std::array<float, 240 + XNN_EXTRA_BYTES / sizeof(float)> w165;
  alignas(16) static std::array<float, 19200 + XNN_EXTRA_BYTES / sizeof(float)> w166;
  alignas(16) static std::array<float, 80 + XNN_EXTRA_BYTES / sizeof(float)> w167;
  alignas(16) static std::array<float, 16000 + XNN_EXTRA_BYTES / sizeof(float)> w168;
  alignas(16) static std::array<float, 200 + XNN_EXTRA_BYTES / sizeof(float)> w169;
  alignas(16) static std::array<float, 1800 + XNN_EXTRA_BYTES / sizeof(float)> w170;
  alignas(16) static std::array<float, 200 + XNN_EXTRA_BYTES / sizeof(float)> w171;
  alignas(16) static std::array<float, 16000 + XNN_EXTRA_BYTES / sizeof(float)> w172;
  alignas(16) static std::array<float, 80 + XNN_EXTRA_BYTES / sizeof(float)> w173;
  alignas(16) static std::array<float, 14720 + XNN_EXTRA_BYTES / sizeof(float)> w174;
  alignas(16) static std::array<float, 184 + XNN_EXTRA_BYTES / sizeof(float)> w175;
  alignas(16) static std::array<float, 1656 + XNN_EXTRA_BYTES / sizeof(float)> w176;
  alignas(16) static std::array<float, 184 + XNN_EXTRA_BYTES / sizeof(float)> w177;
  alignas(16) static std::array<float, 14720 + XNN_EXTRA_BYTES / sizeof(float)> w178;
  alignas(16) static std::array<float, 80 + XNN_EXTRA_BYTES / sizeof(float)> w179;
  alignas(16) static std::array<float, 14720 + XNN_EXTRA_BYTES / sizeof(float)> w180;
  alignas(16) static std::array<float, 184 + XNN_EXTRA_BYTES / sizeof(float)> w181;
  alignas(16) static std::array<float, 1656 + XNN_EXTRA_BYTES / sizeof(float)> w182;
  alignas(16) static std::array<float, 184 + XNN_EXTRA_BYTES / sizeof(float)> w183;
  alignas(16) static std::array<float, 14720 + XNN_EXTRA_BYTES / sizeof(float)> w184;
  alignas(16) static std::array<float, 80 + XNN_EXTRA_BYTES / sizeof(float)> w185;
  alignas(16) static std::array<float, 38400 + XNN_EXTRA_BYTES / sizeof(float)> w186;
  alignas(16) static std::array<float, 480 + XNN_EXTRA_BYTES / sizeof(float)> w187;
  alignas(16) static std::array<float, 4320 + XNN_EXTRA_BYTES / sizeof(float)> w188;
  alignas(16) static std::array<float, 480 + XNN_EXTRA_BYTES / sizeof(float)> w189;
  alignas(16) static std::array<float, 57600 + XNN_EXTRA_BYTES / sizeof(float)> w190;
  alignas(16) static std::array<float, 120 + XNN_EXTRA_BYTES / sizeof(float)> w191;
  alignas(16) static std::array<float, 57600 + XNN_EXTRA_BYTES / sizeof(float)> w192;
  alignas(16) static std::array<float, 480 + XNN_EXTRA_BYTES / sizeof(float)> w193;
  alignas(16) static std::array<float, 53760 + XNN_EXTRA_BYTES / sizeof(float)> w194;
  alignas(16) static std::array<float, 112 + XNN_EXTRA_BYTES / sizeof(float)> w195;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> w196;
  alignas(16) static std::array<float, 672 + XNN_EXTRA_BYTES / sizeof(float)> w197;
  alignas(16) static std::array<float, 6048 + XNN_EXTRA_BYTES / sizeof(float)> w198;
  alignas(16) static std::array<float, 672 + XNN_EXTRA_BYTES / sizeof(float)> w199;
  alignas(16) static std::array<float, 112896 + XNN_EXTRA_BYTES / sizeof(float)> w200;
  alignas(16) static std::array<float, 168 + XNN_EXTRA_BYTES / sizeof(float)> w201;
  alignas(16) static std::array<float, 112896 + XNN_EXTRA_BYTES / sizeof(float)> w202;
  alignas(16) static std::array<float, 672 + XNN_EXTRA_BYTES / sizeof(float)> w203;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> w204;
  alignas(16) static std::array<float, 112 + XNN_EXTRA_BYTES / sizeof(float)> w205;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> w206;
  alignas(16) static std::array<float, 672 + XNN_EXTRA_BYTES / sizeof(float)> w207;
  alignas(16) static std::array<float, 16800 + XNN_EXTRA_BYTES / sizeof(float)> w208;
  alignas(16) static std::array<float, 672 + XNN_EXTRA_BYTES / sizeof(float)> w209;
  alignas(16) static std::array<float, 112896 + XNN_EXTRA_BYTES / sizeof(float)> w210;
  alignas(16) static std::array<float, 168 + XNN_EXTRA_BYTES / sizeof(float)> w211;
  alignas(16) static std::array<float, 112896 + XNN_EXTRA_BYTES / sizeof(float)> w212;
  alignas(16) static std::array<float, 672 + XNN_EXTRA_BYTES / sizeof(float)> w213;
  alignas(16) static std::array<float, 107520 + XNN_EXTRA_BYTES / sizeof(float)> w214;
  alignas(16) static std::array<float, 160 + XNN_EXTRA_BYTES / sizeof(float)> w215;
  alignas(16) static std::array<float, 153600 + XNN_EXTRA_BYTES / sizeof(float)> w216;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> w217;
  alignas(16) static std::array<float, 24000 + XNN_EXTRA_BYTES / sizeof(float)> w218;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> w219;
  alignas(16) static std::array<float, 230400 + XNN_EXTRA_BYTES / sizeof(float)> w220;
  alignas(16) static std::array<float, 240 + XNN_EXTRA_BYTES / sizeof(float)> w221;
  alignas(16) static std::array<float, 230400 + XNN_EXTRA_BYTES / sizeof(float)> w222;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> w223;
  alignas(16) static std::array<float, 153600 + XNN_EXTRA_BYTES / sizeof(float)> w224;
  alignas(16) static std::array<float, 160 + XNN_EXTRA_BYTES / sizeof(float)> w225;
  alignas(16) static std::array<float, 153600 + XNN_EXTRA_BYTES / sizeof(float)> w226;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> w227;
  alignas(16) static std::array<float, 24000 + XNN_EXTRA_BYTES / sizeof(float)> w228;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> w229;
  alignas(16) static std::array<float, 230400 + XNN_EXTRA_BYTES / sizeof(float)> w230;
  alignas(16) static std::array<float, 240 + XNN_EXTRA_BYTES / sizeof(float)> w231;
  alignas(16) static std::array<float, 230400 + XNN_EXTRA_BYTES / sizeof(float)> w232;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> w233;
  alignas(16) static std::array<float, 153600 + XNN_EXTRA_BYTES / sizeof(float)> w234;
  alignas(16) static std::array<float, 160 + XNN_EXTRA_BYTES / sizeof(float)> w235;
  alignas(16) static std::array<float, 153600 + XNN_EXTRA_BYTES / sizeof(float)> w236;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> w237;
  alignas(16) static std::array<float, 1228800 + XNN_EXTRA_BYTES / sizeof(float)> w238;
  alignas(16) static std::array<float, 1280 + XNN_EXTRA_BYTES / sizeof(float)> w239;
  alignas(16) static std::array<float, 1281280 + XNN_EXTRA_BYTES / sizeof(float)> w240;
  alignas(16) static std::array<float, 1001 + XNN_EXTRA_BYTES / sizeof(float)> w241;

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), std::ref(rng));
  std::generate(v0.begin(), v0.end(), std::ref(f32rng));
  std::generate(v1.begin(), v1.end(), std::ref(f32rng));
  std::generate(v2.begin(), v2.end(), std::ref(f32rng));
  std::generate(v3.begin(), v3.end(), std::ref(f32rng));
  std::generate(v4.begin(), v4.end(), std::ref(f32rng));
  std::generate(v5.begin(), v5.end(), std::ref(f32rng));
  std::generate(v6.begin(), v6.end(), std::ref(f32rng));
  std::generate(v7.begin(), v7.end(), std::ref(f32rng));
  std::generate(v8.begin(), v8.end(), std::ref(f32rng));
  std::generate(v9.begin(), v9.end(), std::ref(f32rng));
  std::generate(v10.begin(), v10.end(), std::ref(f32rng));
  std::generate(v11.begin(), v11.end(), std::ref(f32rng));
  std::generate(v12.begin(), v12.end(), std::ref(f32rng));
  std::generate(v13.begin(), v13.end(), std::ref(f32rng));
  std::generate(v14.begin(), v14.end(), std::ref(f32rng));
  std::generate(v15.begin(), v15.end(), std::ref(f32rng));
  std::generate(v16.begin(), v16.end(), std::ref(f32rng));
  std::generate(v17.begin(), v17.end(), std::ref(f32rng));
  std::generate(v18.begin(), v18.end(), std::ref(f32rng));
  std::generate(v19.begin(), v19.end(), std::ref(f32rng));
  std::generate(v20.begin(), v20.end(), std::ref(f32rng));
  std::generate(v21.begin(), v21.end(), std::ref(f32rng));
  std::generate(v22.begin(), v22.end(), std::ref(f32rng));
  std::generate(v23.begin(), v23.end(), std::ref(f32rng));
  std::generate(v24.begin(), v24.end(), std::ref(f32rng));
  std::generate(v25.begin(), v25.end(), std::ref(f32rng));
  std::generate(v26.begin(), v26.end(), std::ref(f32rng));
  std::generate(v27.begin(), v27.end(), std::ref(f32rng));
  std::generate(v28.begin(), v28.end(), std::ref(f32rng));
  std::generate(v29.begin(), v29.end(), std::ref(f32rng));
  std::generate(v30.begin(), v30.end(), std::ref(f32rng));
  std::generate(v31.begin(), v31.end(), std::ref(f32rng));
  std::generate(v32.begin(), v32.end(), std::ref(f32rng));
  std::generate(v33.begin(), v33.end(), std::ref(f32rng));
  std::generate(v34.begin(), v34.end(), std::ref(f32rng));
  std::generate(v35.begin(), v35.end(), std::ref(f32rng));
  std::generate(v36.begin(), v36.end(), std::ref(f32rng));
  std::generate(v37.begin(), v37.end(), std::ref(f32rng));
  std::generate(v38.begin(), v38.end(), std::ref(f32rng));
  std::generate(v39.begin(), v39.end(), std::ref(f32rng));
  std::generate(v40.begin(), v40.end(), std::ref(f32rng));
  std::generate(v41.begin(), v41.end(), std::ref(f32rng));
  std::generate(v42.begin(), v42.end(), std::ref(f32rng));
  std::generate(v43.begin(), v43.end(), std::ref(f32rng));
  std::generate(v44.begin(), v44.end(), std::ref(f32rng));
  std::generate(v45.begin(), v45.end(), std::ref(f32rng));
  std::generate(v46.begin(), v46.end(), std::ref(f32rng));
  std::generate(v47.begin(), v47.end(), std::ref(f32rng));
  std::generate(v48.begin(), v48.end(), std::ref(f32rng));
  std::generate(v49.begin(), v49.end(), std::ref(f32rng));
  std::generate(v50.begin(), v50.end(), std::ref(f32rng));
  std::generate(v51.begin(), v51.end(), std::ref(f32rng));
  std::generate(v52.begin(), v52.end(), std::ref(f32rng));
  std::generate(v53.begin(), v53.end(), std::ref(f32rng));
  std::generate(v54.begin(), v54.end(), std::ref(f32rng));
  std::generate(v55.begin(), v55.end(), std::ref(f32rng));
  std::generate(v56.begin(), v56.end(), std::ref(f32rng));
  std::generate(v57.begin(), v57.end(), std::ref(f32rng));
  std::generate(v58.begin(), v58.end(), std::ref(f32rng));
  std::generate(v59.begin(), v59.end(), std::ref(f32rng));
  std::generate(v60.begin(), v60.end(), std::ref(f32rng));
  std::generate(v61.begin(), v61.end(), std::ref(f32rng));
  std::generate(v62.begin(), v62.end(), std::ref(f32rng));
  std::generate(v63.begin(), v63.end(), std::ref(f32rng));
  std::generate(v64.begin(), v64.end(), std::ref(f32rng));
  std::generate(v65.begin(), v65.end(), std::ref(f32rng));
  std::generate(v66.begin(), v66.end(), std::ref(f32rng));
  std::generate(v67.begin(), v67.end(), std::ref(f32rng));
  std::generate(v68.begin(), v68.end(), std::ref(f32rng));
  std::generate(v69.begin(), v69.end(), std::ref(f32rng));
  std::generate(v70.begin(), v70.end(), std::ref(f32rng));
  std::generate(v71.begin(), v71.end(), std::ref(f32rng));
  std::generate(v72.begin(), v72.end(), std::ref(f32rng));
  std::generate(v73.begin(), v73.end(), std::ref(f32rng));
  std::generate(v74.begin(), v74.end(), std::ref(f32rng));
  std::generate(v75.begin(), v75.end(), std::ref(f32rng));
  std::generate(v76.begin(), v76.end(), std::ref(f32rng));
  std::generate(v77.begin(), v77.end(), std::ref(f32rng));
  std::generate(v78.begin(), v78.end(), std::ref(f32rng));
  std::generate(v79.begin(), v79.end(), std::ref(f32rng));
  std::generate(v80.begin(), v80.end(), std::ref(f32rng));
  std::generate(v81.begin(), v81.end(), std::ref(f32rng));
  std::generate(v82.begin(), v82.end(), std::ref(f32rng));
  std::generate(v83.begin(), v83.end(), std::ref(f32rng));
  std::generate(v84.begin(), v84.end(), std::ref(f32rng));
  std::generate(v85.begin(), v85.end(), std::ref(f32rng));
  std::generate(v86.begin(), v86.end(), std::ref(f32rng));
  std::generate(v87.begin(), v87.end(), std::ref(f32rng));
  std::generate(v88.begin(), v88.end(), std::ref(f32rng));
  std::generate(v89.begin(), v89.end(), std::ref(f32rng));
  std::generate(v90.begin(), v90.end(), std::ref(f32rng));
  std::generate(v91.begin(), v91.end(), std::ref(f32rng));
  std::generate(v92.begin(), v92.end(), std::ref(f32rng));
  std::generate(v93.begin(), v93.end(), std::ref(f32rng));
  std::generate(v94.begin(), v94.end(), std::ref(f32rng));
  std::generate(v95.begin(), v95.end(), std::ref(f32rng));
  std::generate(v96.begin(), v96.end(), std::ref(f32rng));
  std::generate(v97.begin(), v97.end(), std::ref(f32rng));
  std::generate(v98.begin(), v98.end(), std::ref(f32rng));
  std::generate(v99.begin(), v99.end(), std::ref(f32rng));
  std::generate(v100.begin(), v100.end(), std::ref(f32rng));
  std::generate(v101.begin(), v101.end(), std::ref(f32rng));
  std::generate(v102.begin(), v102.end(), std::ref(f32rng));
  std::generate(v103.begin(), v103.end(), std::ref(f32rng));
  std::generate(v104.begin(), v104.end(), std::ref(f32rng));
  std::generate(v105.begin(), v105.end(), std::ref(f32rng));
  std::generate(v106.begin(), v106.end(), std::ref(f32rng));
  std::generate(v107.begin(), v107.end(), std::ref(f32rng));
  std::generate(v108.begin(), v108.end(), std::ref(f32rng));
  std::generate(v109.begin(), v109.end(), std::ref(f32rng));
  std::generate(v110.begin(), v110.end(), std::ref(f32rng));
  std::generate(v111.begin(), v111.end(), std::ref(f32rng));
  std::generate(v112.begin(), v112.end(), std::ref(f32rng));
  std::generate(v113.begin(), v113.end(), std::ref(f32rng));
  std::generate(w114.begin(), w114.end(), std::ref(f32rng));
  std::generate(w115.begin(), w115.end(), std::ref(f32rng));
  std::generate(w116.begin(), w116.end(), std::ref(f32rng));
  std::generate(w117.begin(), w117.end(), std::ref(f32rng));
  std::generate(w118.begin(), w118.end(), std::ref(f32rng));
  std::generate(w119.begin(), w119.end(), std::ref(f32rng));
  std::generate(w120.begin(), w120.end(), std::ref(f32rng));
  std::generate(w121.begin(), w121.end(), std::ref(f32rng));
  std::generate(w122.begin(), w122.end(), std::ref(f32rng));
  std::generate(w123.begin(), w123.end(), std::ref(f32rng));
  std::generate(w124.begin(), w124.end(), std::ref(f32rng));
  std::generate(w125.begin(), w125.end(), std::ref(f32rng));
  std::generate(w126.begin(), w126.end(), std::ref(f32rng));
  std::generate(w127.begin(), w127.end(), std::ref(f32rng));
  std::generate(w128.begin(), w128.end(), std::ref(f32rng));
  std::generate(w129.begin(), w129.end(), std::ref(f32rng));
  std::generate(w130.begin(), w130.end(), std::ref(f32rng));
  std::generate(w131.begin(), w131.end(), std::ref(f32rng));
  std::generate(w132.begin(), w132.end(), std::ref(f32rng));
  std::generate(w133.begin(), w133.end(), std::ref(f32rng));
  std::generate(w134.begin(), w134.end(), std::ref(f32rng));
  std::generate(w135.begin(), w135.end(), std::ref(f32rng));
  std::generate(w136.begin(), w136.end(), std::ref(f32rng));
  std::generate(w137.begin(), w137.end(), std::ref(f32rng));
  std::generate(w138.begin(), w138.end(), std::ref(f32rng));
  std::generate(w139.begin(), w139.end(), std::ref(f32rng));
  std::generate(w140.begin(), w140.end(), std::ref(f32rng));
  std::generate(w141.begin(), w141.end(), std::ref(f32rng));
  std::generate(w142.begin(), w142.end(), std::ref(f32rng));
  std::generate(w143.begin(), w143.end(), std::ref(f32rng));
  std::generate(w144.begin(), w144.end(), std::ref(f32rng));
  std::generate(w145.begin(), w145.end(), std::ref(f32rng));
  std::generate(w146.begin(), w146.end(), std::ref(f32rng));
  std::generate(w147.begin(), w147.end(), std::ref(f32rng));
  std::generate(w148.begin(), w148.end(), std::ref(f32rng));
  std::generate(w149.begin(), w149.end(), std::ref(f32rng));
  std::generate(w150.begin(), w150.end(), std::ref(f32rng));
  std::generate(w151.begin(), w151.end(), std::ref(f32rng));
  std::generate(w152.begin(), w152.end(), std::ref(f32rng));
  std::generate(w153.begin(), w153.end(), std::ref(f32rng));
  std::generate(w154.begin(), w154.end(), std::ref(f32rng));
  std::generate(w155.begin(), w155.end(), std::ref(f32rng));
  std::generate(w156.begin(), w156.end(), std::ref(f32rng));
  std::generate(w157.begin(), w157.end(), std::ref(f32rng));
  std::generate(w158.begin(), w158.end(), std::ref(f32rng));
  std::generate(w159.begin(), w159.end(), std::ref(f32rng));
  std::generate(w160.begin(), w160.end(), std::ref(f32rng));
  std::generate(w161.begin(), w161.end(), std::ref(f32rng));
  std::generate(w162.begin(), w162.end(), std::ref(f32rng));
  std::generate(w163.begin(), w163.end(), std::ref(f32rng));
  std::generate(w164.begin(), w164.end(), std::ref(f32rng));
  std::generate(w165.begin(), w165.end(), std::ref(f32rng));
  std::generate(w166.begin(), w166.end(), std::ref(f32rng));
  std::generate(w167.begin(), w167.end(), std::ref(f32rng));
  std::generate(w168.begin(), w168.end(), std::ref(f32rng));
  std::generate(w169.begin(), w169.end(), std::ref(f32rng));
  std::generate(w170.begin(), w170.end(), std::ref(f32rng));
  std::generate(w171.begin(), w171.end(), std::ref(f32rng));
  std::generate(w172.begin(), w172.end(), std::ref(f32rng));
  std::generate(w173.begin(), w173.end(), std::ref(f32rng));
  std::generate(w174.begin(), w174.end(), std::ref(f32rng));
  std::generate(w175.begin(), w175.end(), std::ref(f32rng));
  std::generate(w176.begin(), w176.end(), std::ref(f32rng));
  std::generate(w177.begin(), w177.end(), std::ref(f32rng));
  std::generate(w178.begin(), w178.end(), std::ref(f32rng));
  std::generate(w179.begin(), w179.end(), std::ref(f32rng));
  std::generate(w180.begin(), w180.end(), std::ref(f32rng));
  std::generate(w181.begin(), w181.end(), std::ref(f32rng));
  std::generate(w182.begin(), w182.end(), std::ref(f32rng));
  std::generate(w183.begin(), w183.end(), std::ref(f32rng));
  std::generate(w184.begin(), w184.end(), std::ref(f32rng));
  std::generate(w185.begin(), w185.end(), std::ref(f32rng));
  std::generate(w186.begin(), w186.end(), std::ref(f32rng));
  std::generate(w187.begin(), w187.end(), std::ref(f32rng));
  std::generate(w188.begin(), w188.end(), std::ref(f32rng));
  std::generate(w189.begin(), w189.end(), std::ref(f32rng));
  std::generate(w190.begin(), w190.end(), std::ref(f32rng));
  std::generate(w191.begin(), w191.end(), std::ref(f32rng));
  std::generate(w192.begin(), w192.end(), std::ref(f32rng));
  std::generate(w193.begin(), w193.end(), std::ref(f32rng));
  std::generate(w194.begin(), w194.end(), std::ref(f32rng));
  std::generate(w195.begin(), w195.end(), std::ref(f32rng));
  std::generate(w196.begin(), w196.end(), std::ref(f32rng));
  std::generate(w197.begin(), w197.end(), std::ref(f32rng));
  std::generate(w198.begin(), w198.end(), std::ref(f32rng));
  std::generate(w199.begin(), w199.end(), std::ref(f32rng));
  std::generate(w200.begin(), w200.end(), std::ref(f32rng));
  std::generate(w201.begin(), w201.end(), std::ref(f32rng));
  std::generate(w202.begin(), w202.end(), std::ref(f32rng));
  std::generate(w203.begin(), w203.end(), std::ref(f32rng));
  std::generate(w204.begin(), w204.end(), std::ref(f32rng));
  std::generate(w205.begin(), w205.end(), std::ref(f32rng));
  std::generate(w206.begin(), w206.end(), std::ref(f32rng));
  std::generate(w207.begin(), w207.end(), std::ref(f32rng));
  std::generate(w208.begin(), w208.end(), std::ref(f32rng));
  std::generate(w209.begin(), w209.end(), std::ref(f32rng));
  std::generate(w210.begin(), w210.end(), std::ref(f32rng));
  std::generate(w211.begin(), w211.end(), std::ref(f32rng));
  std::generate(w212.begin(), w212.end(), std::ref(f32rng));
  std::generate(w213.begin(), w213.end(), std::ref(f32rng));
  std::generate(w214.begin(), w214.end(), std::ref(f32rng));
  std::generate(w215.begin(), w215.end(), std::ref(f32rng));
  std::generate(w216.begin(), w216.end(), std::ref(f32rng));
  std::generate(w217.begin(), w217.end(), std::ref(f32rng));
  std::generate(w218.begin(), w218.end(), std::ref(f32rng));
  std::generate(w219.begin(), w219.end(), std::ref(f32rng));
  std::generate(w220.begin(), w220.end(), std::ref(f32rng));
  std::generate(w221.begin(), w221.end(), std::ref(f32rng));
  std::generate(w222.begin(), w222.end(), std::ref(f32rng));
  std::generate(w223.begin(), w223.end(), std::ref(f32rng));
  std::generate(w224.begin(), w224.end(), std::ref(f32rng));
  std::generate(w225.begin(), w225.end(), std::ref(f32rng));
  std::generate(w226.begin(), w226.end(), std::ref(f32rng));
  std::generate(w227.begin(), w227.end(), std::ref(f32rng));
  std::generate(w228.begin(), w228.end(), std::ref(f32rng));
  std::generate(w229.begin(), w229.end(), std::ref(f32rng));
  std::generate(w230.begin(), w230.end(), std::ref(f32rng));
  std::generate(w231.begin(), w231.end(), std::ref(f32rng));
  std::generate(w232.begin(), w232.end(), std::ref(f32rng));
  std::generate(w233.begin(), w233.end(), std::ref(f32rng));
  std::generate(w234.begin(), w234.end(), std::ref(f32rng));
  std::generate(w235.begin(), w235.end(), std::ref(f32rng));
  std::generate(w236.begin(), w236.end(), std::ref(f32rng));
  std::generate(w237.begin(), w237.end(), std::ref(f32rng));
  std::generate(w238.begin(), w238.end(), std::ref(f32rng));
  std::generate(w239.begin(), w239.end(), std::ref(f32rng));
  std::generate(w240.begin(), w240.end(), std::ref(f32rng));
  std::generate(w241.begin(), w241.end(), std::ref(f32rng));

  ExecutionPlan operators;
  xnn_status status;
  xnn_caches caches = {};
#if XNN_PLATFORM_JIT && XNN_ENABLE_JIT
  xnn_code_cache code_cache;
  xnn_init_code_cache(&code_cache);
  caches.code_cache = &code_cache;
#endif

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
    w114.data(), w115.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w116.data(), w117.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w118.data(), w119.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #3" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op3, xnn_delete_operator);

  xnn_operator_t op4 = nullptr;
  status = xnn_create_add_nd_f32(
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
    w120.data(), w121.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w122.data(), w123.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w124.data(), w125.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w126.data(), w127.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w128.data(), w129.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w130.data(), w131.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #10" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op10, xnn_delete_operator);

  xnn_operator_t op11 = nullptr;
  status = xnn_create_add_nd_f32(
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
    w132.data(), w133.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w134.data(), w135.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w136.data(), w137.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w138.data(), w139.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &caches,
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
    w140.data(), w141.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w142.data(), w143.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w144.data(), w145.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w146.data(), w147.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w148.data(), w149.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &caches,
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
    w150.data(), w151.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #25" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op25, xnn_delete_operator);

  xnn_operator_t op26 = nullptr;
  status = xnn_create_add_nd_f32(
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
    w152.data(), w153.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w154.data(), w155.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w156.data(), w157.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w158.data(), w159.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &caches,
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
    w160.data(), w161.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op33);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #33" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op33, xnn_delete_operator);

  xnn_operator_t op34 = nullptr;
  status = xnn_create_add_nd_f32(
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
    w162.data(), w163.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w164.data(), w165.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w166.data(), w167.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w168.data(), w169.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w170.data(), w171.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w172.data(), w173.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op44);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #44" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op44, xnn_delete_operator);

  xnn_operator_t op45 = nullptr;
  status = xnn_create_add_nd_f32(
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
    w174.data(), w175.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w176.data(), w177.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w178.data(), w179.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op50);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #50" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op50, xnn_delete_operator);

  xnn_operator_t op51 = nullptr;
  status = xnn_create_add_nd_f32(
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
    w180.data(), w181.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w182.data(), w183.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w184.data(), w185.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op56);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #56" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op56, xnn_delete_operator);

  xnn_operator_t op57 = nullptr;
  status = xnn_create_add_nd_f32(
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
    w186.data(), w187.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w188.data(), w189.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w190.data(), w191.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w192.data(), w193.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &caches,
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
    w194.data(), w195.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w196.data(), w197.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w198.data(), w199.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w200.data(), w201.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w202.data(), w203.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &caches,
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
    w204.data(), w205.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op75);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #75" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op75, xnn_delete_operator);

  xnn_operator_t op76 = nullptr;
  status = xnn_create_add_nd_f32(
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
    w206.data(), w207.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w208.data(), w209.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w210.data(), w211.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w212.data(), w213.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &caches,
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
    w214.data(), w215.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w216.data(), w217.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w218.data(), w219.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w220.data(), w221.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w222.data(), w223.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &caches,
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
    w224.data(), w225.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op94);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #94" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op94, xnn_delete_operator);

  xnn_operator_t op95 = nullptr;
  status = xnn_create_add_nd_f32(
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
    w226.data(), w227.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w228.data(), w229.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w230.data(), w231.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w232.data(), w233.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    &caches,
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
    w234.data(), w235.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op104);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #104" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op104, xnn_delete_operator);

  xnn_operator_t op105 = nullptr;
  status = xnn_create_add_nd_f32(
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
    w236.data(), w237.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w238.data(), w239.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
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
    w240.data(), w241.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op112);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #112" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op112, xnn_delete_operator);

#if XNN_PLATFORM_JIT && XNN_ENABLE_JIT
  xnn_finalize_code_memory(&code_cache.cache.code);
#endif

  status = xnn_setup_convolution2d_nhwc_f32(
    op0,
    1 /* batch size */, 224 /* input height */, 224 /* input width */,
    v0.data() /* input */, v1.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op1,
    12544 /* batch size */,
    v1.data() /* input */, v2.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #1" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op2,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v2.data() /* input */, v3.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op3,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v3.data() /* input */, v4.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #3" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 112, 112, 16 };
    const size_t b_shape[] = { 1, 112, 112, 16 };
    status = xnn_setup_add_nd_f32(
      op4,
      4, a_shape, 4, b_shape,
      v4.data() /* a */, v2.data() /* b */, v5.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op5,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v5.data() /* input */, v6.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #5" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op6,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v6.data() /* input */, v7.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op7,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v7.data() /* input */, v8.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op8,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v8.data() /* input */, v9.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #8" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op9,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v9.data() /* input */, v10.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op10,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v10.data() /* input */, v11.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #10" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 56, 56, 24 };
    const size_t b_shape[] = { 1, 56, 56, 24 };
    status = xnn_setup_add_nd_f32(
      op11,
      4, a_shape, 4, b_shape,
      v11.data() /* a */, v8.data() /* b */, v12.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op12,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v12.data() /* input */, v13.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op13,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v13.data() /* input */, v14.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #13" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op14,
    1 /* batch size */, 784 /* width */,
    v14.data() /* input */, v15.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op15,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v15.data() /* input */, v16.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #15" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op16,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v16.data() /* input */, v17.data() /* output */,
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
      v14.data() /* a */, v17.data() /* b */, v18.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op18,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v18.data() /* input */, v19.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op19,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v19.data() /* input */, v20.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #19" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op20,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v20.data() /* input */, v21.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op21,
    1 /* batch size */, 784 /* width */,
    v21.data() /* input */, v22.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #21" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op22,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v22.data() /* input */, v23.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op23,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v23.data() /* input */, v24.data() /* output */,
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
      v21.data() /* a */, v24.data() /* b */, v25.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op25,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v25.data() /* input */, v26.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #25" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 28, 28, 40 };
    const size_t b_shape[] = { 1, 28, 28, 40 };
    status = xnn_setup_add_nd_f32(
      op26,
      4, a_shape, 4, b_shape,
      v26.data() /* a */, v19.data() /* b */, v27.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #26" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op27,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v27.data() /* input */, v28.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op28,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v28.data() /* input */, v29.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #28" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op29,
    1 /* batch size */, 784 /* width */,
    v29.data() /* input */, v30.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #29" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op30,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v30.data() /* input */, v31.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #30" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op31,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v31.data() /* input */, v32.data() /* output */,
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
      v29.data() /* a */, v32.data() /* b */, v33.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #32" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op33,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v33.data() /* input */, v34.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #33" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 28, 28, 40 };
    const size_t b_shape[] = { 1, 28, 28, 40 };
    status = xnn_setup_add_nd_f32(
      op34,
      4, a_shape, 4, b_shape,
      v34.data() /* a */, v27.data() /* b */, v35.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #34" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op35,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v35.data() /* input */, v36.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #35" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op36,
    784 /* batch size */,
    v36.data() /* input */, v37.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #36" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op37,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v37.data() /* input */, v38.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #37" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op38,
    196 /* batch size */,
    v38.data() /* input */, v39.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #38" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op39,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v39.data() /* input */, v40.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #39" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op40,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v40.data() /* input */, v41.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #40" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op41,
    196 /* batch size */,
    v41.data() /* input */, v42.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #41" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op42,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v42.data() /* input */, v43.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #42" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op43,
    196 /* batch size */,
    v43.data() /* input */, v44.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #43" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op44,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v44.data() /* input */, v45.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #44" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 80 };
    const size_t b_shape[] = { 1, 14, 14, 80 };
    status = xnn_setup_add_nd_f32(
      op45,
      4, a_shape, 4, b_shape,
      v45.data() /* a */, v40.data() /* b */, v46.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #45" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op46,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v46.data() /* input */, v47.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #46" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op47,
    196 /* batch size */,
    v47.data() /* input */, v48.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #47" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op48,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v48.data() /* input */, v49.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #48" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op49,
    196 /* batch size */,
    v49.data() /* input */, v50.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #49" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op50,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v50.data() /* input */, v51.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #50" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 80 };
    const size_t b_shape[] = { 1, 14, 14, 80 };
    status = xnn_setup_add_nd_f32(
      op51,
      4, a_shape, 4, b_shape,
      v51.data() /* a */, v46.data() /* b */, v52.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #51" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op52,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v52.data() /* input */, v53.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #52" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op53,
    196 /* batch size */,
    v53.data() /* input */, v54.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #53" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op54,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v54.data() /* input */, v55.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #54" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op55,
    196 /* batch size */,
    v55.data() /* input */, v56.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #55" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op56,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v56.data() /* input */, v57.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #56" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 80 };
    const size_t b_shape[] = { 1, 14, 14, 80 };
    status = xnn_setup_add_nd_f32(
      op57,
      4, a_shape, 4, b_shape,
      v57.data() /* a */, v52.data() /* b */, v58.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #57" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op58,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v58.data() /* input */, v59.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #58" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op59,
    196 /* batch size */,
    v59.data() /* input */, v60.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #59" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op60,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v60.data() /* input */, v61.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #60" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op61,
    196 /* batch size */,
    v61.data() /* input */, v62.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #61" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op62,
    1 /* batch size */, 196 /* width */,
    v62.data() /* input */, v63.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #62" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op63,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v63.data() /* input */, v64.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #63" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op64,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v64.data() /* input */, v65.data() /* output */,
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
      v62.data() /* a */, v65.data() /* b */, v66.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #65" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op66,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v66.data() /* input */, v67.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #66" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op67,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v67.data() /* input */, v68.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #67" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op68,
    196 /* batch size */,
    v68.data() /* input */, v69.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #68" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op69,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v69.data() /* input */, v70.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #69" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op70,
    196 /* batch size */,
    v70.data() /* input */, v71.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #70" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op71,
    1 /* batch size */, 196 /* width */,
    v71.data() /* input */, v72.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #71" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op72,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v72.data() /* input */, v73.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #72" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op73,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v73.data() /* input */, v74.data() /* output */,
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
      v71.data() /* a */, v74.data() /* b */, v75.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #74" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op75,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v75.data() /* input */, v76.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #75" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 112 };
    const size_t b_shape[] = { 1, 14, 14, 112 };
    status = xnn_setup_add_nd_f32(
      op76,
      4, a_shape, 4, b_shape,
      v76.data() /* a */, v67.data() /* b */, v77.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #76" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op77,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v77.data() /* input */, v78.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #77" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op78,
    196 /* batch size */,
    v78.data() /* input */, v79.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #78" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op79,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v79.data() /* input */, v80.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #79" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op80,
    49 /* batch size */,
    v80.data() /* input */, v81.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #80" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op81,
    1 /* batch size */, 49 /* width */,
    v81.data() /* input */, v82.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #81" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op82,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v82.data() /* input */, v83.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #82" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op83,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v83.data() /* input */, v84.data() /* output */,
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
      v81.data() /* a */, v84.data() /* b */, v85.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #84" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op85,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v85.data() /* input */, v86.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #85" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op86,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v86.data() /* input */, v87.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #86" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op87,
    49 /* batch size */,
    v87.data() /* input */, v88.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #87" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op88,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v88.data() /* input */, v89.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #88" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op89,
    49 /* batch size */,
    v89.data() /* input */, v90.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #89" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op90,
    1 /* batch size */, 49 /* width */,
    v90.data() /* input */, v91.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #90" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op91,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v91.data() /* input */, v92.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #91" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op92,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v92.data() /* input */, v93.data() /* output */,
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
      v90.data() /* a */, v93.data() /* b */, v94.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #93" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op94,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v94.data() /* input */, v95.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #94" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 160 };
    const size_t b_shape[] = { 1, 7, 7, 160 };
    status = xnn_setup_add_nd_f32(
      op95,
      4, a_shape, 4, b_shape,
      v95.data() /* a */, v86.data() /* b */, v96.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #95" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op96,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v96.data() /* input */, v97.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #96" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op97,
    49 /* batch size */,
    v97.data() /* input */, v98.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #97" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op98,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v98.data() /* input */, v99.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #98" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op99,
    49 /* batch size */,
    v99.data() /* input */, v100.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #99" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op100,
    1 /* batch size */, 49 /* width */,
    v100.data() /* input */, v101.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #100" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op101,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v101.data() /* input */, v102.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #101" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op102,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v102.data() /* input */, v103.data() /* output */,
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
      v100.data() /* a */, v103.data() /* b */, v104.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #103" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op104,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v104.data() /* input */, v105.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #104" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 160 };
    const size_t b_shape[] = { 1, 7, 7, 160 };
    status = xnn_setup_add_nd_f32(
      op105,
      4, a_shape, 4, b_shape,
      v105.data() /* a */, v96.data() /* b */, v106.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #105" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op106,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v106.data() /* input */, v107.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #106" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op107,
    49 /* batch size */,
    v107.data() /* input */, v108.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #107" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op108,
    1 /* batch size */, 49 /* width */,
    v108.data() /* input */, v109.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #108" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op109,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v109.data() /* input */, v110.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #109" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op110,
    1 /* batch size */,
    v110.data() /* input */, v111.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #110" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op111,
    1 /* batch size */, 1 /* width */,
    v111.data() /* input */, v112.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #111" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op112,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v112.data() /* input */, v113.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #112" << std::endl;
    return ExecutionPlan();
  }

  XNN_PRAGMA_CLANG("clang diagnostic push")
  XNN_PRAGMA_CLANG("clang diagnostic ignored \"-Wpessimizing-move\"")
  return operators;
  XNN_PRAGMA_CLANG("clang diagnostic pop")
}

}  // namespace models
