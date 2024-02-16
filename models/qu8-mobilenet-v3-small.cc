// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!

#include <xnnpack.h>

#include <array>
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include <xnnpack/cache.h>
#include <xnnpack/common.h>
#include <xnnpack/models.h>

namespace models {

ExecutionPlan QU8MobileNetV3Small(pthreadpool_t threadpool) {
  alignas(16) static std::array<uint8_t, 150528 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v0;
  alignas(16) static std::array<uint8_t, 200704 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v1;
  alignas(16) static std::array<uint8_t, 200704 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v2;
  alignas(16) static std::array<uint8_t, 50176 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v3;
  alignas(16) static std::array<uint8_t, 16 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v4;
  alignas(16) static std::array<uint8_t, 8 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v5;
  alignas(16) static std::array<uint8_t, 16 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v6;
  alignas(16) static std::array<uint8_t, 16 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v7;
  alignas(16) static std::array<uint8_t, 16 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v8;
  alignas(16) static std::array<uint8_t, 50176 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v9;
  alignas(16) static std::array<uint8_t, 50176 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v10;
  alignas(16) static std::array<uint8_t, 225792 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v11;
  alignas(16) static std::array<uint8_t, 56448 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v12;
  alignas(16) static std::array<uint8_t, 18816 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v13;
  alignas(16) static std::array<uint8_t, 68992 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v14;
  alignas(16) static std::array<uint8_t, 68992 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v15;
  alignas(16) static std::array<uint8_t, 18816 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v16;
  alignas(16) static std::array<uint8_t, 18816 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v17;
  alignas(16) static std::array<uint8_t, 75264 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v18;
  alignas(16) static std::array<uint8_t, 75264 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v19;
  alignas(16) static std::array<uint8_t, 18816 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v20;
  alignas(16) static std::array<uint8_t, 18816 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v21;
  alignas(16) static std::array<uint8_t, 96 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v22;
  alignas(16) static std::array<uint8_t, 24 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v23;
  alignas(16) static std::array<uint8_t, 96 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v24;
  alignas(16) static std::array<uint8_t, 96 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v25;
  alignas(16) static std::array<uint8_t, 96 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v26;
  alignas(16) static std::array<uint8_t, 18816 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v27;
  alignas(16) static std::array<uint8_t, 7840 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v28;
  alignas(16) static std::array<uint8_t, 47040 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v29;
  alignas(16) static std::array<uint8_t, 47040 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v30;
  alignas(16) static std::array<uint8_t, 47040 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v31;
  alignas(16) static std::array<uint8_t, 47040 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v32;
  alignas(16) static std::array<uint8_t, 240 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v33;
  alignas(16) static std::array<uint8_t, 64 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v34;
  alignas(16) static std::array<uint8_t, 240 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v35;
  alignas(16) static std::array<uint8_t, 240 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v36;
  alignas(16) static std::array<uint8_t, 240 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v37;
  alignas(16) static std::array<uint8_t, 47040 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v38;
  alignas(16) static std::array<uint8_t, 7840 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v39;
  alignas(16) static std::array<uint8_t, 7840 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v40;
  alignas(16) static std::array<uint8_t, 47040 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v41;
  alignas(16) static std::array<uint8_t, 47040 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v42;
  alignas(16) static std::array<uint8_t, 47040 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v43;
  alignas(16) static std::array<uint8_t, 47040 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v44;
  alignas(16) static std::array<uint8_t, 240 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v45;
  alignas(16) static std::array<uint8_t, 64 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v46;
  alignas(16) static std::array<uint8_t, 240 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v47;
  alignas(16) static std::array<uint8_t, 240 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v48;
  alignas(16) static std::array<uint8_t, 240 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v49;
  alignas(16) static std::array<uint8_t, 47040 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v50;
  alignas(16) static std::array<uint8_t, 7840 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v51;
  alignas(16) static std::array<uint8_t, 7840 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v52;
  alignas(16) static std::array<uint8_t, 23520 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v53;
  alignas(16) static std::array<uint8_t, 23520 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v54;
  alignas(16) static std::array<uint8_t, 23520 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v55;
  alignas(16) static std::array<uint8_t, 23520 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v56;
  alignas(16) static std::array<uint8_t, 120 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v57;
  alignas(16) static std::array<uint8_t, 32 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v58;
  alignas(16) static std::array<uint8_t, 120 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v59;
  alignas(16) static std::array<uint8_t, 120 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v60;
  alignas(16) static std::array<uint8_t, 120 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v61;
  alignas(16) static std::array<uint8_t, 23520 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v62;
  alignas(16) static std::array<uint8_t, 9408 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v63;
  alignas(16) static std::array<uint8_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v64;
  alignas(16) static std::array<uint8_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v65;
  alignas(16) static std::array<uint8_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v66;
  alignas(16) static std::array<uint8_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v67;
  alignas(16) static std::array<uint8_t, 144 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v68;
  alignas(16) static std::array<uint8_t, 40 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v69;
  alignas(16) static std::array<uint8_t, 144 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v70;
  alignas(16) static std::array<uint8_t, 144 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v71;
  alignas(16) static std::array<uint8_t, 144 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v72;
  alignas(16) static std::array<uint8_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v73;
  alignas(16) static std::array<uint8_t, 9408 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v74;
  alignas(16) static std::array<uint8_t, 9408 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v75;
  alignas(16) static std::array<uint8_t, 56448 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v76;
  alignas(16) static std::array<uint8_t, 56448 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v77;
  alignas(16) static std::array<uint8_t, 14112 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v78;
  alignas(16) static std::array<uint8_t, 14112 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v79;
  alignas(16) static std::array<uint8_t, 288 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v80;
  alignas(16) static std::array<uint8_t, 72 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v81;
  alignas(16) static std::array<uint8_t, 288 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v82;
  alignas(16) static std::array<uint8_t, 288 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v83;
  alignas(16) static std::array<uint8_t, 288 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v84;
  alignas(16) static std::array<uint8_t, 14112 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v85;
  alignas(16) static std::array<uint8_t, 4704 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v86;
  alignas(16) static std::array<uint8_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v87;
  alignas(16) static std::array<uint8_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v88;
  alignas(16) static std::array<uint8_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v89;
  alignas(16) static std::array<uint8_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v90;
  alignas(16) static std::array<uint8_t, 576 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v91;
  alignas(16) static std::array<uint8_t, 144 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v92;
  alignas(16) static std::array<uint8_t, 576 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v93;
  alignas(16) static std::array<uint8_t, 576 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v94;
  alignas(16) static std::array<uint8_t, 576 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v95;
  alignas(16) static std::array<uint8_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v96;
  alignas(16) static std::array<uint8_t, 4704 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v97;
  alignas(16) static std::array<uint8_t, 4704 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v98;
  alignas(16) static std::array<uint8_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v99;
  alignas(16) static std::array<uint8_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v100;
  alignas(16) static std::array<uint8_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v101;
  alignas(16) static std::array<uint8_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v102;
  alignas(16) static std::array<uint8_t, 576 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v103;
  alignas(16) static std::array<uint8_t, 144 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v104;
  alignas(16) static std::array<uint8_t, 576 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v105;
  alignas(16) static std::array<uint8_t, 576 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v106;
  alignas(16) static std::array<uint8_t, 576 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v107;
  alignas(16) static std::array<uint8_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v108;
  alignas(16) static std::array<uint8_t, 4704 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v109;
  alignas(16) static std::array<uint8_t, 4704 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v110;
  alignas(16) static std::array<uint8_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v111;
  alignas(16) static std::array<uint8_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v112;
  alignas(16) static std::array<uint8_t, 576 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v113;
  alignas(16) static std::array<uint8_t, 1024 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v114;
  alignas(16) static std::array<uint8_t, 1024 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v115;
  alignas(16) static std::array<uint8_t, 1024 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v116;
  alignas(16) static std::array<uint8_t, 1001 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v117;
  alignas(16) static std::array<uint8_t, 1001 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v118;
  alignas(16) static std::array<uint8_t, 1001 + XNN_EXTRA_BYTES / sizeof(uint8_t)> v119;
  alignas(16) static std::array<uint8_t, 432 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w120;
  alignas(16) static std::array<int32_t, 16 + XNN_EXTRA_BYTES / sizeof(int32_t)> w121;
  alignas(16) static std::array<uint8_t, 144 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w122;
  alignas(16) static std::array<int32_t, 16 + XNN_EXTRA_BYTES / sizeof(int32_t)> w123;
  alignas(16) static std::array<uint8_t, 128 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w124;
  alignas(16) static std::array<int32_t, 8 + XNN_EXTRA_BYTES / sizeof(int32_t)> w125;
  alignas(16) static std::array<uint8_t, 128 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w126;
  alignas(16) static std::array<int32_t, 16 + XNN_EXTRA_BYTES / sizeof(int32_t)> w127;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w128;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w129;
  alignas(16) static std::array<uint8_t, 256 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w130;
  alignas(16) static std::array<int32_t, 16 + XNN_EXTRA_BYTES / sizeof(int32_t)> w131;
  alignas(16) static std::array<uint8_t, 1152 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w132;
  alignas(16) static std::array<int32_t, 72 + XNN_EXTRA_BYTES / sizeof(int32_t)> w133;
  alignas(16) static std::array<uint8_t, 648 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w134;
  alignas(16) static std::array<int32_t, 72 + XNN_EXTRA_BYTES / sizeof(int32_t)> w135;
  alignas(16) static std::array<uint8_t, 1728 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w136;
  alignas(16) static std::array<int32_t, 24 + XNN_EXTRA_BYTES / sizeof(int32_t)> w137;
  alignas(16) static std::array<uint8_t, 2112 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w138;
  alignas(16) static std::array<int32_t, 88 + XNN_EXTRA_BYTES / sizeof(int32_t)> w139;
  alignas(16) static std::array<uint8_t, 792 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w140;
  alignas(16) static std::array<int32_t, 88 + XNN_EXTRA_BYTES / sizeof(int32_t)> w141;
  alignas(16) static std::array<uint8_t, 2112 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w142;
  alignas(16) static std::array<int32_t, 24 + XNN_EXTRA_BYTES / sizeof(int32_t)> w143;
  alignas(16) static std::array<uint8_t, 2304 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w144;
  alignas(16) static std::array<int32_t, 96 + XNN_EXTRA_BYTES / sizeof(int32_t)> w145;
  alignas(16) static std::array<uint8_t, 2400 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w146;
  alignas(16) static std::array<int32_t, 96 + XNN_EXTRA_BYTES / sizeof(int32_t)> w147;
  alignas(16) static std::array<uint8_t, 2304 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w148;
  alignas(16) static std::array<int32_t, 24 + XNN_EXTRA_BYTES / sizeof(int32_t)> w149;
  alignas(16) static std::array<uint8_t, 2304 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w150;
  alignas(16) static std::array<int32_t, 96 + XNN_EXTRA_BYTES / sizeof(int32_t)> w151;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w152;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w153;
  alignas(16) static std::array<uint8_t, 3840 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w154;
  alignas(16) static std::array<int32_t, 40 + XNN_EXTRA_BYTES / sizeof(int32_t)> w155;
  alignas(16) static std::array<uint8_t, 9600 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w156;
  alignas(16) static std::array<int32_t, 240 + XNN_EXTRA_BYTES / sizeof(int32_t)> w157;
  alignas(16) static std::array<uint8_t, 6000 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w158;
  alignas(16) static std::array<int32_t, 240 + XNN_EXTRA_BYTES / sizeof(int32_t)> w159;
  alignas(16) static std::array<uint8_t, 15360 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w160;
  alignas(16) static std::array<int32_t, 64 + XNN_EXTRA_BYTES / sizeof(int32_t)> w161;
  alignas(16) static std::array<uint8_t, 15360 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w162;
  alignas(16) static std::array<int32_t, 240 + XNN_EXTRA_BYTES / sizeof(int32_t)> w163;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w164;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w165;
  alignas(16) static std::array<uint8_t, 9600 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w166;
  alignas(16) static std::array<int32_t, 40 + XNN_EXTRA_BYTES / sizeof(int32_t)> w167;
  alignas(16) static std::array<uint8_t, 9600 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w168;
  alignas(16) static std::array<int32_t, 240 + XNN_EXTRA_BYTES / sizeof(int32_t)> w169;
  alignas(16) static std::array<uint8_t, 6000 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w170;
  alignas(16) static std::array<int32_t, 240 + XNN_EXTRA_BYTES / sizeof(int32_t)> w171;
  alignas(16) static std::array<uint8_t, 15360 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w172;
  alignas(16) static std::array<int32_t, 64 + XNN_EXTRA_BYTES / sizeof(int32_t)> w173;
  alignas(16) static std::array<uint8_t, 15360 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w174;
  alignas(16) static std::array<int32_t, 240 + XNN_EXTRA_BYTES / sizeof(int32_t)> w175;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w176;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w177;
  alignas(16) static std::array<uint8_t, 9600 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w178;
  alignas(16) static std::array<int32_t, 40 + XNN_EXTRA_BYTES / sizeof(int32_t)> w179;
  alignas(16) static std::array<uint8_t, 4800 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w180;
  alignas(16) static std::array<int32_t, 120 + XNN_EXTRA_BYTES / sizeof(int32_t)> w181;
  alignas(16) static std::array<uint8_t, 3000 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w182;
  alignas(16) static std::array<int32_t, 120 + XNN_EXTRA_BYTES / sizeof(int32_t)> w183;
  alignas(16) static std::array<uint8_t, 3840 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w184;
  alignas(16) static std::array<int32_t, 32 + XNN_EXTRA_BYTES / sizeof(int32_t)> w185;
  alignas(16) static std::array<uint8_t, 3840 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w186;
  alignas(16) static std::array<int32_t, 120 + XNN_EXTRA_BYTES / sizeof(int32_t)> w187;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w188;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w189;
  alignas(16) static std::array<uint8_t, 5760 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w190;
  alignas(16) static std::array<int32_t, 48 + XNN_EXTRA_BYTES / sizeof(int32_t)> w191;
  alignas(16) static std::array<uint8_t, 6912 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w192;
  alignas(16) static std::array<int32_t, 144 + XNN_EXTRA_BYTES / sizeof(int32_t)> w193;
  alignas(16) static std::array<uint8_t, 3600 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w194;
  alignas(16) static std::array<int32_t, 144 + XNN_EXTRA_BYTES / sizeof(int32_t)> w195;
  alignas(16) static std::array<uint8_t, 5760 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w196;
  alignas(16) static std::array<int32_t, 40 + XNN_EXTRA_BYTES / sizeof(int32_t)> w197;
  alignas(16) static std::array<uint8_t, 5760 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w198;
  alignas(16) static std::array<int32_t, 144 + XNN_EXTRA_BYTES / sizeof(int32_t)> w199;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w200;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w201;
  alignas(16) static std::array<uint8_t, 6912 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w202;
  alignas(16) static std::array<int32_t, 48 + XNN_EXTRA_BYTES / sizeof(int32_t)> w203;
  alignas(16) static std::array<uint8_t, 13824 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w204;
  alignas(16) static std::array<int32_t, 288 + XNN_EXTRA_BYTES / sizeof(int32_t)> w205;
  alignas(16) static std::array<uint8_t, 7200 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w206;
  alignas(16) static std::array<int32_t, 288 + XNN_EXTRA_BYTES / sizeof(int32_t)> w207;
  alignas(16) static std::array<uint8_t, 20736 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w208;
  alignas(16) static std::array<int32_t, 72 + XNN_EXTRA_BYTES / sizeof(int32_t)> w209;
  alignas(16) static std::array<uint8_t, 20736 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w210;
  alignas(16) static std::array<int32_t, 288 + XNN_EXTRA_BYTES / sizeof(int32_t)> w211;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w212;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w213;
  alignas(16) static std::array<uint8_t, 27648 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w214;
  alignas(16) static std::array<int32_t, 96 + XNN_EXTRA_BYTES / sizeof(int32_t)> w215;
  alignas(16) static std::array<uint8_t, 55296 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w216;
  alignas(16) static std::array<int32_t, 576 + XNN_EXTRA_BYTES / sizeof(int32_t)> w217;
  alignas(16) static std::array<uint8_t, 14400 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w218;
  alignas(16) static std::array<int32_t, 576 + XNN_EXTRA_BYTES / sizeof(int32_t)> w219;
  alignas(16) static std::array<uint8_t, 82944 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w220;
  alignas(16) static std::array<int32_t, 144 + XNN_EXTRA_BYTES / sizeof(int32_t)> w221;
  alignas(16) static std::array<uint8_t, 82944 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w222;
  alignas(16) static std::array<int32_t, 576 + XNN_EXTRA_BYTES / sizeof(int32_t)> w223;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w224;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w225;
  alignas(16) static std::array<uint8_t, 55296 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w226;
  alignas(16) static std::array<int32_t, 96 + XNN_EXTRA_BYTES / sizeof(int32_t)> w227;
  alignas(16) static std::array<uint8_t, 55296 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w228;
  alignas(16) static std::array<int32_t, 576 + XNN_EXTRA_BYTES / sizeof(int32_t)> w229;
  alignas(16) static std::array<uint8_t, 14400 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w230;
  alignas(16) static std::array<int32_t, 576 + XNN_EXTRA_BYTES / sizeof(int32_t)> w231;
  alignas(16) static std::array<uint8_t, 82944 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w232;
  alignas(16) static std::array<int32_t, 144 + XNN_EXTRA_BYTES / sizeof(int32_t)> w233;
  alignas(16) static std::array<uint8_t, 82944 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w234;
  alignas(16) static std::array<int32_t, 576 + XNN_EXTRA_BYTES / sizeof(int32_t)> w235;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w236;
  alignas(16) static std::array<uint8_t, 1 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w237;
  alignas(16) static std::array<uint8_t, 55296 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w238;
  alignas(16) static std::array<int32_t, 96 + XNN_EXTRA_BYTES / sizeof(int32_t)> w239;
  alignas(16) static std::array<uint8_t, 55296 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w240;
  alignas(16) static std::array<int32_t, 576 + XNN_EXTRA_BYTES / sizeof(int32_t)> w241;
  alignas(16) static std::array<uint8_t, 589824 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w242;
  alignas(16) static std::array<int32_t, 1024 + XNN_EXTRA_BYTES / sizeof(int32_t)> w243;
  alignas(16) static std::array<uint8_t, 1025024 + XNN_EXTRA_BYTES / sizeof(uint8_t)> w244;
  alignas(16) static std::array<int32_t, 1001 + XNN_EXTRA_BYTES / sizeof(int32_t)> w245;

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto qu8rng = std::bind(std::uniform_int_distribution<int>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max()), std::ref(rng));
  auto qs32rng = std::bind(std::uniform_int_distribution<int>(-10000, 10000), std::ref(rng));
  std::generate(v0.begin(), v0.end(), std::ref(qu8rng));
  std::generate(v1.begin(), v1.end(), std::ref(qu8rng));
  std::generate(v2.begin(), v2.end(), std::ref(qu8rng));
  std::generate(v3.begin(), v3.end(), std::ref(qu8rng));
  std::generate(v4.begin(), v4.end(), std::ref(qu8rng));
  std::generate(v5.begin(), v5.end(), std::ref(qu8rng));
  std::generate(v6.begin(), v6.end(), std::ref(qu8rng));
  std::generate(v7.begin(), v7.end(), std::ref(qu8rng));
  std::generate(v8.begin(), v8.end(), std::ref(qu8rng));
  std::generate(v9.begin(), v9.end(), std::ref(qu8rng));
  std::generate(v10.begin(), v10.end(), std::ref(qu8rng));
  std::generate(v11.begin(), v11.end(), std::ref(qu8rng));
  std::generate(v12.begin(), v12.end(), std::ref(qu8rng));
  std::generate(v13.begin(), v13.end(), std::ref(qu8rng));
  std::generate(v14.begin(), v14.end(), std::ref(qu8rng));
  std::generate(v15.begin(), v15.end(), std::ref(qu8rng));
  std::generate(v16.begin(), v16.end(), std::ref(qu8rng));
  std::generate(v17.begin(), v17.end(), std::ref(qu8rng));
  std::generate(v18.begin(), v18.end(), std::ref(qu8rng));
  std::generate(v19.begin(), v19.end(), std::ref(qu8rng));
  std::generate(v20.begin(), v20.end(), std::ref(qu8rng));
  std::generate(v21.begin(), v21.end(), std::ref(qu8rng));
  std::generate(v22.begin(), v22.end(), std::ref(qu8rng));
  std::generate(v23.begin(), v23.end(), std::ref(qu8rng));
  std::generate(v24.begin(), v24.end(), std::ref(qu8rng));
  std::generate(v25.begin(), v25.end(), std::ref(qu8rng));
  std::generate(v26.begin(), v26.end(), std::ref(qu8rng));
  std::generate(v27.begin(), v27.end(), std::ref(qu8rng));
  std::generate(v28.begin(), v28.end(), std::ref(qu8rng));
  std::generate(v29.begin(), v29.end(), std::ref(qu8rng));
  std::generate(v30.begin(), v30.end(), std::ref(qu8rng));
  std::generate(v31.begin(), v31.end(), std::ref(qu8rng));
  std::generate(v32.begin(), v32.end(), std::ref(qu8rng));
  std::generate(v33.begin(), v33.end(), std::ref(qu8rng));
  std::generate(v34.begin(), v34.end(), std::ref(qu8rng));
  std::generate(v35.begin(), v35.end(), std::ref(qu8rng));
  std::generate(v36.begin(), v36.end(), std::ref(qu8rng));
  std::generate(v37.begin(), v37.end(), std::ref(qu8rng));
  std::generate(v38.begin(), v38.end(), std::ref(qu8rng));
  std::generate(v39.begin(), v39.end(), std::ref(qu8rng));
  std::generate(v40.begin(), v40.end(), std::ref(qu8rng));
  std::generate(v41.begin(), v41.end(), std::ref(qu8rng));
  std::generate(v42.begin(), v42.end(), std::ref(qu8rng));
  std::generate(v43.begin(), v43.end(), std::ref(qu8rng));
  std::generate(v44.begin(), v44.end(), std::ref(qu8rng));
  std::generate(v45.begin(), v45.end(), std::ref(qu8rng));
  std::generate(v46.begin(), v46.end(), std::ref(qu8rng));
  std::generate(v47.begin(), v47.end(), std::ref(qu8rng));
  std::generate(v48.begin(), v48.end(), std::ref(qu8rng));
  std::generate(v49.begin(), v49.end(), std::ref(qu8rng));
  std::generate(v50.begin(), v50.end(), std::ref(qu8rng));
  std::generate(v51.begin(), v51.end(), std::ref(qu8rng));
  std::generate(v52.begin(), v52.end(), std::ref(qu8rng));
  std::generate(v53.begin(), v53.end(), std::ref(qu8rng));
  std::generate(v54.begin(), v54.end(), std::ref(qu8rng));
  std::generate(v55.begin(), v55.end(), std::ref(qu8rng));
  std::generate(v56.begin(), v56.end(), std::ref(qu8rng));
  std::generate(v57.begin(), v57.end(), std::ref(qu8rng));
  std::generate(v58.begin(), v58.end(), std::ref(qu8rng));
  std::generate(v59.begin(), v59.end(), std::ref(qu8rng));
  std::generate(v60.begin(), v60.end(), std::ref(qu8rng));
  std::generate(v61.begin(), v61.end(), std::ref(qu8rng));
  std::generate(v62.begin(), v62.end(), std::ref(qu8rng));
  std::generate(v63.begin(), v63.end(), std::ref(qu8rng));
  std::generate(v64.begin(), v64.end(), std::ref(qu8rng));
  std::generate(v65.begin(), v65.end(), std::ref(qu8rng));
  std::generate(v66.begin(), v66.end(), std::ref(qu8rng));
  std::generate(v67.begin(), v67.end(), std::ref(qu8rng));
  std::generate(v68.begin(), v68.end(), std::ref(qu8rng));
  std::generate(v69.begin(), v69.end(), std::ref(qu8rng));
  std::generate(v70.begin(), v70.end(), std::ref(qu8rng));
  std::generate(v71.begin(), v71.end(), std::ref(qu8rng));
  std::generate(v72.begin(), v72.end(), std::ref(qu8rng));
  std::generate(v73.begin(), v73.end(), std::ref(qu8rng));
  std::generate(v74.begin(), v74.end(), std::ref(qu8rng));
  std::generate(v75.begin(), v75.end(), std::ref(qu8rng));
  std::generate(v76.begin(), v76.end(), std::ref(qu8rng));
  std::generate(v77.begin(), v77.end(), std::ref(qu8rng));
  std::generate(v78.begin(), v78.end(), std::ref(qu8rng));
  std::generate(v79.begin(), v79.end(), std::ref(qu8rng));
  std::generate(v80.begin(), v80.end(), std::ref(qu8rng));
  std::generate(v81.begin(), v81.end(), std::ref(qu8rng));
  std::generate(v82.begin(), v82.end(), std::ref(qu8rng));
  std::generate(v83.begin(), v83.end(), std::ref(qu8rng));
  std::generate(v84.begin(), v84.end(), std::ref(qu8rng));
  std::generate(v85.begin(), v85.end(), std::ref(qu8rng));
  std::generate(v86.begin(), v86.end(), std::ref(qu8rng));
  std::generate(v87.begin(), v87.end(), std::ref(qu8rng));
  std::generate(v88.begin(), v88.end(), std::ref(qu8rng));
  std::generate(v89.begin(), v89.end(), std::ref(qu8rng));
  std::generate(v90.begin(), v90.end(), std::ref(qu8rng));
  std::generate(v91.begin(), v91.end(), std::ref(qu8rng));
  std::generate(v92.begin(), v92.end(), std::ref(qu8rng));
  std::generate(v93.begin(), v93.end(), std::ref(qu8rng));
  std::generate(v94.begin(), v94.end(), std::ref(qu8rng));
  std::generate(v95.begin(), v95.end(), std::ref(qu8rng));
  std::generate(v96.begin(), v96.end(), std::ref(qu8rng));
  std::generate(v97.begin(), v97.end(), std::ref(qu8rng));
  std::generate(v98.begin(), v98.end(), std::ref(qu8rng));
  std::generate(v99.begin(), v99.end(), std::ref(qu8rng));
  std::generate(v100.begin(), v100.end(), std::ref(qu8rng));
  std::generate(v101.begin(), v101.end(), std::ref(qu8rng));
  std::generate(v102.begin(), v102.end(), std::ref(qu8rng));
  std::generate(v103.begin(), v103.end(), std::ref(qu8rng));
  std::generate(v104.begin(), v104.end(), std::ref(qu8rng));
  std::generate(v105.begin(), v105.end(), std::ref(qu8rng));
  std::generate(v106.begin(), v106.end(), std::ref(qu8rng));
  std::generate(v107.begin(), v107.end(), std::ref(qu8rng));
  std::generate(v108.begin(), v108.end(), std::ref(qu8rng));
  std::generate(v109.begin(), v109.end(), std::ref(qu8rng));
  std::generate(v110.begin(), v110.end(), std::ref(qu8rng));
  std::generate(v111.begin(), v111.end(), std::ref(qu8rng));
  std::generate(v112.begin(), v112.end(), std::ref(qu8rng));
  std::generate(v113.begin(), v113.end(), std::ref(qu8rng));
  std::generate(v114.begin(), v114.end(), std::ref(qu8rng));
  std::generate(v115.begin(), v115.end(), std::ref(qu8rng));
  std::generate(v116.begin(), v116.end(), std::ref(qu8rng));
  std::generate(v117.begin(), v117.end(), std::ref(qu8rng));
  std::generate(v118.begin(), v118.end(), std::ref(qu8rng));
  std::generate(v119.begin(), v119.end(), std::ref(qu8rng));
  std::generate(w120.begin(), w120.end(), std::ref(qu8rng));
  std::generate(w121.begin(), w121.end(), std::ref(qs32rng));
  std::generate(w122.begin(), w122.end(), std::ref(qu8rng));
  std::generate(w123.begin(), w123.end(), std::ref(qs32rng));
  std::generate(w124.begin(), w124.end(), std::ref(qu8rng));
  std::generate(w125.begin(), w125.end(), std::ref(qs32rng));
  std::generate(w126.begin(), w126.end(), std::ref(qu8rng));
  std::generate(w127.begin(), w127.end(), std::ref(qs32rng));
  std::generate(w128.begin(), w128.end(), std::ref(qu8rng));
  std::generate(w129.begin(), w129.end(), std::ref(qu8rng));
  std::generate(w130.begin(), w130.end(), std::ref(qu8rng));
  std::generate(w131.begin(), w131.end(), std::ref(qs32rng));
  std::generate(w132.begin(), w132.end(), std::ref(qu8rng));
  std::generate(w133.begin(), w133.end(), std::ref(qs32rng));
  std::generate(w134.begin(), w134.end(), std::ref(qu8rng));
  std::generate(w135.begin(), w135.end(), std::ref(qs32rng));
  std::generate(w136.begin(), w136.end(), std::ref(qu8rng));
  std::generate(w137.begin(), w137.end(), std::ref(qs32rng));
  std::generate(w138.begin(), w138.end(), std::ref(qu8rng));
  std::generate(w139.begin(), w139.end(), std::ref(qs32rng));
  std::generate(w140.begin(), w140.end(), std::ref(qu8rng));
  std::generate(w141.begin(), w141.end(), std::ref(qs32rng));
  std::generate(w142.begin(), w142.end(), std::ref(qu8rng));
  std::generate(w143.begin(), w143.end(), std::ref(qs32rng));
  std::generate(w144.begin(), w144.end(), std::ref(qu8rng));
  std::generate(w145.begin(), w145.end(), std::ref(qs32rng));
  std::generate(w146.begin(), w146.end(), std::ref(qu8rng));
  std::generate(w147.begin(), w147.end(), std::ref(qs32rng));
  std::generate(w148.begin(), w148.end(), std::ref(qu8rng));
  std::generate(w149.begin(), w149.end(), std::ref(qs32rng));
  std::generate(w150.begin(), w150.end(), std::ref(qu8rng));
  std::generate(w151.begin(), w151.end(), std::ref(qs32rng));
  std::generate(w152.begin(), w152.end(), std::ref(qu8rng));
  std::generate(w153.begin(), w153.end(), std::ref(qu8rng));
  std::generate(w154.begin(), w154.end(), std::ref(qu8rng));
  std::generate(w155.begin(), w155.end(), std::ref(qs32rng));
  std::generate(w156.begin(), w156.end(), std::ref(qu8rng));
  std::generate(w157.begin(), w157.end(), std::ref(qs32rng));
  std::generate(w158.begin(), w158.end(), std::ref(qu8rng));
  std::generate(w159.begin(), w159.end(), std::ref(qs32rng));
  std::generate(w160.begin(), w160.end(), std::ref(qu8rng));
  std::generate(w161.begin(), w161.end(), std::ref(qs32rng));
  std::generate(w162.begin(), w162.end(), std::ref(qu8rng));
  std::generate(w163.begin(), w163.end(), std::ref(qs32rng));
  std::generate(w164.begin(), w164.end(), std::ref(qu8rng));
  std::generate(w165.begin(), w165.end(), std::ref(qu8rng));
  std::generate(w166.begin(), w166.end(), std::ref(qu8rng));
  std::generate(w167.begin(), w167.end(), std::ref(qs32rng));
  std::generate(w168.begin(), w168.end(), std::ref(qu8rng));
  std::generate(w169.begin(), w169.end(), std::ref(qs32rng));
  std::generate(w170.begin(), w170.end(), std::ref(qu8rng));
  std::generate(w171.begin(), w171.end(), std::ref(qs32rng));
  std::generate(w172.begin(), w172.end(), std::ref(qu8rng));
  std::generate(w173.begin(), w173.end(), std::ref(qs32rng));
  std::generate(w174.begin(), w174.end(), std::ref(qu8rng));
  std::generate(w175.begin(), w175.end(), std::ref(qs32rng));
  std::generate(w176.begin(), w176.end(), std::ref(qu8rng));
  std::generate(w177.begin(), w177.end(), std::ref(qu8rng));
  std::generate(w178.begin(), w178.end(), std::ref(qu8rng));
  std::generate(w179.begin(), w179.end(), std::ref(qs32rng));
  std::generate(w180.begin(), w180.end(), std::ref(qu8rng));
  std::generate(w181.begin(), w181.end(), std::ref(qs32rng));
  std::generate(w182.begin(), w182.end(), std::ref(qu8rng));
  std::generate(w183.begin(), w183.end(), std::ref(qs32rng));
  std::generate(w184.begin(), w184.end(), std::ref(qu8rng));
  std::generate(w185.begin(), w185.end(), std::ref(qs32rng));
  std::generate(w186.begin(), w186.end(), std::ref(qu8rng));
  std::generate(w187.begin(), w187.end(), std::ref(qs32rng));
  std::generate(w188.begin(), w188.end(), std::ref(qu8rng));
  std::generate(w189.begin(), w189.end(), std::ref(qu8rng));
  std::generate(w190.begin(), w190.end(), std::ref(qu8rng));
  std::generate(w191.begin(), w191.end(), std::ref(qs32rng));
  std::generate(w192.begin(), w192.end(), std::ref(qu8rng));
  std::generate(w193.begin(), w193.end(), std::ref(qs32rng));
  std::generate(w194.begin(), w194.end(), std::ref(qu8rng));
  std::generate(w195.begin(), w195.end(), std::ref(qs32rng));
  std::generate(w196.begin(), w196.end(), std::ref(qu8rng));
  std::generate(w197.begin(), w197.end(), std::ref(qs32rng));
  std::generate(w198.begin(), w198.end(), std::ref(qu8rng));
  std::generate(w199.begin(), w199.end(), std::ref(qs32rng));
  std::generate(w200.begin(), w200.end(), std::ref(qu8rng));
  std::generate(w201.begin(), w201.end(), std::ref(qu8rng));
  std::generate(w202.begin(), w202.end(), std::ref(qu8rng));
  std::generate(w203.begin(), w203.end(), std::ref(qs32rng));
  std::generate(w204.begin(), w204.end(), std::ref(qu8rng));
  std::generate(w205.begin(), w205.end(), std::ref(qs32rng));
  std::generate(w206.begin(), w206.end(), std::ref(qu8rng));
  std::generate(w207.begin(), w207.end(), std::ref(qs32rng));
  std::generate(w208.begin(), w208.end(), std::ref(qu8rng));
  std::generate(w209.begin(), w209.end(), std::ref(qs32rng));
  std::generate(w210.begin(), w210.end(), std::ref(qu8rng));
  std::generate(w211.begin(), w211.end(), std::ref(qs32rng));
  std::generate(w212.begin(), w212.end(), std::ref(qu8rng));
  std::generate(w213.begin(), w213.end(), std::ref(qu8rng));
  std::generate(w214.begin(), w214.end(), std::ref(qu8rng));
  std::generate(w215.begin(), w215.end(), std::ref(qs32rng));
  std::generate(w216.begin(), w216.end(), std::ref(qu8rng));
  std::generate(w217.begin(), w217.end(), std::ref(qs32rng));
  std::generate(w218.begin(), w218.end(), std::ref(qu8rng));
  std::generate(w219.begin(), w219.end(), std::ref(qs32rng));
  std::generate(w220.begin(), w220.end(), std::ref(qu8rng));
  std::generate(w221.begin(), w221.end(), std::ref(qs32rng));
  std::generate(w222.begin(), w222.end(), std::ref(qu8rng));
  std::generate(w223.begin(), w223.end(), std::ref(qs32rng));
  std::generate(w224.begin(), w224.end(), std::ref(qu8rng));
  std::generate(w225.begin(), w225.end(), std::ref(qu8rng));
  std::generate(w226.begin(), w226.end(), std::ref(qu8rng));
  std::generate(w227.begin(), w227.end(), std::ref(qs32rng));
  std::generate(w228.begin(), w228.end(), std::ref(qu8rng));
  std::generate(w229.begin(), w229.end(), std::ref(qs32rng));
  std::generate(w230.begin(), w230.end(), std::ref(qu8rng));
  std::generate(w231.begin(), w231.end(), std::ref(qs32rng));
  std::generate(w232.begin(), w232.end(), std::ref(qu8rng));
  std::generate(w233.begin(), w233.end(), std::ref(qs32rng));
  std::generate(w234.begin(), w234.end(), std::ref(qu8rng));
  std::generate(w235.begin(), w235.end(), std::ref(qs32rng));
  std::generate(w236.begin(), w236.end(), std::ref(qu8rng));
  std::generate(w237.begin(), w237.end(), std::ref(qu8rng));
  std::generate(w238.begin(), w238.end(), std::ref(qu8rng));
  std::generate(w239.begin(), w239.end(), std::ref(qs32rng));
  std::generate(w240.begin(), w240.end(), std::ref(qu8rng));
  std::generate(w241.begin(), w241.end(), std::ref(qs32rng));
  std::generate(w242.begin(), w242.end(), std::ref(qu8rng));
  std::generate(w243.begin(), w243.end(), std::ref(qs32rng));
  std::generate(w244.begin(), w244.end(), std::ref(qu8rng));
  std::generate(w245.begin(), w245.end(), std::ref(qs32rng));

  Operators operators;
  xnn_status status;
  xnn_code_cache* code_cache_ptr = nullptr;
  size_t max_workspace_size = 0;

  xnn_operator_t op0 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/3,
    /*group_output_channels=*/16,
    /*input_channel_stride=*/3,
    /*output_channel_stride=*/16,
    /*input_zero_point=*/(uint8_t) 128,
    /*input_scale=*/0.007874015718698502,
    /*kernel_zero_point=*/(uint8_t) 108,
    /*kernel_scale=*/0.03232726827263832,
    /*kernel=*/w120.data(), /*bias=*/w121.data(),
    /*output_zero_point=*/(uint8_t) 99,
    /*output_scale=*/0.2920726239681244,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #0" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op0, xnn_delete_operator);

  xnn_operator_t op1 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #1" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op1, xnn_delete_operator);

  xnn_operator_t op2 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/16,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/16,
    /*output_channel_stride=*/16,
    /*input_zero_point=*/(uint8_t) 2,
    /*input_scale=*/0.173323854804039,
    /*kernel_zero_point=*/(uint8_t) 127,
    /*kernel_scale=*/0.0927056297659874,
    /*kernel=*/w122.data(), /*bias=*/w123.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.9925559759140015,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #2" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op2, xnn_delete_operator);

  xnn_operator_t op3 = nullptr;
  status = xnn_create_global_average_pooling_nwc_qu8(
    0 /* input zero point */, 0.9925559759140015 /* input scale */,
    0 /* output zero point */, 0.9925559759140015 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #3" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op3, xnn_delete_operator);

  xnn_operator_t op4 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/16,
    /*group_output_channels=*/8,
    /*input_channel_stride=*/16,
    /*output_channel_stride=*/8,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.9925559759140015,
    /*kernel_zero_point=*/(uint8_t) 57,
    /*kernel_scale=*/0.00045214465353637934,
    /*kernel=*/w124.data(), /*bias=*/w125.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.03975478187203407,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op4);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #4" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op4, xnn_delete_operator);

  xnn_operator_t op5 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/8,
    /*group_output_channels=*/16,
    /*input_channel_stride=*/8,
    /*output_channel_stride=*/16,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.03975478187203407,
    /*kernel_zero_point=*/(uint8_t) 1,
    /*kernel_scale=*/0.0007166960276663303,
    /*kernel=*/w126.data(), /*bias=*/w127.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.014429666101932526,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #5" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op5, xnn_delete_operator);

  xnn_operator_t op6 = nullptr;
  status = xnn_create_add_nd_qu8(
    0 /* input1 zero point */, 0.014429666101932526 /* input1 scale */,
    0 /* input2 zero point */, 0.0117647061124444 /* input2 scale */,
    0 /* output zero point */, 0.023528477177023888 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op6);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #6" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op6, xnn_delete_operator);

  xnn_operator_t op7 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    0 /* input1 zero point */, 0.023528477177023888 /* input1 scale */,
    0 /* input2 zero point */, 0.0006536078290082514 /* input2 scale */,
    0 /* output zero point */, 0.003921509720385075 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op7);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #7" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op7, xnn_delete_operator);

  xnn_operator_t op8 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    0 /* input1 zero point */, 0.9925559759140015 /* input1 scale */,
    0 /* input2 zero point */, 0.003921509720385075 /* input2 scale */,
    0 /* output zero point */, 0.9246004223823547 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #8" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op8, xnn_delete_operator);

  xnn_operator_t op9 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/16,
    /*group_output_channels=*/16,
    /*input_channel_stride=*/16,
    /*output_channel_stride=*/16,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.9246004223823547,
    /*kernel_zero_point=*/(uint8_t) 146,
    /*kernel_scale=*/0.017008759081363678,
    /*kernel=*/w130.data(), /*bias=*/w131.data(),
    /*output_zero_point=*/(uint8_t) 130,
    /*output_scale=*/2.010422706604004,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #9" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op9, xnn_delete_operator);

  xnn_operator_t op10 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/16,
    /*group_output_channels=*/72,
    /*input_channel_stride=*/16,
    /*output_channel_stride=*/72,
    /*input_zero_point=*/(uint8_t) 130,
    /*input_scale=*/2.010422706604004,
    /*kernel_zero_point=*/(uint8_t) 123,
    /*kernel_scale=*/0.005887787323445082,
    /*kernel=*/w132.data(), /*bias=*/w133.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.9493569135665894,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #10" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op10, xnn_delete_operator);

  xnn_operator_t op11 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/72,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/72,
    /*output_channel_stride=*/72,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.9493569135665894,
    /*kernel_zero_point=*/(uint8_t) 113,
    /*kernel_scale=*/0.033502571284770966,
    /*kernel=*/w134.data(), /*bias=*/w135.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.6341432929039001,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #11" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op11, xnn_delete_operator);

  xnn_operator_t op12 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/72,
    /*group_output_channels=*/24,
    /*input_channel_stride=*/72,
    /*output_channel_stride=*/24,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.6341432929039001,
    /*kernel_zero_point=*/(uint8_t) 153,
    /*kernel_scale=*/0.017683790996670723,
    /*kernel=*/w136.data(), /*bias=*/w137.data(),
    /*output_zero_point=*/(uint8_t) 119,
    /*output_scale=*/1.0579205751419067,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #12" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op12, xnn_delete_operator);

  xnn_operator_t op13 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/24,
    /*group_output_channels=*/88,
    /*input_channel_stride=*/24,
    /*output_channel_stride=*/88,
    /*input_zero_point=*/(uint8_t) 119,
    /*input_scale=*/1.0579205751419067,
    /*kernel_zero_point=*/(uint8_t) 99,
    /*kernel_scale=*/0.005299868993461132,
    /*kernel=*/w138.data(), /*bias=*/w139.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.20985400676727295,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #13" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op13, xnn_delete_operator);

  xnn_operator_t op14 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/88,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/88,
    /*output_channel_stride=*/88,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.20985400676727295,
    /*kernel_zero_point=*/(uint8_t) 144,
    /*kernel_scale=*/0.05344513803720474,
    /*kernel=*/w140.data(), /*bias=*/w141.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.22873805463314056,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op14);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #14" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op14, xnn_delete_operator);

  xnn_operator_t op15 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/88,
    /*group_output_channels=*/24,
    /*input_channel_stride=*/88,
    /*output_channel_stride=*/24,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.22873805463314056,
    /*kernel_zero_point=*/(uint8_t) 139,
    /*kernel_scale=*/0.015702862292528152,
    /*kernel=*/w142.data(), /*bias=*/w143.data(),
    /*output_zero_point=*/(uint8_t) 124,
    /*output_scale=*/0.8896244764328003,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #15" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op15, xnn_delete_operator);

  xnn_operator_t op16 = nullptr;
  status = xnn_create_add_nd_qu8(
    124 /* input1 zero point */, 0.8896244764328003 /* input1 scale */,
    119 /* input2 zero point */, 1.0579205751419067 /* input2 scale */,
    123 /* output zero point */, 1.0426580905914307 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #16" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op16, xnn_delete_operator);

  xnn_operator_t op17 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/24,
    /*group_output_channels=*/96,
    /*input_channel_stride=*/24,
    /*output_channel_stride=*/96,
    /*input_zero_point=*/(uint8_t) 123,
    /*input_scale=*/1.0426580905914307,
    /*kernel_zero_point=*/(uint8_t) 154,
    /*kernel_scale=*/0.002672378672286868,
    /*kernel=*/w144.data(), /*bias=*/w145.data(),
    /*output_zero_point=*/(uint8_t) 110,
    /*output_scale=*/0.3380434811115265,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #17" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op17, xnn_delete_operator);

  xnn_operator_t op18 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op18);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #18" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op18, xnn_delete_operator);

  xnn_operator_t op19 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/1, /*input_padding_right=*/2,
    /*input_padding_bottom=*/2, /*input_padding_left=*/1,
    /*kernel_height=*/5, /*kernel_width=*/5,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/96,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/96,
    /*output_channel_stride=*/96,
    /*input_zero_point=*/(uint8_t) 2,
    /*input_scale=*/0.18497292697429657,
    /*kernel_zero_point=*/(uint8_t) 142,
    /*kernel_scale=*/0.031311504542827606,
    /*kernel=*/w146.data(), /*bias=*/w147.data(),
    /*output_zero_point=*/(uint8_t) 134,
    /*output_scale=*/0.24109338223934174,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #19" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op19, xnn_delete_operator);

  xnn_operator_t op20 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op20);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #20" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op20, xnn_delete_operator);

  xnn_operator_t op21 = nullptr;
  status = xnn_create_global_average_pooling_nwc_qu8(
    3 /* input zero point */, 0.1065792664885521 /* input scale */,
    3 /* output zero point */, 0.1065792664885521 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op21);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #21" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op21, xnn_delete_operator);

  xnn_operator_t op22 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/96,
    /*group_output_channels=*/24,
    /*input_channel_stride=*/96,
    /*output_channel_stride=*/24,
    /*input_zero_point=*/(uint8_t) 3,
    /*input_scale=*/0.1065792664885521,
    /*kernel_zero_point=*/(uint8_t) 98,
    /*kernel_scale=*/0.005171800963580608,
    /*kernel=*/w148.data(), /*bias=*/w149.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.051762163639068604,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op22);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #22" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op22, xnn_delete_operator);

  xnn_operator_t op23 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/24,
    /*group_output_channels=*/96,
    /*input_channel_stride=*/24,
    /*output_channel_stride=*/96,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.051762163639068604,
    /*kernel_zero_point=*/(uint8_t) 106,
    /*kernel_scale=*/0.005030923057347536,
    /*kernel=*/w150.data(), /*bias=*/w151.data(),
    /*output_zero_point=*/(uint8_t) 98,
    /*output_scale=*/0.03421778604388237,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op23);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #23" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op23, xnn_delete_operator);

  xnn_operator_t op24 = nullptr;
  status = xnn_create_add_nd_qu8(
    98 /* input1 zero point */, 0.03421778604388237 /* input1 scale */,
    0 /* input2 zero point */, 0.0117647061124444 /* input2 scale */,
    0 /* output zero point */, 0.023528477177023888 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #24" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op24, xnn_delete_operator);

  xnn_operator_t op25 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    0 /* input1 zero point */, 0.023528477177023888 /* input1 scale */,
    0 /* input2 zero point */, 0.0006536078290082514 /* input2 scale */,
    0 /* output zero point */, 0.003921509254723787 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #25" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op25, xnn_delete_operator);

  xnn_operator_t op26 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    3 /* input1 zero point */, 0.1065792664885521 /* input1 scale */,
    0 /* input2 zero point */, 0.003921509254723787 /* input2 scale */,
    4 /* output zero point */, 0.07695811986923218 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #26" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op26, xnn_delete_operator);

  xnn_operator_t op27 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/96,
    /*group_output_channels=*/40,
    /*input_channel_stride=*/96,
    /*output_channel_stride=*/40,
    /*input_zero_point=*/(uint8_t) 4,
    /*input_scale=*/0.07695811986923218,
    /*kernel_zero_point=*/(uint8_t) 128,
    /*kernel_scale=*/0.03726894408464432,
    /*kernel=*/w154.data(), /*bias=*/w155.data(),
    /*output_zero_point=*/(uint8_t) 127,
    /*output_scale=*/0.3759814500808716,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #27" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op27, xnn_delete_operator);

  xnn_operator_t op28 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/40,
    /*group_output_channels=*/240,
    /*input_channel_stride=*/40,
    /*output_channel_stride=*/240,
    /*input_zero_point=*/(uint8_t) 127,
    /*input_scale=*/0.3759814500808716,
    /*kernel_zero_point=*/(uint8_t) 159,
    /*kernel_scale=*/0.003184415865689516,
    /*kernel=*/w156.data(), /*bias=*/w157.data(),
    /*output_zero_point=*/(uint8_t) 128,
    /*output_scale=*/0.17979219555854797,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #28" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op28, xnn_delete_operator);

  xnn_operator_t op29 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op29);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #29" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op29, xnn_delete_operator);

  xnn_operator_t op30 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/2, /*input_padding_right=*/2,
    /*input_padding_bottom=*/2, /*input_padding_left=*/2,
    /*kernel_height=*/5, /*kernel_width=*/5,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/240,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/240,
    /*output_channel_stride=*/240,
    /*input_zero_point=*/(uint8_t) 4,
    /*input_scale=*/0.08579955250024796,
    /*kernel_zero_point=*/(uint8_t) 143,
    /*kernel_scale=*/0.1883949190378189,
    /*kernel=*/w158.data(), /*bias=*/w159.data(),
    /*output_zero_point=*/(uint8_t) 130,
    /*output_scale=*/0.49307096004486084,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op30);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #30" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op30, xnn_delete_operator);

  xnn_operator_t op31 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op31);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #31" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op31, xnn_delete_operator);

  xnn_operator_t op32 = nullptr;
  status = xnn_create_global_average_pooling_nwc_qu8(
    2 /* input zero point */, 0.220099538564682 /* input scale */,
    2 /* output zero point */, 0.220099538564682 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #32" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op32, xnn_delete_operator);

  xnn_operator_t op33 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/240,
    /*group_output_channels=*/64,
    /*input_channel_stride=*/240,
    /*output_channel_stride=*/64,
    /*input_zero_point=*/(uint8_t) 2,
    /*input_scale=*/0.220099538564682,
    /*kernel_zero_point=*/(uint8_t) 149,
    /*kernel_scale=*/0.009354852139949799,
    /*kernel=*/w160.data(), /*bias=*/w161.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.016910869628190994,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op33);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #33" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op33, xnn_delete_operator);

  xnn_operator_t op34 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/64,
    /*group_output_channels=*/240,
    /*input_channel_stride=*/64,
    /*output_channel_stride=*/240,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.016910869628190994,
    /*kernel_zero_point=*/(uint8_t) 108,
    /*kernel_scale=*/0.006087664980441332,
    /*kernel=*/w162.data(), /*bias=*/w163.data(),
    /*output_zero_point=*/(uint8_t) 144,
    /*output_scale=*/0.03480793163180351,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op34);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #34" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op34, xnn_delete_operator);

  xnn_operator_t op35 = nullptr;
  status = xnn_create_add_nd_qu8(
    144 /* input1 zero point */, 0.03480793163180351 /* input1 scale */,
    0 /* input2 zero point */, 0.0117647061124444 /* input2 scale */,
    0 /* output zero point */, 0.023528477177023888 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op35);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #35" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op35, xnn_delete_operator);

  xnn_operator_t op36 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    0 /* input1 zero point */, 0.023528477177023888 /* input1 scale */,
    0 /* input2 zero point */, 0.0006536078290082514 /* input2 scale */,
    0 /* output zero point */, 0.003921509254723787 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op36);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #36" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op36, xnn_delete_operator);

  xnn_operator_t op37 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    2 /* input1 zero point */, 0.220099538564682 /* input1 scale */,
    0 /* input2 zero point */, 0.003921509254723787 /* input2 scale */,
    20 /* output zero point */, 0.02150452695786953 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op37);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #37" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op37, xnn_delete_operator);

  xnn_operator_t op38 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/240,
    /*group_output_channels=*/40,
    /*input_channel_stride=*/240,
    /*output_channel_stride=*/40,
    /*input_zero_point=*/(uint8_t) 20,
    /*input_scale=*/0.02150452695786953,
    /*kernel_zero_point=*/(uint8_t) 115,
    /*kernel_scale=*/0.1018327996134758,
    /*kernel=*/w166.data(), /*bias=*/w167.data(),
    /*output_zero_point=*/(uint8_t) 137,
    /*output_scale=*/0.4652852416038513,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op38);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #38" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op38, xnn_delete_operator);

  xnn_operator_t op39 = nullptr;
  status = xnn_create_add_nd_qu8(
    137 /* input1 zero point */, 0.4652852416038513 /* input1 scale */,
    127 /* input2 zero point */, 0.3759814500808716 /* input2 scale */,
    132 /* output zero point */, 0.44771137833595276 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op39);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #39" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op39, xnn_delete_operator);

  xnn_operator_t op40 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/40,
    /*group_output_channels=*/240,
    /*input_channel_stride=*/40,
    /*output_channel_stride=*/240,
    /*input_zero_point=*/(uint8_t) 132,
    /*input_scale=*/0.44771137833595276,
    /*kernel_zero_point=*/(uint8_t) 129,
    /*kernel_scale=*/0.0009919562144204974,
    /*kernel=*/w168.data(), /*bias=*/w169.data(),
    /*output_zero_point=*/(uint8_t) 118,
    /*output_scale=*/0.12498034536838531,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op40);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #40" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op40, xnn_delete_operator);

  xnn_operator_t op41 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op41);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #41" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op41, xnn_delete_operator);

  xnn_operator_t op42 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/2, /*input_padding_right=*/2,
    /*input_padding_bottom=*/2, /*input_padding_left=*/2,
    /*kernel_height=*/5, /*kernel_width=*/5,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/240,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/240,
    /*output_channel_stride=*/240,
    /*input_zero_point=*/(uint8_t) 6,
    /*input_scale=*/0.06491293758153915,
    /*kernel_zero_point=*/(uint8_t) 101,
    /*kernel_scale=*/0.13295969367027283,
    /*kernel=*/w170.data(), /*bias=*/w171.data(),
    /*output_zero_point=*/(uint8_t) 150,
    /*output_scale=*/0.29956355690956116,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op42);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #42" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op42, xnn_delete_operator);

  xnn_operator_t op43 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op43);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #43" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op43, xnn_delete_operator);

  xnn_operator_t op44 = nullptr;
  status = xnn_create_global_average_pooling_nwc_qu8(
    3 /* input zero point */, 0.11336661875247955 /* input scale */,
    3 /* output zero point */, 0.11336661875247955 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op44);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #44" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op44, xnn_delete_operator);

  xnn_operator_t op45 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/240,
    /*group_output_channels=*/64,
    /*input_channel_stride=*/240,
    /*output_channel_stride=*/64,
    /*input_zero_point=*/(uint8_t) 3,
    /*input_scale=*/0.11336661875247955,
    /*kernel_zero_point=*/(uint8_t) 163,
    /*kernel_scale=*/0.007440278306603432,
    /*kernel=*/w172.data(), /*bias=*/w173.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.008480816148221493,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op45);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #45" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op45, xnn_delete_operator);

  xnn_operator_t op46 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/64,
    /*group_output_channels=*/240,
    /*input_channel_stride=*/64,
    /*output_channel_stride=*/240,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.008480816148221493,
    /*kernel_zero_point=*/(uint8_t) 110,
    /*kernel_scale=*/0.006039419211447239,
    /*kernel=*/w174.data(), /*bias=*/w175.data(),
    /*output_zero_point=*/(uint8_t) 135,
    /*output_scale=*/0.027621593326330185,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op46);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #46" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op46, xnn_delete_operator);

  xnn_operator_t op47 = nullptr;
  status = xnn_create_add_nd_qu8(
    135 /* input1 zero point */, 0.027621593326330185 /* input1 scale */,
    0 /* input2 zero point */, 0.0117647061124444 /* input2 scale */,
    0 /* output zero point */, 0.023528477177023888 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op47);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #47" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op47, xnn_delete_operator);

  xnn_operator_t op48 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    0 /* input1 zero point */, 0.023528477177023888 /* input1 scale */,
    0 /* input2 zero point */, 0.0006536078290082514 /* input2 scale */,
    0 /* output zero point */, 0.003921374212950468 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op48);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #48" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op48, xnn_delete_operator);

  xnn_operator_t op49 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    3 /* input1 zero point */, 0.11336661875247955 /* input1 scale */,
    0 /* input2 zero point */, 0.003921374212950468 /* input2 scale */,
    21 /* output zero point */, 0.0160247553139925 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op49);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #49" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op49, xnn_delete_operator);

  xnn_operator_t op50 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/240,
    /*group_output_channels=*/40,
    /*input_channel_stride=*/240,
    /*output_channel_stride=*/40,
    /*input_zero_point=*/(uint8_t) 21,
    /*input_scale=*/0.0160247553139925,
    /*kernel_zero_point=*/(uint8_t) 131,
    /*kernel_scale=*/0.22305507957935333,
    /*kernel=*/w178.data(), /*bias=*/w179.data(),
    /*output_zero_point=*/(uint8_t) 139,
    /*output_scale=*/0.544162929058075,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op50);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #50" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op50, xnn_delete_operator);

  xnn_operator_t op51 = nullptr;
  status = xnn_create_add_nd_qu8(
    139 /* input1 zero point */, 0.544162929058075 /* input1 scale */,
    132 /* input2 zero point */, 0.44771137833595276 /* input2 scale */,
    137 /* output zero point */, 0.6061347723007202 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op51);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #51" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op51, xnn_delete_operator);

  xnn_operator_t op52 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/40,
    /*group_output_channels=*/120,
    /*input_channel_stride=*/40,
    /*output_channel_stride=*/120,
    /*input_zero_point=*/(uint8_t) 137,
    /*input_scale=*/0.6061347723007202,
    /*kernel_zero_point=*/(uint8_t) 90,
    /*kernel_scale=*/0.0014072866179049015,
    /*kernel=*/w180.data(), /*bias=*/w181.data(),
    /*output_zero_point=*/(uint8_t) 117,
    /*output_scale=*/0.13909709453582764,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op52);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #52" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op52, xnn_delete_operator);

  xnn_operator_t op53 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op53);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #53" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op53, xnn_delete_operator);

  xnn_operator_t op54 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/2, /*input_padding_right=*/2,
    /*input_padding_bottom=*/2, /*input_padding_left=*/2,
    /*kernel_height=*/5, /*kernel_width=*/5,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/120,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/120,
    /*output_channel_stride=*/120,
    /*input_zero_point=*/(uint8_t) 5,
    /*input_scale=*/0.0727764368057251,
    /*kernel_zero_point=*/(uint8_t) 121,
    /*kernel_scale=*/0.09157519787549973,
    /*kernel=*/w182.data(), /*bias=*/w183.data(),
    /*output_zero_point=*/(uint8_t) 140,
    /*output_scale=*/0.28514617681503296,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op54);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #54" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op54, xnn_delete_operator);

  xnn_operator_t op55 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op55);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #55" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op55, xnn_delete_operator);

  xnn_operator_t op56 = nullptr;
  status = xnn_create_global_average_pooling_nwc_qu8(
    3 /* input zero point */, 0.1223522424697876 /* input scale */,
    3 /* output zero point */, 0.1223522424697876 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op56);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #56" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op56, xnn_delete_operator);

  xnn_operator_t op57 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/120,
    /*group_output_channels=*/32,
    /*input_channel_stride=*/120,
    /*output_channel_stride=*/32,
    /*input_zero_point=*/(uint8_t) 3,
    /*input_scale=*/0.1223522424697876,
    /*kernel_zero_point=*/(uint8_t) 40,
    /*kernel_scale=*/0.0008257423178292811,
    /*kernel=*/w184.data(), /*bias=*/w185.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.025332391262054443,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op57);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #57" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op57, xnn_delete_operator);

  xnn_operator_t op58 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/32,
    /*group_output_channels=*/120,
    /*input_channel_stride=*/32,
    /*output_channel_stride=*/120,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.025332391262054443,
    /*kernel_zero_point=*/(uint8_t) 220,
    /*kernel_scale=*/0.0021832138299942017,
    /*kernel=*/w186.data(), /*bias=*/w187.data(),
    /*output_zero_point=*/(uint8_t) 139,
    /*output_scale=*/0.026293933391571045,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op58);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #58" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op58, xnn_delete_operator);

  xnn_operator_t op59 = nullptr;
  status = xnn_create_add_nd_qu8(
    139 /* input1 zero point */, 0.026293933391571045 /* input1 scale */,
    0 /* input2 zero point */, 0.0117647061124444 /* input2 scale */,
    0 /* output zero point */, 0.023528477177023888 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op59);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #59" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op59, xnn_delete_operator);

  xnn_operator_t op60 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    0 /* input1 zero point */, 0.023528477177023888 /* input1 scale */,
    0 /* input2 zero point */, 0.0006536078290082514 /* input2 scale */,
    0 /* output zero point */, 0.003921508323401213 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op60);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #60" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op60, xnn_delete_operator);

  xnn_operator_t op61 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    3 /* input1 zero point */, 0.1223522424697876 /* input1 scale */,
    0 /* input2 zero point */, 0.003921508323401213 /* input2 scale */,
    7 /* output zero point */, 0.04942065477371216 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op61);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #61" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op61, xnn_delete_operator);

  xnn_operator_t op62 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/120,
    /*group_output_channels=*/48,
    /*input_channel_stride=*/120,
    /*output_channel_stride=*/48,
    /*input_zero_point=*/(uint8_t) 7,
    /*input_scale=*/0.04942065477371216,
    /*kernel_zero_point=*/(uint8_t) 101,
    /*kernel_scale=*/0.03507576882839203,
    /*kernel=*/w190.data(), /*bias=*/w191.data(),
    /*output_zero_point=*/(uint8_t) 129,
    /*output_scale=*/0.39454951882362366,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op62);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #62" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op62, xnn_delete_operator);

  xnn_operator_t op63 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/48,
    /*group_output_channels=*/144,
    /*input_channel_stride=*/48,
    /*output_channel_stride=*/144,
    /*input_zero_point=*/(uint8_t) 129,
    /*input_scale=*/0.39454951882362366,
    /*kernel_zero_point=*/(uint8_t) 148,
    /*kernel_scale=*/0.0015211983118206263,
    /*kernel=*/w192.data(), /*bias=*/w193.data(),
    /*output_zero_point=*/(uint8_t) 114,
    /*output_scale=*/0.18048983812332153,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op63);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #63" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op63, xnn_delete_operator);

  xnn_operator_t op64 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op64);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #64" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op64, xnn_delete_operator);

  xnn_operator_t op65 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/2, /*input_padding_right=*/2,
    /*input_padding_bottom=*/2, /*input_padding_left=*/2,
    /*kernel_height=*/5, /*kernel_width=*/5,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/144,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/144,
    /*output_channel_stride=*/144,
    /*input_zero_point=*/(uint8_t) 4,
    /*input_scale=*/0.09509307891130447,
    /*kernel_zero_point=*/(uint8_t) 115,
    /*kernel_scale=*/0.0958247184753418,
    /*kernel=*/w194.data(), /*bias=*/w195.data(),
    /*output_zero_point=*/(uint8_t) 151,
    /*output_scale=*/0.3922523558139801,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op65);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #65" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op65, xnn_delete_operator);

  xnn_operator_t op66 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op66);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #66" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op66, xnn_delete_operator);

  xnn_operator_t op67 = nullptr;
  status = xnn_create_global_average_pooling_nwc_qu8(
    2 /* input zero point */, 0.14624309539794922 /* input scale */,
    2 /* output zero point */, 0.14624309539794922 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op67);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #67" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op67, xnn_delete_operator);

  xnn_operator_t op68 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/144,
    /*group_output_channels=*/40,
    /*input_channel_stride=*/144,
    /*output_channel_stride=*/40,
    /*input_zero_point=*/(uint8_t) 2,
    /*input_scale=*/0.14624309539794922,
    /*kernel_zero_point=*/(uint8_t) 130,
    /*kernel_scale=*/0.0060674939304590225,
    /*kernel=*/w196.data(), /*bias=*/w197.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.008962834253907204,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op68);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #68" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op68, xnn_delete_operator);

  xnn_operator_t op69 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/40,
    /*group_output_channels=*/144,
    /*input_channel_stride=*/40,
    /*output_channel_stride=*/144,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.008962834253907204,
    /*kernel_zero_point=*/(uint8_t) 124,
    /*kernel_scale=*/0.004431542940437794,
    /*kernel=*/w198.data(), /*bias=*/w199.data(),
    /*output_zero_point=*/(uint8_t) 134,
    /*output_scale=*/0.02729739062488079,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op69);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #69" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op69, xnn_delete_operator);

  xnn_operator_t op70 = nullptr;
  status = xnn_create_add_nd_qu8(
    134 /* input1 zero point */, 0.02729739062488079 /* input1 scale */,
    0 /* input2 zero point */, 0.0117647061124444 /* input2 scale */,
    0 /* output zero point */, 0.023528477177023888 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op70);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #70" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op70, xnn_delete_operator);

  xnn_operator_t op71 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    0 /* input1 zero point */, 0.023528477177023888 /* input1 scale */,
    0 /* input2 zero point */, 0.0006536078290082514 /* input2 scale */,
    0 /* output zero point */, 0.003921374212950468 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op71);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #71" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op71, xnn_delete_operator);

  xnn_operator_t op72 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    2 /* input1 zero point */, 0.14624309539794922 /* input1 scale */,
    0 /* input2 zero point */, 0.003921374212950468 /* input2 scale */,
    13 /* output zero point */, 0.023374175652861595 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op72);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #72" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op72, xnn_delete_operator);

  xnn_operator_t op73 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/144,
    /*group_output_channels=*/48,
    /*input_channel_stride=*/144,
    /*output_channel_stride=*/48,
    /*input_zero_point=*/(uint8_t) 13,
    /*input_scale=*/0.023374175652861595,
    /*kernel_zero_point=*/(uint8_t) 125,
    /*kernel_scale=*/0.13331712782382965,
    /*kernel=*/w202.data(), /*bias=*/w203.data(),
    /*output_zero_point=*/(uint8_t) 140,
    /*output_scale=*/0.42487239837646484,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op73);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #73" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op73, xnn_delete_operator);

  xnn_operator_t op74 = nullptr;
  status = xnn_create_add_nd_qu8(
    140 /* input1 zero point */, 0.42487239837646484 /* input1 scale */,
    129 /* input2 zero point */, 0.39454951882362366 /* input2 scale */,
    137 /* output zero point */, 0.48052287101745605 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op74);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #74" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op74, xnn_delete_operator);

  xnn_operator_t op75 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/48,
    /*group_output_channels=*/288,
    /*input_channel_stride=*/48,
    /*output_channel_stride=*/288,
    /*input_zero_point=*/(uint8_t) 137,
    /*input_scale=*/0.48052287101745605,
    /*kernel_zero_point=*/(uint8_t) 132,
    /*kernel_scale=*/0.0014037908986210823,
    /*kernel=*/w204.data(), /*bias=*/w205.data(),
    /*output_zero_point=*/(uint8_t) 113,
    /*output_scale=*/0.14607380330562592,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op75);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #75" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op75, xnn_delete_operator);

  xnn_operator_t op76 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op76);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #76" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op76, xnn_delete_operator);

  xnn_operator_t op77 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/1, /*input_padding_right=*/2,
    /*input_padding_bottom=*/2, /*input_padding_left=*/1,
    /*kernel_height=*/5, /*kernel_width=*/5,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/288,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/288,
    /*output_channel_stride=*/288,
    /*input_zero_point=*/(uint8_t) 5,
    /*input_scale=*/0.07805965095758438,
    /*kernel_zero_point=*/(uint8_t) 105,
    /*kernel_scale=*/0.035035692155361176,
    /*kernel=*/w206.data(), /*bias=*/w207.data(),
    /*output_zero_point=*/(uint8_t) 83,
    /*output_scale=*/0.13729262351989746,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op77);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #77" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op77, xnn_delete_operator);

  xnn_operator_t op78 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op78);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #78" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op78, xnn_delete_operator);

  xnn_operator_t op79 = nullptr;
  status = xnn_create_global_average_pooling_nwc_qu8(
    4 /* input zero point */, 0.08693098276853561 /* input scale */,
    4 /* output zero point */, 0.08693098276853561 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op79);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #79" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op79, xnn_delete_operator);

  xnn_operator_t op80 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/288,
    /*group_output_channels=*/72,
    /*input_channel_stride=*/288,
    /*output_channel_stride=*/72,
    /*input_zero_point=*/(uint8_t) 4,
    /*input_scale=*/0.08693098276853561,
    /*kernel_zero_point=*/(uint8_t) 120,
    /*kernel_scale=*/0.006728844251483679,
    /*kernel=*/w208.data(), /*bias=*/w209.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.016604457050561905,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op80);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #80" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op80, xnn_delete_operator);

  xnn_operator_t op81 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/72,
    /*group_output_channels=*/288,
    /*input_channel_stride=*/72,
    /*output_channel_stride=*/288,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.016604457050561905,
    /*kernel_zero_point=*/(uint8_t) 152,
    /*kernel_scale=*/0.005426046904176474,
    /*kernel=*/w210.data(), /*bias=*/w211.data(),
    /*output_zero_point=*/(uint8_t) 123,
    /*output_scale=*/0.03482932597398758,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op81);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #81" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op81, xnn_delete_operator);

  xnn_operator_t op82 = nullptr;
  status = xnn_create_add_nd_qu8(
    123 /* input1 zero point */, 0.03482932597398758 /* input1 scale */,
    0 /* input2 zero point */, 0.0117647061124444 /* input2 scale */,
    0 /* output zero point */, 0.023528477177023888 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op82);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #82" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op82, xnn_delete_operator);

  xnn_operator_t op83 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    0 /* input1 zero point */, 0.023528477177023888 /* input1 scale */,
    0 /* input2 zero point */, 0.0006536078290082514 /* input2 scale */,
    0 /* output zero point */, 0.003921508323401213 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op83);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #83" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op83, xnn_delete_operator);

  xnn_operator_t op84 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    4 /* input1 zero point */, 0.08693098276853561 /* input1 scale */,
    0 /* input2 zero point */, 0.003921508323401213 /* input2 scale */,
    10 /* output zero point */, 0.03586701303720474 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op84);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #84" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op84, xnn_delete_operator);

  xnn_operator_t op85 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/288,
    /*group_output_channels=*/96,
    /*input_channel_stride=*/288,
    /*output_channel_stride=*/96,
    /*input_zero_point=*/(uint8_t) 10,
    /*input_scale=*/0.03586701303720474,
    /*kernel_zero_point=*/(uint8_t) 122,
    /*kernel_scale=*/0.019641198217868805,
    /*kernel=*/w214.data(), /*bias=*/w215.data(),
    /*output_zero_point=*/(uint8_t) 130,
    /*output_scale=*/0.2735706567764282,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op85);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #85" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op85, xnn_delete_operator);

  xnn_operator_t op86 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/96,
    /*group_output_channels=*/576,
    /*input_channel_stride=*/96,
    /*output_channel_stride=*/576,
    /*input_zero_point=*/(uint8_t) 130,
    /*input_scale=*/0.2735706567764282,
    /*kernel_zero_point=*/(uint8_t) 145,
    /*kernel_scale=*/0.0017236428102478385,
    /*kernel=*/w216.data(), /*bias=*/w217.data(),
    /*output_zero_point=*/(uint8_t) 118,
    /*output_scale=*/0.14194171130657196,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op86);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #86" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op86, xnn_delete_operator);

  xnn_operator_t op87 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op87);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #87" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op87, xnn_delete_operator);

  xnn_operator_t op88 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/2, /*input_padding_right=*/2,
    /*input_padding_bottom=*/2, /*input_padding_left=*/2,
    /*kernel_height=*/5, /*kernel_width=*/5,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/576,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/576,
    /*output_channel_stride=*/576,
    /*input_zero_point=*/(uint8_t) 5,
    /*input_scale=*/0.07257640361785889,
    /*kernel_zero_point=*/(uint8_t) 96,
    /*kernel_scale=*/0.174177348613739,
    /*kernel=*/w218.data(), /*bias=*/w219.data(),
    /*output_zero_point=*/(uint8_t) 104,
    /*output_scale=*/0.23463939130306244,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op88);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #88" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op88, xnn_delete_operator);

  xnn_operator_t op89 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op89);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #89" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op89, xnn_delete_operator);

  xnn_operator_t op90 = nullptr;
  status = xnn_create_global_average_pooling_nwc_qu8(
    3 /* input zero point */, 0.1241951510310173 /* input scale */,
    3 /* output zero point */, 0.1241951510310173 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op90);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #90" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op90, xnn_delete_operator);

  xnn_operator_t op91 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/576,
    /*group_output_channels=*/144,
    /*input_channel_stride=*/576,
    /*output_channel_stride=*/144,
    /*input_zero_point=*/(uint8_t) 3,
    /*input_scale=*/0.1241951510310173,
    /*kernel_zero_point=*/(uint8_t) 115,
    /*kernel_scale=*/0.005609261337667704,
    /*kernel=*/w220.data(), /*bias=*/w221.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.014928853139281273,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op91);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #91" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op91, xnn_delete_operator);

  xnn_operator_t op92 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/144,
    /*group_output_channels=*/576,
    /*input_channel_stride=*/144,
    /*output_channel_stride=*/576,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.014928853139281273,
    /*kernel_zero_point=*/(uint8_t) 91,
    /*kernel_scale=*/0.008804556913673878,
    /*kernel=*/w222.data(), /*bias=*/w223.data(),
    /*output_zero_point=*/(uint8_t) 129,
    /*output_scale=*/0.04489157348871231,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op92);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #92" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op92, xnn_delete_operator);

  xnn_operator_t op93 = nullptr;
  status = xnn_create_add_nd_qu8(
    129 /* input1 zero point */, 0.04489157348871231 /* input1 scale */,
    0 /* input2 zero point */, 0.0117647061124444 /* input2 scale */,
    0 /* output zero point */, 0.023463299497961998 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op93);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #93" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op93, xnn_delete_operator);

  xnn_operator_t op94 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    0 /* input1 zero point */, 0.023463299497961998 /* input1 scale */,
    0 /* input2 zero point */, 0.0006536078290082514 /* input2 scale */,
    0 /* output zero point */, 0.003899596631526947 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op94);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #94" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op94, xnn_delete_operator);

  xnn_operator_t op95 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    3 /* input1 zero point */, 0.1241951510310173 /* input1 scale */,
    0 /* input2 zero point */, 0.003899596631526947 /* input2 scale */,
    15 /* output zero point */, 0.023340530693531036 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op95);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #95" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op95, xnn_delete_operator);

  xnn_operator_t op96 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/576,
    /*group_output_channels=*/96,
    /*input_channel_stride=*/576,
    /*output_channel_stride=*/96,
    /*input_zero_point=*/(uint8_t) 15,
    /*input_scale=*/0.023340530693531036,
    /*kernel_zero_point=*/(uint8_t) 132,
    /*kernel_scale=*/0.11193376779556274,
    /*kernel=*/w226.data(), /*bias=*/w227.data(),
    /*output_zero_point=*/(uint8_t) 131,
    /*output_scale=*/0.3130902945995331,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op96);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #96" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op96, xnn_delete_operator);

  xnn_operator_t op97 = nullptr;
  status = xnn_create_add_nd_qu8(
    131 /* input1 zero point */, 0.3130902945995331 /* input1 scale */,
    130 /* input2 zero point */, 0.2735706567764282 /* input2 scale */,
    130 /* output zero point */, 0.3734561800956726 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op97);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #97" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op97, xnn_delete_operator);

  xnn_operator_t op98 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/96,
    /*group_output_channels=*/576,
    /*input_channel_stride=*/96,
    /*output_channel_stride=*/576,
    /*input_zero_point=*/(uint8_t) 130,
    /*input_scale=*/0.3734561800956726,
    /*kernel_zero_point=*/(uint8_t) 153,
    /*kernel_scale=*/0.0030694138258695602,
    /*kernel=*/w228.data(), /*bias=*/w229.data(),
    /*output_zero_point=*/(uint8_t) 157,
    /*output_scale=*/0.3907496929168701,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op98);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #98" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op98, xnn_delete_operator);

  xnn_operator_t op99 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op99);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #99" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op99, xnn_delete_operator);

  xnn_operator_t op100 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/2, /*input_padding_right=*/2,
    /*input_padding_bottom=*/2, /*input_padding_left=*/2,
    /*kernel_height=*/5, /*kernel_width=*/5,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/576,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/576,
    /*output_channel_stride=*/576,
    /*input_zero_point=*/(uint8_t) 3,
    /*input_scale=*/0.1398279368877411,
    /*kernel_zero_point=*/(uint8_t) 218,
    /*kernel_scale=*/2.1697041988372803,
    /*kernel=*/w230.data(), /*bias=*/w231.data(),
    /*output_zero_point=*/(uint8_t) 110,
    /*output_scale=*/0.6755003929138184,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op100);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #100" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op100, xnn_delete_operator);

  xnn_operator_t op101 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op101);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #101" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op101, xnn_delete_operator);

  xnn_operator_t op102 = nullptr;
  status = xnn_create_global_average_pooling_nwc_qu8(
    1 /* input zero point */, 0.3347671329975128 /* input scale */,
    1 /* output zero point */, 0.3347671329975128 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op102);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #102" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op102, xnn_delete_operator);

  xnn_operator_t op103 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/576,
    /*group_output_channels=*/144,
    /*input_channel_stride=*/576,
    /*output_channel_stride=*/144,
    /*input_zero_point=*/(uint8_t) 1,
    /*input_scale=*/0.3347671329975128,
    /*kernel_zero_point=*/(uint8_t) 96,
    /*kernel_scale=*/0.006274337414652109,
    /*kernel=*/w232.data(), /*bias=*/w233.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.04336833581328392,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op103);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #103" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op103, xnn_delete_operator);

  xnn_operator_t op104 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/144,
    /*group_output_channels=*/576,
    /*input_channel_stride=*/144,
    /*output_channel_stride=*/576,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.04336833581328392,
    /*kernel_zero_point=*/(uint8_t) 91,
    /*kernel_scale=*/0.008546789176762104,
    /*kernel=*/w234.data(), /*bias=*/w235.data(),
    /*output_zero_point=*/(uint8_t) 115,
    /*output_scale=*/0.09501760452985764,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op104);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #104" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op104, xnn_delete_operator);

  xnn_operator_t op105 = nullptr;
  status = xnn_create_add_nd_qu8(
    115 /* input1 zero point */, 0.09501760452985764 /* input1 scale */,
    0 /* input2 zero point */, 0.0117647061124444 /* input2 scale */,
    0 /* output zero point */, 0.023528477177023888 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op105);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #105" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op105, xnn_delete_operator);

  xnn_operator_t op106 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    0 /* input1 zero point */, 0.023528477177023888 /* input1 scale */,
    0 /* input2 zero point */, 0.0006536078290082514 /* input2 scale */,
    0 /* output zero point */, 0.003921508323401213 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op106);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #106" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op106, xnn_delete_operator);

  xnn_operator_t op107 = nullptr;
  status = xnn_create_multiply_nd_qu8(
    1 /* input1 zero point */, 0.3347671329975128 /* input1 scale */,
    0 /* input2 zero point */, 0.003921508323401213 /* input2 scale */,
    2 /* output zero point */, 0.19521307945251465 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op107);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #107" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op107, xnn_delete_operator);

  xnn_operator_t op108 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/576,
    /*group_output_channels=*/96,
    /*input_channel_stride=*/576,
    /*output_channel_stride=*/96,
    /*input_zero_point=*/(uint8_t) 2,
    /*input_scale=*/0.19521307945251465,
    /*kernel_zero_point=*/(uint8_t) 130,
    /*kernel_scale=*/0.02609884925186634,
    /*kernel=*/w238.data(), /*bias=*/w239.data(),
    /*output_zero_point=*/(uint8_t) 129,
    /*output_scale=*/0.7081664800643921,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op108);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #108" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op108, xnn_delete_operator);

  xnn_operator_t op109 = nullptr;
  status = xnn_create_add_nd_qu8(
    129 /* input1 zero point */, 0.7081664800643921 /* input1 scale */,
    130 /* input2 zero point */, 0.3734561800956726 /* input2 scale */,
    127 /* output zero point */, 0.808801531791687 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op109);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #109" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op109, xnn_delete_operator);

  xnn_operator_t op110 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/96,
    /*group_output_channels=*/576,
    /*input_channel_stride=*/96,
    /*output_channel_stride=*/576,
    /*input_zero_point=*/(uint8_t) 127,
    /*input_scale=*/0.808801531791687,
    /*kernel_zero_point=*/(uint8_t) 142,
    /*kernel_scale=*/0.003396135289222002,
    /*kernel=*/w240.data(), /*bias=*/w241.data(),
    /*output_zero_point=*/(uint8_t) 131,
    /*output_scale=*/0.9106870889663696,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op110);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #110" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op110, xnn_delete_operator);

  xnn_operator_t op111 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op111);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #111" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op111, xnn_delete_operator);

  xnn_operator_t op112 = nullptr;
  status = xnn_create_global_average_pooling_nwc_qu8(
    1 /* input zero point */, 0.40212398767471313 /* input scale */,
    1 /* output zero point */, 0.40212398767471313 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op112);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #112" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op112, xnn_delete_operator);

  xnn_operator_t op113 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/576,
    /*group_output_channels=*/1024,
    /*input_channel_stride=*/576,
    /*output_channel_stride=*/1024,
    /*input_zero_point=*/(uint8_t) 1,
    /*input_scale=*/0.40212398767471313,
    /*kernel_zero_point=*/(uint8_t) 97,
    /*kernel_scale=*/0.006370874121785164,
    /*kernel=*/w242.data(), /*bias=*/w243.data(),
    /*output_zero_point=*/(uint8_t) 170,
    /*output_scale=*/0.05783478170633316,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op113);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #113" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op113, xnn_delete_operator);

  xnn_operator_t op114 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op114);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #114" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op114, xnn_delete_operator);

  xnn_operator_t op115 = nullptr;
  status = xnn_create_global_average_pooling_nwc_qu8(
    19 /* input zero point */, 0.01954001374542713 /* input scale */,
    19 /* output zero point */, 0.01954001374542713 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op115);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #115" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op115, xnn_delete_operator);

  xnn_operator_t op116 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/1024,
    /*group_output_channels=*/1001,
    /*input_channel_stride=*/1024,
    /*output_channel_stride=*/1001,
    /*input_zero_point=*/(uint8_t) 19,
    /*input_scale=*/0.01954001374542713,
    /*kernel_zero_point=*/(uint8_t) 113,
    /*kernel_scale=*/0.0029929860029369593,
    /*kernel=*/w244.data(), /*bias=*/w245.data(),
    /*output_zero_point=*/(uint8_t) 77,
    /*output_scale=*/0.07862140238285065,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op116);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #116" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op116, xnn_delete_operator);

  xnn_operator_t op117 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op117);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #117" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op117, xnn_delete_operator);

  xnn_operator_t op118 = nullptr;
  status = xnn_create_softmax_nc_qu8(
    /*input_scale=*/0.07862140238285065,
    /*output_zero_point=*/0,
    /*output_scale=*/0.00390625,
    /*flags=*/0,
    &op118);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #118" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op118, xnn_delete_operator);

  size_t op0_workspace_size = 0;
  size_t op0_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op0,
    /*batch_size=*/1, /*input_height=*/224, /*input_width=*/224,
    &op0_workspace_size, &op0_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op0_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op1,
    /*batch_size=*/12544,
    16 /* channels */,
    16 /* input stride */,
    16 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #1" << std::endl;
    return ExecutionPlan();
  }

  size_t op2_workspace_size = 0;
  size_t op2_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op2,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    &op2_workspace_size, &op2_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op2_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #2" << std::endl;
    return ExecutionPlan();
  }

  size_t op3_workspace_size = 0;
  size_t op3_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_qu8(
    op3,
    /*batch_size=*/1, 3136 /* width */,
    16 /* channels */, 16 /* input stride */, 16 /* output stride */,
    &op3_workspace_size, &op3_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op3_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #3" << std::endl;
    return ExecutionPlan();
  }

  size_t op4_workspace_size = 0;
  size_t op4_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op4,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op4_workspace_size, &op4_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op4_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #4" << std::endl;
    return ExecutionPlan();
  }

  size_t op5_workspace_size = 0;
  size_t op5_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op5,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op5_workspace_size, &op5_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op5_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #5" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 16 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_add_nd_qu8(
      op6,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #6" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 16 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_multiply_nd_qu8(
      op7,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #7" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 56, 56, 16 };
    const size_t b_shape[] = { 1, 1, 1, 16 };
    status = xnn_reshape_multiply_nd_qu8(
      op8,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #8" << std::endl;
    return ExecutionPlan();
  }

  size_t op9_workspace_size = 0;
  size_t op9_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op9,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    &op9_workspace_size, &op9_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op9_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #9" << std::endl;
    return ExecutionPlan();
  }

  size_t op10_workspace_size = 0;
  size_t op10_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op10,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    &op10_workspace_size, &op10_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op10_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #10" << std::endl;
    return ExecutionPlan();
  }

  size_t op11_workspace_size = 0;
  size_t op11_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op11,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    &op11_workspace_size, &op11_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op11_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #11" << std::endl;
    return ExecutionPlan();
  }

  size_t op12_workspace_size = 0;
  size_t op12_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op12,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op12_workspace_size, &op12_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op12_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #12" << std::endl;
    return ExecutionPlan();
  }

  size_t op13_workspace_size = 0;
  size_t op13_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op13,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op13_workspace_size, &op13_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op13_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #13" << std::endl;
    return ExecutionPlan();
  }

  size_t op14_workspace_size = 0;
  size_t op14_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op14,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op14_workspace_size, &op14_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op14_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #14" << std::endl;
    return ExecutionPlan();
  }

  size_t op15_workspace_size = 0;
  size_t op15_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op15,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op15_workspace_size, &op15_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op15_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #15" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 28, 28, 24 };
    const size_t b_shape[] = { 1, 28, 28, 24 };
    status = xnn_reshape_add_nd_qu8(
      op16,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #16" << std::endl;
    return ExecutionPlan();
  }

  size_t op17_workspace_size = 0;
  size_t op17_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op17,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op17_workspace_size, &op17_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op17_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op18,
    /*batch_size=*/784,
    96 /* channels */,
    96 /* input stride */,
    96 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #18" << std::endl;
    return ExecutionPlan();
  }

  size_t op19_workspace_size = 0;
  size_t op19_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op19,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op19_workspace_size, &op19_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op19_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #19" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op20,
    /*batch_size=*/196,
    96 /* channels */,
    96 /* input stride */,
    96 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #20" << std::endl;
    return ExecutionPlan();
  }

  size_t op21_workspace_size = 0;
  size_t op21_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_qu8(
    op21,
    /*batch_size=*/1, 196 /* width */,
    96 /* channels */, 96 /* input stride */, 96 /* output stride */,
    &op21_workspace_size, &op21_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op21_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #21" << std::endl;
    return ExecutionPlan();
  }

  size_t op22_workspace_size = 0;
  size_t op22_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op22,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op22_workspace_size, &op22_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op22_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #22" << std::endl;
    return ExecutionPlan();
  }

  size_t op23_workspace_size = 0;
  size_t op23_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op23,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op23_workspace_size, &op23_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op23_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #23" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 96 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_add_nd_qu8(
      op24,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #24" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 96 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_multiply_nd_qu8(
      op25,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #25" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 96 };
    const size_t b_shape[] = { 1, 1, 1, 96 };
    status = xnn_reshape_multiply_nd_qu8(
      op26,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #26" << std::endl;
    return ExecutionPlan();
  }

  size_t op27_workspace_size = 0;
  size_t op27_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op27,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op27_workspace_size, &op27_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op27_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #27" << std::endl;
    return ExecutionPlan();
  }

  size_t op28_workspace_size = 0;
  size_t op28_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op28,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op28_workspace_size, &op28_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op28_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #28" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op29,
    /*batch_size=*/196,
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #29" << std::endl;
    return ExecutionPlan();
  }

  size_t op30_workspace_size = 0;
  size_t op30_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op30,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op30_workspace_size, &op30_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op30_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #30" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op31,
    /*batch_size=*/196,
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #31" << std::endl;
    return ExecutionPlan();
  }

  size_t op32_workspace_size = 0;
  size_t op32_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_qu8(
    op32,
    /*batch_size=*/1, 196 /* width */,
    240 /* channels */, 240 /* input stride */, 240 /* output stride */,
    &op32_workspace_size, &op32_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op32_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #32" << std::endl;
    return ExecutionPlan();
  }

  size_t op33_workspace_size = 0;
  size_t op33_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op33,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op33_workspace_size, &op33_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op33_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #33" << std::endl;
    return ExecutionPlan();
  }

  size_t op34_workspace_size = 0;
  size_t op34_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op34,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op34_workspace_size, &op34_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op34_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #34" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 240 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_add_nd_qu8(
      op35,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #35" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 240 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_multiply_nd_qu8(
      op36,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #36" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 240 };
    const size_t b_shape[] = { 1, 1, 1, 240 };
    status = xnn_reshape_multiply_nd_qu8(
      op37,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #37" << std::endl;
    return ExecutionPlan();
  }

  size_t op38_workspace_size = 0;
  size_t op38_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op38,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op38_workspace_size, &op38_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op38_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #38" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 40 };
    const size_t b_shape[] = { 1, 14, 14, 40 };
    status = xnn_reshape_add_nd_qu8(
      op39,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #39" << std::endl;
    return ExecutionPlan();
  }

  size_t op40_workspace_size = 0;
  size_t op40_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op40,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op40_workspace_size, &op40_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op40_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #40" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op41,
    /*batch_size=*/196,
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #41" << std::endl;
    return ExecutionPlan();
  }

  size_t op42_workspace_size = 0;
  size_t op42_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op42,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op42_workspace_size, &op42_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op42_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #42" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op43,
    /*batch_size=*/196,
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #43" << std::endl;
    return ExecutionPlan();
  }

  size_t op44_workspace_size = 0;
  size_t op44_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_qu8(
    op44,
    /*batch_size=*/1, 196 /* width */,
    240 /* channels */, 240 /* input stride */, 240 /* output stride */,
    &op44_workspace_size, &op44_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op44_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #44" << std::endl;
    return ExecutionPlan();
  }

  size_t op45_workspace_size = 0;
  size_t op45_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op45,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op45_workspace_size, &op45_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op45_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #45" << std::endl;
    return ExecutionPlan();
  }

  size_t op46_workspace_size = 0;
  size_t op46_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op46,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op46_workspace_size, &op46_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op46_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #46" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 240 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_add_nd_qu8(
      op47,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #47" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 240 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_multiply_nd_qu8(
      op48,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #48" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 240 };
    const size_t b_shape[] = { 1, 1, 1, 240 };
    status = xnn_reshape_multiply_nd_qu8(
      op49,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #49" << std::endl;
    return ExecutionPlan();
  }

  size_t op50_workspace_size = 0;
  size_t op50_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op50,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op50_workspace_size, &op50_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op50_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #50" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 40 };
    const size_t b_shape[] = { 1, 14, 14, 40 };
    status = xnn_reshape_add_nd_qu8(
      op51,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #51" << std::endl;
    return ExecutionPlan();
  }

  size_t op52_workspace_size = 0;
  size_t op52_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op52,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op52_workspace_size, &op52_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op52_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #52" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op53,
    /*batch_size=*/196,
    120 /* channels */,
    120 /* input stride */,
    120 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #53" << std::endl;
    return ExecutionPlan();
  }

  size_t op54_workspace_size = 0;
  size_t op54_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op54,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op54_workspace_size, &op54_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op54_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #54" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op55,
    /*batch_size=*/196,
    120 /* channels */,
    120 /* input stride */,
    120 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #55" << std::endl;
    return ExecutionPlan();
  }

  size_t op56_workspace_size = 0;
  size_t op56_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_qu8(
    op56,
    /*batch_size=*/1, 196 /* width */,
    120 /* channels */, 120 /* input stride */, 120 /* output stride */,
    &op56_workspace_size, &op56_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op56_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #56" << std::endl;
    return ExecutionPlan();
  }

  size_t op57_workspace_size = 0;
  size_t op57_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op57,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op57_workspace_size, &op57_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op57_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #57" << std::endl;
    return ExecutionPlan();
  }

  size_t op58_workspace_size = 0;
  size_t op58_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op58,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op58_workspace_size, &op58_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op58_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #58" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 120 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_add_nd_qu8(
      op59,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #59" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 120 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_multiply_nd_qu8(
      op60,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #60" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 120 };
    const size_t b_shape[] = { 1, 1, 1, 120 };
    status = xnn_reshape_multiply_nd_qu8(
      op61,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #61" << std::endl;
    return ExecutionPlan();
  }

  size_t op62_workspace_size = 0;
  size_t op62_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op62,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op62_workspace_size, &op62_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op62_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #62" << std::endl;
    return ExecutionPlan();
  }

  size_t op63_workspace_size = 0;
  size_t op63_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op63,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op63_workspace_size, &op63_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op63_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #63" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op64,
    /*batch_size=*/196,
    144 /* channels */,
    144 /* input stride */,
    144 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #64" << std::endl;
    return ExecutionPlan();
  }

  size_t op65_workspace_size = 0;
  size_t op65_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op65,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op65_workspace_size, &op65_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op65_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #65" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op66,
    /*batch_size=*/196,
    144 /* channels */,
    144 /* input stride */,
    144 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #66" << std::endl;
    return ExecutionPlan();
  }

  size_t op67_workspace_size = 0;
  size_t op67_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_qu8(
    op67,
    /*batch_size=*/1, 196 /* width */,
    144 /* channels */, 144 /* input stride */, 144 /* output stride */,
    &op67_workspace_size, &op67_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op67_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #67" << std::endl;
    return ExecutionPlan();
  }

  size_t op68_workspace_size = 0;
  size_t op68_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op68,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op68_workspace_size, &op68_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op68_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #68" << std::endl;
    return ExecutionPlan();
  }

  size_t op69_workspace_size = 0;
  size_t op69_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op69,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op69_workspace_size, &op69_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op69_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #69" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 144 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_add_nd_qu8(
      op70,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #70" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 144 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_multiply_nd_qu8(
      op71,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #71" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 144 };
    const size_t b_shape[] = { 1, 1, 1, 144 };
    status = xnn_reshape_multiply_nd_qu8(
      op72,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #72" << std::endl;
    return ExecutionPlan();
  }

  size_t op73_workspace_size = 0;
  size_t op73_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op73,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op73_workspace_size, &op73_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op73_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #73" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 48 };
    const size_t b_shape[] = { 1, 14, 14, 48 };
    status = xnn_reshape_add_nd_qu8(
      op74,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #74" << std::endl;
    return ExecutionPlan();
  }

  size_t op75_workspace_size = 0;
  size_t op75_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op75,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op75_workspace_size, &op75_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op75_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #75" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op76,
    /*batch_size=*/196,
    288 /* channels */,
    288 /* input stride */,
    288 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #76" << std::endl;
    return ExecutionPlan();
  }

  size_t op77_workspace_size = 0;
  size_t op77_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op77,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op77_workspace_size, &op77_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op77_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #77" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op78,
    /*batch_size=*/49,
    288 /* channels */,
    288 /* input stride */,
    288 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #78" << std::endl;
    return ExecutionPlan();
  }

  size_t op79_workspace_size = 0;
  size_t op79_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_qu8(
    op79,
    /*batch_size=*/1, 49 /* width */,
    288 /* channels */, 288 /* input stride */, 288 /* output stride */,
    &op79_workspace_size, &op79_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op79_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #79" << std::endl;
    return ExecutionPlan();
  }

  size_t op80_workspace_size = 0;
  size_t op80_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op80,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op80_workspace_size, &op80_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op80_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #80" << std::endl;
    return ExecutionPlan();
  }

  size_t op81_workspace_size = 0;
  size_t op81_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op81,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op81_workspace_size, &op81_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op81_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #81" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 288 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_add_nd_qu8(
      op82,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #82" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 288 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_multiply_nd_qu8(
      op83,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #83" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 288 };
    const size_t b_shape[] = { 1, 1, 1, 288 };
    status = xnn_reshape_multiply_nd_qu8(
      op84,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #84" << std::endl;
    return ExecutionPlan();
  }

  size_t op85_workspace_size = 0;
  size_t op85_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op85,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op85_workspace_size, &op85_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op85_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #85" << std::endl;
    return ExecutionPlan();
  }

  size_t op86_workspace_size = 0;
  size_t op86_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op86,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op86_workspace_size, &op86_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op86_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #86" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op87,
    /*batch_size=*/49,
    576 /* channels */,
    576 /* input stride */,
    576 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #87" << std::endl;
    return ExecutionPlan();
  }

  size_t op88_workspace_size = 0;
  size_t op88_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op88,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op88_workspace_size, &op88_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op88_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #88" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op89,
    /*batch_size=*/49,
    576 /* channels */,
    576 /* input stride */,
    576 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #89" << std::endl;
    return ExecutionPlan();
  }

  size_t op90_workspace_size = 0;
  size_t op90_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_qu8(
    op90,
    /*batch_size=*/1, 49 /* width */,
    576 /* channels */, 576 /* input stride */, 576 /* output stride */,
    &op90_workspace_size, &op90_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op90_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #90" << std::endl;
    return ExecutionPlan();
  }

  size_t op91_workspace_size = 0;
  size_t op91_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op91,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op91_workspace_size, &op91_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op91_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #91" << std::endl;
    return ExecutionPlan();
  }

  size_t op92_workspace_size = 0;
  size_t op92_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op92,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op92_workspace_size, &op92_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op92_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #92" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 576 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_add_nd_qu8(
      op93,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #93" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 576 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_multiply_nd_qu8(
      op94,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #94" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 576 };
    const size_t b_shape[] = { 1, 1, 1, 576 };
    status = xnn_reshape_multiply_nd_qu8(
      op95,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #95" << std::endl;
    return ExecutionPlan();
  }

  size_t op96_workspace_size = 0;
  size_t op96_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op96,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op96_workspace_size, &op96_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op96_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #96" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 96 };
    const size_t b_shape[] = { 1, 7, 7, 96 };
    status = xnn_reshape_add_nd_qu8(
      op97,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #97" << std::endl;
    return ExecutionPlan();
  }

  size_t op98_workspace_size = 0;
  size_t op98_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op98,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op98_workspace_size, &op98_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op98_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #98" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op99,
    /*batch_size=*/49,
    576 /* channels */,
    576 /* input stride */,
    576 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #99" << std::endl;
    return ExecutionPlan();
  }

  size_t op100_workspace_size = 0;
  size_t op100_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op100,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op100_workspace_size, &op100_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op100_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #100" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op101,
    /*batch_size=*/49,
    576 /* channels */,
    576 /* input stride */,
    576 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #101" << std::endl;
    return ExecutionPlan();
  }

  size_t op102_workspace_size = 0;
  size_t op102_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_qu8(
    op102,
    /*batch_size=*/1, 49 /* width */,
    576 /* channels */, 576 /* input stride */, 576 /* output stride */,
    &op102_workspace_size, &op102_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op102_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #102" << std::endl;
    return ExecutionPlan();
  }

  size_t op103_workspace_size = 0;
  size_t op103_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op103,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op103_workspace_size, &op103_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op103_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #103" << std::endl;
    return ExecutionPlan();
  }

  size_t op104_workspace_size = 0;
  size_t op104_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op104,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op104_workspace_size, &op104_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op104_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #104" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 576 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_add_nd_qu8(
      op105,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #105" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 1, 1, 576 };
    const size_t b_shape[] = { 1 };
    status = xnn_reshape_multiply_nd_qu8(
      op106,
      4, a_shape, 1, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #106" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 576 };
    const size_t b_shape[] = { 1, 1, 1, 576 };
    status = xnn_reshape_multiply_nd_qu8(
      op107,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #107" << std::endl;
    return ExecutionPlan();
  }

  size_t op108_workspace_size = 0;
  size_t op108_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op108,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op108_workspace_size, &op108_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op108_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #108" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 96 };
    const size_t b_shape[] = { 1, 7, 7, 96 };
    status = xnn_reshape_add_nd_qu8(
      op109,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #109" << std::endl;
    return ExecutionPlan();
  }

  size_t op110_workspace_size = 0;
  size_t op110_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op110,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op110_workspace_size, &op110_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op110_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #110" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op111,
    /*batch_size=*/49,
    576 /* channels */,
    576 /* input stride */,
    576 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #111" << std::endl;
    return ExecutionPlan();
  }

  size_t op112_workspace_size = 0;
  size_t op112_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_qu8(
    op112,
    /*batch_size=*/1, 49 /* width */,
    576 /* channels */, 576 /* input stride */, 576 /* output stride */,
    &op112_workspace_size, &op112_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op112_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #112" << std::endl;
    return ExecutionPlan();
  }

  size_t op113_workspace_size = 0;
  size_t op113_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op113,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op113_workspace_size, &op113_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op113_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #113" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op114,
    /*batch_size=*/1,
    1024 /* channels */,
    1024 /* input stride */,
    1024 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #114" << std::endl;
    return ExecutionPlan();
  }

  size_t op115_workspace_size = 0;
  size_t op115_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_qu8(
    op115,
    /*batch_size=*/1, 1 /* width */,
    1024 /* channels */, 1024 /* input stride */, 1024 /* output stride */,
    &op115_workspace_size, &op115_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op115_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #115" << std::endl;
    return ExecutionPlan();
  }

  size_t op116_workspace_size = 0;
  size_t op116_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op116,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op116_workspace_size, &op116_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op116_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #116" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op117,
    /*batch_size=*/1001,
    1 /* channels */,
    1 /* input stride */,
    1 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #117" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_softmax_nc_qu8(
    op118,
    /*channels=*/1001,
    /*input_stride=*/1001,
    /*output_stride=*/1001,
    /*batch_size=*/1,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #118" << std::endl;
    return ExecutionPlan();
  }

  Workspace workspace(max_workspace_size);

  status = xnn_setup_convolution2d_nhwc_qu8(
    op0,
    workspace.data(), /*input=*/v0.data(), /*output=*/v1.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op1,
    /*input=*/v1.data(), /*output=*/v2.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #1" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op2,
    workspace.data(), /*input=*/v2.data(), /*output=*/v3.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_qu8(
    op3,
    workspace.data(),
    /*input=*/v3.data(), /*output=*/v4.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #3" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op4,
    workspace.data(), /*input=*/v4.data(), /*output=*/v5.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op5,
    workspace.data(), /*input=*/v5.data(), /*output=*/v6.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #5" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op6,
    v6.data() /* a */, w128.data() /* b */, /*output=*/v7.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op7,
    v7.data() /* a */, w129.data() /* b */, /*output=*/v8.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op8,
    v3.data() /* a */, v8.data() /* b */, /*output=*/v9.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #8" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op9,
    workspace.data(), /*input=*/v9.data(), /*output=*/v10.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op10,
    workspace.data(), /*input=*/v10.data(), /*output=*/v11.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #10" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op11,
    workspace.data(), /*input=*/v11.data(), /*output=*/v12.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op12,
    workspace.data(), /*input=*/v12.data(), /*output=*/v13.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op13,
    workspace.data(), /*input=*/v13.data(), /*output=*/v14.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #13" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op14,
    workspace.data(), /*input=*/v14.data(), /*output=*/v15.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op15,
    workspace.data(), /*input=*/v15.data(), /*output=*/v16.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #15" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op16,
    v16.data() /* a */, v13.data() /* b */, /*output=*/v17.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #16" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op17,
    workspace.data(), /*input=*/v17.data(), /*output=*/v18.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op18,
    /*input=*/v18.data(), /*output=*/v19.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op19,
    workspace.data(), /*input=*/v19.data(), /*output=*/v20.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #19" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op20,
    /*input=*/v20.data(), /*output=*/v21.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_qu8(
    op21,
    workspace.data(),
    /*input=*/v21.data(), /*output=*/v22.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #21" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op22,
    workspace.data(), /*input=*/v22.data(), /*output=*/v23.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op23,
    workspace.data(), /*input=*/v23.data(), /*output=*/v24.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #23" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op24,
    v24.data() /* a */, w152.data() /* b */, /*output=*/v25.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op25,
    v25.data() /* a */, w153.data() /* b */, /*output=*/v26.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #25" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op26,
    v21.data() /* a */, v26.data() /* b */, /*output=*/v27.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #26" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op27,
    workspace.data(), /*input=*/v27.data(), /*output=*/v28.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op28,
    workspace.data(), /*input=*/v28.data(), /*output=*/v29.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #28" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op29,
    /*input=*/v29.data(), /*output=*/v30.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #29" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op30,
    workspace.data(), /*input=*/v30.data(), /*output=*/v31.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #30" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op31,
    /*input=*/v31.data(), /*output=*/v32.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #31" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_qu8(
    op32,
    workspace.data(),
    /*input=*/v32.data(), /*output=*/v33.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #32" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op33,
    workspace.data(), /*input=*/v33.data(), /*output=*/v34.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #33" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op34,
    workspace.data(), /*input=*/v34.data(), /*output=*/v35.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #34" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op35,
    v35.data() /* a */, w164.data() /* b */, /*output=*/v36.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #35" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op36,
    v36.data() /* a */, w165.data() /* b */, /*output=*/v37.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #36" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op37,
    v32.data() /* a */, v37.data() /* b */, /*output=*/v38.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #37" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op38,
    workspace.data(), /*input=*/v38.data(), /*output=*/v39.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #38" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op39,
    v39.data() /* a */, v28.data() /* b */, /*output=*/v40.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #39" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op40,
    workspace.data(), /*input=*/v40.data(), /*output=*/v41.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #40" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op41,
    /*input=*/v41.data(), /*output=*/v42.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #41" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op42,
    workspace.data(), /*input=*/v42.data(), /*output=*/v43.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #42" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op43,
    /*input=*/v43.data(), /*output=*/v44.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #43" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_qu8(
    op44,
    workspace.data(),
    /*input=*/v44.data(), /*output=*/v45.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #44" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op45,
    workspace.data(), /*input=*/v45.data(), /*output=*/v46.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #45" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op46,
    workspace.data(), /*input=*/v46.data(), /*output=*/v47.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #46" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op47,
    v47.data() /* a */, w176.data() /* b */, /*output=*/v48.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #47" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op48,
    v48.data() /* a */, w177.data() /* b */, /*output=*/v49.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #48" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op49,
    v44.data() /* a */, v49.data() /* b */, /*output=*/v50.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #49" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op50,
    workspace.data(), /*input=*/v50.data(), /*output=*/v51.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #50" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op51,
    v51.data() /* a */, v40.data() /* b */, /*output=*/v52.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #51" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op52,
    workspace.data(), /*input=*/v52.data(), /*output=*/v53.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #52" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op53,
    /*input=*/v53.data(), /*output=*/v54.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #53" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op54,
    workspace.data(), /*input=*/v54.data(), /*output=*/v55.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #54" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op55,
    /*input=*/v55.data(), /*output=*/v56.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #55" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_qu8(
    op56,
    workspace.data(),
    /*input=*/v56.data(), /*output=*/v57.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #56" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op57,
    workspace.data(), /*input=*/v57.data(), /*output=*/v58.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #57" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op58,
    workspace.data(), /*input=*/v58.data(), /*output=*/v59.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #58" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op59,
    v59.data() /* a */, w188.data() /* b */, /*output=*/v60.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #59" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op60,
    v60.data() /* a */, w189.data() /* b */, /*output=*/v61.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #60" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op61,
    v56.data() /* a */, v61.data() /* b */, /*output=*/v62.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #61" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op62,
    workspace.data(), /*input=*/v62.data(), /*output=*/v63.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #62" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op63,
    workspace.data(), /*input=*/v63.data(), /*output=*/v64.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #63" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op64,
    /*input=*/v64.data(), /*output=*/v65.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #64" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op65,
    workspace.data(), /*input=*/v65.data(), /*output=*/v66.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #65" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op66,
    /*input=*/v66.data(), /*output=*/v67.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #66" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_qu8(
    op67,
    workspace.data(),
    /*input=*/v67.data(), /*output=*/v68.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #67" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op68,
    workspace.data(), /*input=*/v68.data(), /*output=*/v69.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #68" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op69,
    workspace.data(), /*input=*/v69.data(), /*output=*/v70.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #69" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op70,
    v70.data() /* a */, w200.data() /* b */, /*output=*/v71.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #70" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op71,
    v71.data() /* a */, w201.data() /* b */, /*output=*/v72.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #71" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op72,
    v67.data() /* a */, v72.data() /* b */, /*output=*/v73.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #72" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op73,
    workspace.data(), /*input=*/v73.data(), /*output=*/v74.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #73" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op74,
    v74.data() /* a */, v63.data() /* b */, /*output=*/v75.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #74" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op75,
    workspace.data(), /*input=*/v75.data(), /*output=*/v76.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #75" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op76,
    /*input=*/v76.data(), /*output=*/v77.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #76" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op77,
    workspace.data(), /*input=*/v77.data(), /*output=*/v78.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #77" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op78,
    /*input=*/v78.data(), /*output=*/v79.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #78" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_qu8(
    op79,
    workspace.data(),
    /*input=*/v79.data(), /*output=*/v80.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #79" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op80,
    workspace.data(), /*input=*/v80.data(), /*output=*/v81.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #80" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op81,
    workspace.data(), /*input=*/v81.data(), /*output=*/v82.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #81" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op82,
    v82.data() /* a */, w212.data() /* b */, /*output=*/v83.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #82" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op83,
    v83.data() /* a */, w213.data() /* b */, /*output=*/v84.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #83" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op84,
    v79.data() /* a */, v84.data() /* b */, /*output=*/v85.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #84" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op85,
    workspace.data(), /*input=*/v85.data(), /*output=*/v86.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #85" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op86,
    workspace.data(), /*input=*/v86.data(), /*output=*/v87.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #86" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op87,
    /*input=*/v87.data(), /*output=*/v88.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #87" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op88,
    workspace.data(), /*input=*/v88.data(), /*output=*/v89.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #88" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op89,
    /*input=*/v89.data(), /*output=*/v90.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #89" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_qu8(
    op90,
    workspace.data(),
    /*input=*/v90.data(), /*output=*/v91.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #90" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op91,
    workspace.data(), /*input=*/v91.data(), /*output=*/v92.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #91" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op92,
    workspace.data(), /*input=*/v92.data(), /*output=*/v93.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #92" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op93,
    v93.data() /* a */, w224.data() /* b */, /*output=*/v94.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #93" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op94,
    v94.data() /* a */, w225.data() /* b */, /*output=*/v95.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #94" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op95,
    v90.data() /* a */, v95.data() /* b */, /*output=*/v96.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #95" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op96,
    workspace.data(), /*input=*/v96.data(), /*output=*/v97.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #96" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op97,
    v97.data() /* a */, v86.data() /* b */, /*output=*/v98.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #97" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op98,
    workspace.data(), /*input=*/v98.data(), /*output=*/v99.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #98" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op99,
    /*input=*/v99.data(), /*output=*/v100.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #99" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op100,
    workspace.data(), /*input=*/v100.data(), /*output=*/v101.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #100" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op101,
    /*input=*/v101.data(), /*output=*/v102.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #101" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_qu8(
    op102,
    workspace.data(),
    /*input=*/v102.data(), /*output=*/v103.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #102" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op103,
    workspace.data(), /*input=*/v103.data(), /*output=*/v104.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #103" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op104,
    workspace.data(), /*input=*/v104.data(), /*output=*/v105.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #104" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op105,
    v105.data() /* a */, w236.data() /* b */, /*output=*/v106.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #105" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op106,
    v106.data() /* a */, w237.data() /* b */, /*output=*/v107.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #106" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_qu8(
    op107,
    v102.data() /* a */, v107.data() /* b */, /*output=*/v108.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #107" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op108,
    workspace.data(), /*input=*/v108.data(), /*output=*/v109.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #108" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op109,
    v109.data() /* a */, v98.data() /* b */, /*output=*/v110.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #109" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op110,
    workspace.data(), /*input=*/v110.data(), /*output=*/v111.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #110" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op111,
    /*input=*/v111.data(), /*output=*/v112.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #111" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_qu8(
    op112,
    workspace.data(),
    /*input=*/v112.data(), /*output=*/v113.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #112" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op113,
    workspace.data(), /*input=*/v113.data(), /*output=*/v114.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #113" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op114,
    /*input=*/v114.data(), /*output=*/v115.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #114" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_qu8(
    op115,
    workspace.data(),
    /*input=*/v115.data(), /*output=*/v116.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #115" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op116,
    workspace.data(), /*input=*/v116.data(), /*output=*/v117.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #116" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op117,
    /*input=*/v117.data(), /*output=*/v118.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #117" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_softmax_nc_qu8(
    op118,
    /*input=*/v118.data(), /*output=*/v119.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #118" << std::endl;
    return ExecutionPlan();
  }

  XNN_PRAGMA_CLANG("clang diagnostic push")
  XNN_PRAGMA_CLANG("clang diagnostic ignored \"-Wpessimizing-move\"")
  return ExecutionPlan{operators, workspace};
  XNN_PRAGMA_CLANG("clang diagnostic pop")
}

}  // namespace models
