// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack.h"

#include <array>
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include "xnnpack/cache.h"
#include "xnnpack/common.h"
#include "xnnpack/models.h"

#include <fp16/fp16.h>

namespace models {

ExecutionPlan FP16SparseMobileNetV3Large(float sparsity, pthreadpool_t threadpool) {
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(150528, uint16_t)> v0;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(200704, uint16_t)> v1;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(200704, uint16_t)> v2;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(200704, uint16_t)> v3;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(200704, uint16_t)> v4;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(200704, uint16_t)> v5;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(802816, uint16_t)> v6;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(200704, uint16_t)> v7;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(75264, uint16_t)> v8;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(225792, uint16_t)> v9;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(225792, uint16_t)> v10;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(75264, uint16_t)> v11;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(75264, uint16_t)> v12;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(225792, uint16_t)> v13;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(56448, uint16_t)> v14;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(72, uint16_t)> v15;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(24, uint16_t)> v16;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(72, uint16_t)> v17;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(56448, uint16_t)> v18;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(31360, uint16_t)> v19;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(94080, uint16_t)> v20;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(94080, uint16_t)> v21;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(120, uint16_t)> v22;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(32, uint16_t)> v23;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(120, uint16_t)> v24;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(94080, uint16_t)> v25;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(31360, uint16_t)> v26;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(31360, uint16_t)> v27;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(94080, uint16_t)> v28;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(94080, uint16_t)> v29;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(120, uint16_t)> v30;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(32, uint16_t)> v31;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(120, uint16_t)> v32;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(94080, uint16_t)> v33;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(31360, uint16_t)> v34;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(31360, uint16_t)> v35;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(188160, uint16_t)> v36;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(188160, uint16_t)> v37;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(47040, uint16_t)> v38;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(47040, uint16_t)> v39;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(15680, uint16_t)> v40;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(39200, uint16_t)> v41;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(39200, uint16_t)> v42;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(39200, uint16_t)> v43;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(39200, uint16_t)> v44;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(15680, uint16_t)> v45;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(15680, uint16_t)> v46;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(36064, uint16_t)> v47;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(36064, uint16_t)> v48;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(36064, uint16_t)> v49;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(36064, uint16_t)> v50;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(15680, uint16_t)> v51;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(15680, uint16_t)> v52;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(36064, uint16_t)> v53;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(36064, uint16_t)> v54;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(36064, uint16_t)> v55;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(36064, uint16_t)> v56;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(15680, uint16_t)> v57;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(15680, uint16_t)> v58;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(94080, uint16_t)> v59;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(94080, uint16_t)> v60;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(94080, uint16_t)> v61;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(94080, uint16_t)> v62;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(480, uint16_t)> v63;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(120, uint16_t)> v64;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(480, uint16_t)> v65;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(94080, uint16_t)> v66;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(21952, uint16_t)> v67;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(131712, uint16_t)> v68;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(131712, uint16_t)> v69;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(131712, uint16_t)> v70;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(131712, uint16_t)> v71;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(672, uint16_t)> v72;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(168, uint16_t)> v73;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(672, uint16_t)> v74;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(131712, uint16_t)> v75;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(21952, uint16_t)> v76;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(21952, uint16_t)> v77;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(131712, uint16_t)> v78;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(131712, uint16_t)> v79;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(32928, uint16_t)> v80;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(32928, uint16_t)> v81;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(672, uint16_t)> v82;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(168, uint16_t)> v83;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(672, uint16_t)> v84;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(32928, uint16_t)> v85;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(7840, uint16_t)> v86;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(47040, uint16_t)> v87;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(47040, uint16_t)> v88;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(47040, uint16_t)> v89;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(47040, uint16_t)> v90;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(960, uint16_t)> v91;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(240, uint16_t)> v92;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(960, uint16_t)> v93;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(47040, uint16_t)> v94;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(7840, uint16_t)> v95;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(7840, uint16_t)> v96;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(47040, uint16_t)> v97;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(47040, uint16_t)> v98;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(47040, uint16_t)> v99;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(47040, uint16_t)> v100;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(960, uint16_t)> v101;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(240, uint16_t)> v102;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(960, uint16_t)> v103;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(47040, uint16_t)> v104;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(7840, uint16_t)> v105;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(7840, uint16_t)> v106;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(47040, uint16_t)> v107;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(47040, uint16_t)> v108;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(960, uint16_t)> v109;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1280, uint16_t)> v110;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1280, uint16_t)> v111;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1280, uint16_t)> v112;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1001, uint16_t)> v113;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(432, uint16_t)> w114;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(16, uint16_t)> w115;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(144, uint16_t)> w116;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(16, uint16_t)> w117;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(256, uint16_t)> w118;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(16, uint16_t)> w119;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1024, uint16_t)> w120;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(64, uint16_t)> w121;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(576, uint16_t)> w122;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(64, uint16_t)> w123;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1536, uint16_t)> w124;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(24, uint16_t)> w125;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1728, uint16_t)> w126;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(72, uint16_t)> w127;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(648, uint16_t)> w128;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(72, uint16_t)> w129;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1728, uint16_t)> w130;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(24, uint16_t)> w131;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1728, uint16_t)> w132;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(72, uint16_t)> w133;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1800, uint16_t)> w134;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(72, uint16_t)> w135;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1728, uint16_t)> w136;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(24, uint16_t)> w137;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1728, uint16_t)> w138;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(72, uint16_t)> w139;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(2880, uint16_t)> w140;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(40, uint16_t)> w141;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(4800, uint16_t)> w142;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(120, uint16_t)> w143;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(3000, uint16_t)> w144;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(120, uint16_t)> w145;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(3840, uint16_t)> w146;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(32, uint16_t)> w147;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(3840, uint16_t)> w148;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(120, uint16_t)> w149;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(4800, uint16_t)> w150;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(40, uint16_t)> w151;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(4800, uint16_t)> w152;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(120, uint16_t)> w153;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(3000, uint16_t)> w154;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(120, uint16_t)> w155;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(3840, uint16_t)> w156;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(32, uint16_t)> w157;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(3840, uint16_t)> w158;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(120, uint16_t)> w159;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(4800, uint16_t)> w160;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(40, uint16_t)> w161;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(9600, uint16_t)> w162;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(240, uint16_t)> w163;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(2160, uint16_t)> w164;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(240, uint16_t)> w165;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(19200, uint16_t)> w166;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(80, uint16_t)> w167;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(16000, uint16_t)> w168;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(200, uint16_t)> w169;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1800, uint16_t)> w170;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(200, uint16_t)> w171;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(16000, uint16_t)> w172;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(80, uint16_t)> w173;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(14720, uint16_t)> w174;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(184, uint16_t)> w175;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1656, uint16_t)> w176;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(184, uint16_t)> w177;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(14720, uint16_t)> w178;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(80, uint16_t)> w179;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(14720, uint16_t)> w180;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(184, uint16_t)> w181;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1656, uint16_t)> w182;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(184, uint16_t)> w183;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(14720, uint16_t)> w184;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(80, uint16_t)> w185;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(38400, uint16_t)> w186;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(480, uint16_t)> w187;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(4320, uint16_t)> w188;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(480, uint16_t)> w189;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(57600, uint16_t)> w190;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(120, uint16_t)> w191;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(57600, uint16_t)> w192;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(480, uint16_t)> w193;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(53760, uint16_t)> w194;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(112, uint16_t)> w195;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(75264, uint16_t)> w196;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(672, uint16_t)> w197;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(6048, uint16_t)> w198;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(672, uint16_t)> w199;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(112896, uint16_t)> w200;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(168, uint16_t)> w201;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(112896, uint16_t)> w202;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(672, uint16_t)> w203;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(75264, uint16_t)> w204;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(112, uint16_t)> w205;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(75264, uint16_t)> w206;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(672, uint16_t)> w207;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(16800, uint16_t)> w208;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(672, uint16_t)> w209;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(112896, uint16_t)> w210;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(168, uint16_t)> w211;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(112896, uint16_t)> w212;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(672, uint16_t)> w213;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(107520, uint16_t)> w214;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(160, uint16_t)> w215;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(153600, uint16_t)> w216;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(960, uint16_t)> w217;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(24000, uint16_t)> w218;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(960, uint16_t)> w219;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(230400, uint16_t)> w220;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(240, uint16_t)> w221;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(230400, uint16_t)> w222;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(960, uint16_t)> w223;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(153600, uint16_t)> w224;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(160, uint16_t)> w225;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(153600, uint16_t)> w226;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(960, uint16_t)> w227;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(24000, uint16_t)> w228;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(960, uint16_t)> w229;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(230400, uint16_t)> w230;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(240, uint16_t)> w231;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(230400, uint16_t)> w232;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(960, uint16_t)> w233;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(153600, uint16_t)> w234;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(160, uint16_t)> w235;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(153600, uint16_t)> w236;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(960, uint16_t)> w237;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1228800, uint16_t)> w238;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1280, uint16_t)> w239;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1281280, uint16_t)> w240;
  alignas(16) static std::array<uint16_t, XNN_PAD_EXTRA_BYTES(1001, uint16_t)> w241;

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);
  std::generate(v0.begin(), v0.end(), std::ref(f16rng));
  std::generate(v1.begin(), v1.end(), std::ref(f16rng));
  std::generate(v2.begin(), v2.end(), std::ref(f16rng));
  std::generate(v3.begin(), v3.end(), std::ref(f16rng));
  std::generate(v4.begin(), v4.end(), std::ref(f16rng));
  std::generate(v5.begin(), v5.end(), std::ref(f16rng));
  std::generate(v6.begin(), v6.end(), std::ref(f16rng));
  std::generate(v7.begin(), v7.end(), std::ref(f16rng));
  std::generate(v8.begin(), v8.end(), std::ref(f16rng));
  std::generate(v9.begin(), v9.end(), std::ref(f16rng));
  std::generate(v10.begin(), v10.end(), std::ref(f16rng));
  std::generate(v11.begin(), v11.end(), std::ref(f16rng));
  std::generate(v12.begin(), v12.end(), std::ref(f16rng));
  std::generate(v13.begin(), v13.end(), std::ref(f16rng));
  std::generate(v14.begin(), v14.end(), std::ref(f16rng));
  std::generate(v15.begin(), v15.end(), std::ref(f16rng));
  std::generate(v16.begin(), v16.end(), std::ref(f16rng));
  std::generate(v17.begin(), v17.end(), std::ref(f16rng));
  std::generate(v18.begin(), v18.end(), std::ref(f16rng));
  std::generate(v19.begin(), v19.end(), std::ref(f16rng));
  std::generate(v20.begin(), v20.end(), std::ref(f16rng));
  std::generate(v21.begin(), v21.end(), std::ref(f16rng));
  std::generate(v22.begin(), v22.end(), std::ref(f16rng));
  std::generate(v23.begin(), v23.end(), std::ref(f16rng));
  std::generate(v24.begin(), v24.end(), std::ref(f16rng));
  std::generate(v25.begin(), v25.end(), std::ref(f16rng));
  std::generate(v26.begin(), v26.end(), std::ref(f16rng));
  std::generate(v27.begin(), v27.end(), std::ref(f16rng));
  std::generate(v28.begin(), v28.end(), std::ref(f16rng));
  std::generate(v29.begin(), v29.end(), std::ref(f16rng));
  std::generate(v30.begin(), v30.end(), std::ref(f16rng));
  std::generate(v31.begin(), v31.end(), std::ref(f16rng));
  std::generate(v32.begin(), v32.end(), std::ref(f16rng));
  std::generate(v33.begin(), v33.end(), std::ref(f16rng));
  std::generate(v34.begin(), v34.end(), std::ref(f16rng));
  std::generate(v35.begin(), v35.end(), std::ref(f16rng));
  std::generate(v36.begin(), v36.end(), std::ref(f16rng));
  std::generate(v37.begin(), v37.end(), std::ref(f16rng));
  std::generate(v38.begin(), v38.end(), std::ref(f16rng));
  std::generate(v39.begin(), v39.end(), std::ref(f16rng));
  std::generate(v40.begin(), v40.end(), std::ref(f16rng));
  std::generate(v41.begin(), v41.end(), std::ref(f16rng));
  std::generate(v42.begin(), v42.end(), std::ref(f16rng));
  std::generate(v43.begin(), v43.end(), std::ref(f16rng));
  std::generate(v44.begin(), v44.end(), std::ref(f16rng));
  std::generate(v45.begin(), v45.end(), std::ref(f16rng));
  std::generate(v46.begin(), v46.end(), std::ref(f16rng));
  std::generate(v47.begin(), v47.end(), std::ref(f16rng));
  std::generate(v48.begin(), v48.end(), std::ref(f16rng));
  std::generate(v49.begin(), v49.end(), std::ref(f16rng));
  std::generate(v50.begin(), v50.end(), std::ref(f16rng));
  std::generate(v51.begin(), v51.end(), std::ref(f16rng));
  std::generate(v52.begin(), v52.end(), std::ref(f16rng));
  std::generate(v53.begin(), v53.end(), std::ref(f16rng));
  std::generate(v54.begin(), v54.end(), std::ref(f16rng));
  std::generate(v55.begin(), v55.end(), std::ref(f16rng));
  std::generate(v56.begin(), v56.end(), std::ref(f16rng));
  std::generate(v57.begin(), v57.end(), std::ref(f16rng));
  std::generate(v58.begin(), v58.end(), std::ref(f16rng));
  std::generate(v59.begin(), v59.end(), std::ref(f16rng));
  std::generate(v60.begin(), v60.end(), std::ref(f16rng));
  std::generate(v61.begin(), v61.end(), std::ref(f16rng));
  std::generate(v62.begin(), v62.end(), std::ref(f16rng));
  std::generate(v63.begin(), v63.end(), std::ref(f16rng));
  std::generate(v64.begin(), v64.end(), std::ref(f16rng));
  std::generate(v65.begin(), v65.end(), std::ref(f16rng));
  std::generate(v66.begin(), v66.end(), std::ref(f16rng));
  std::generate(v67.begin(), v67.end(), std::ref(f16rng));
  std::generate(v68.begin(), v68.end(), std::ref(f16rng));
  std::generate(v69.begin(), v69.end(), std::ref(f16rng));
  std::generate(v70.begin(), v70.end(), std::ref(f16rng));
  std::generate(v71.begin(), v71.end(), std::ref(f16rng));
  std::generate(v72.begin(), v72.end(), std::ref(f16rng));
  std::generate(v73.begin(), v73.end(), std::ref(f16rng));
  std::generate(v74.begin(), v74.end(), std::ref(f16rng));
  std::generate(v75.begin(), v75.end(), std::ref(f16rng));
  std::generate(v76.begin(), v76.end(), std::ref(f16rng));
  std::generate(v77.begin(), v77.end(), std::ref(f16rng));
  std::generate(v78.begin(), v78.end(), std::ref(f16rng));
  std::generate(v79.begin(), v79.end(), std::ref(f16rng));
  std::generate(v80.begin(), v80.end(), std::ref(f16rng));
  std::generate(v81.begin(), v81.end(), std::ref(f16rng));
  std::generate(v82.begin(), v82.end(), std::ref(f16rng));
  std::generate(v83.begin(), v83.end(), std::ref(f16rng));
  std::generate(v84.begin(), v84.end(), std::ref(f16rng));
  std::generate(v85.begin(), v85.end(), std::ref(f16rng));
  std::generate(v86.begin(), v86.end(), std::ref(f16rng));
  std::generate(v87.begin(), v87.end(), std::ref(f16rng));
  std::generate(v88.begin(), v88.end(), std::ref(f16rng));
  std::generate(v89.begin(), v89.end(), std::ref(f16rng));
  std::generate(v90.begin(), v90.end(), std::ref(f16rng));
  std::generate(v91.begin(), v91.end(), std::ref(f16rng));
  std::generate(v92.begin(), v92.end(), std::ref(f16rng));
  std::generate(v93.begin(), v93.end(), std::ref(f16rng));
  std::generate(v94.begin(), v94.end(), std::ref(f16rng));
  std::generate(v95.begin(), v95.end(), std::ref(f16rng));
  std::generate(v96.begin(), v96.end(), std::ref(f16rng));
  std::generate(v97.begin(), v97.end(), std::ref(f16rng));
  std::generate(v98.begin(), v98.end(), std::ref(f16rng));
  std::generate(v99.begin(), v99.end(), std::ref(f16rng));
  std::generate(v100.begin(), v100.end(), std::ref(f16rng));
  std::generate(v101.begin(), v101.end(), std::ref(f16rng));
  std::generate(v102.begin(), v102.end(), std::ref(f16rng));
  std::generate(v103.begin(), v103.end(), std::ref(f16rng));
  std::generate(v104.begin(), v104.end(), std::ref(f16rng));
  std::generate(v105.begin(), v105.end(), std::ref(f16rng));
  std::generate(v106.begin(), v106.end(), std::ref(f16rng));
  std::generate(v107.begin(), v107.end(), std::ref(f16rng));
  std::generate(v108.begin(), v108.end(), std::ref(f16rng));
  std::generate(v109.begin(), v109.end(), std::ref(f16rng));
  std::generate(v110.begin(), v110.end(), std::ref(f16rng));
  std::generate(v111.begin(), v111.end(), std::ref(f16rng));
  std::generate(v112.begin(), v112.end(), std::ref(f16rng));
  std::generate(v113.begin(), v113.end(), std::ref(f16rng));
  std::generate(w114.begin(), w114.end(), std::ref(f16rng));
  std::generate(w115.begin(), w115.end(), std::ref(f16rng));
  std::generate(w116.begin(), w116.end(), std::ref(f16rng));
  std::generate(w117.begin(), w117.end(), std::ref(f16rng));
  std::fill(w118.begin(), w118.end(), 0);
  std::generate(w118.begin(), w118.end() - size_t(sparsity * w118.size()), std::ref(f16rng));
  std::shuffle(w118.begin(), w118.end(), rng);
  std::generate(w119.begin(), w119.end(), std::ref(f16rng));
  std::fill(w120.begin(), w120.end(), 0);
  std::generate(w120.begin(), w120.end() - size_t(sparsity * w120.size()), std::ref(f16rng));
  std::shuffle(w120.begin(), w120.end(), rng);
  std::generate(w121.begin(), w121.end(), std::ref(f16rng));
  std::generate(w122.begin(), w122.end(), std::ref(f16rng));
  std::generate(w123.begin(), w123.end(), std::ref(f16rng));
  std::fill(w124.begin(), w124.end(), 0);
  std::generate(w124.begin(), w124.end() - size_t(sparsity * w124.size()), std::ref(f16rng));
  std::shuffle(w124.begin(), w124.end(), rng);
  std::generate(w125.begin(), w125.end(), std::ref(f16rng));
  std::fill(w126.begin(), w126.end(), 0);
  std::generate(w126.begin(), w126.end() - size_t(sparsity * w126.size()), std::ref(f16rng));
  std::shuffle(w126.begin(), w126.end(), rng);
  std::generate(w127.begin(), w127.end(), std::ref(f16rng));
  std::generate(w128.begin(), w128.end(), std::ref(f16rng));
  std::generate(w129.begin(), w129.end(), std::ref(f16rng));
  std::fill(w130.begin(), w130.end(), 0);
  std::generate(w130.begin(), w130.end() - size_t(sparsity * w130.size()), std::ref(f16rng));
  std::shuffle(w130.begin(), w130.end(), rng);
  std::generate(w131.begin(), w131.end(), std::ref(f16rng));
  std::fill(w132.begin(), w132.end(), 0);
  std::generate(w132.begin(), w132.end() - size_t(sparsity * w132.size()), std::ref(f16rng));
  std::shuffle(w132.begin(), w132.end(), rng);
  std::generate(w133.begin(), w133.end(), std::ref(f16rng));
  std::generate(w134.begin(), w134.end(), std::ref(f16rng));
  std::generate(w135.begin(), w135.end(), std::ref(f16rng));
  std::fill(w136.begin(), w136.end(), 0);
  std::generate(w136.begin(), w136.end() - size_t(sparsity * w136.size()), std::ref(f16rng));
  std::shuffle(w136.begin(), w136.end(), rng);
  std::generate(w137.begin(), w137.end(), std::ref(f16rng));
  std::fill(w138.begin(), w138.end(), 0);
  std::generate(w138.begin(), w138.end() - size_t(sparsity * w138.size()), std::ref(f16rng));
  std::shuffle(w138.begin(), w138.end(), rng);
  std::generate(w139.begin(), w139.end(), std::ref(f16rng));
  std::fill(w140.begin(), w140.end(), 0);
  std::generate(w140.begin(), w140.end() - size_t(sparsity * w140.size()), std::ref(f16rng));
  std::shuffle(w140.begin(), w140.end(), rng);
  std::generate(w141.begin(), w141.end(), std::ref(f16rng));
  std::fill(w142.begin(), w142.end(), 0);
  std::generate(w142.begin(), w142.end() - size_t(sparsity * w142.size()), std::ref(f16rng));
  std::shuffle(w142.begin(), w142.end(), rng);
  std::generate(w143.begin(), w143.end(), std::ref(f16rng));
  std::generate(w144.begin(), w144.end(), std::ref(f16rng));
  std::generate(w145.begin(), w145.end(), std::ref(f16rng));
  std::fill(w146.begin(), w146.end(), 0);
  std::generate(w146.begin(), w146.end() - size_t(sparsity * w146.size()), std::ref(f16rng));
  std::shuffle(w146.begin(), w146.end(), rng);
  std::generate(w147.begin(), w147.end(), std::ref(f16rng));
  std::fill(w148.begin(), w148.end(), 0);
  std::generate(w148.begin(), w148.end() - size_t(sparsity * w148.size()), std::ref(f16rng));
  std::shuffle(w148.begin(), w148.end(), rng);
  std::generate(w149.begin(), w149.end(), std::ref(f16rng));
  std::fill(w150.begin(), w150.end(), 0);
  std::generate(w150.begin(), w150.end() - size_t(sparsity * w150.size()), std::ref(f16rng));
  std::shuffle(w150.begin(), w150.end(), rng);
  std::generate(w151.begin(), w151.end(), std::ref(f16rng));
  std::fill(w152.begin(), w152.end(), 0);
  std::generate(w152.begin(), w152.end() - size_t(sparsity * w152.size()), std::ref(f16rng));
  std::shuffle(w152.begin(), w152.end(), rng);
  std::generate(w153.begin(), w153.end(), std::ref(f16rng));
  std::generate(w154.begin(), w154.end(), std::ref(f16rng));
  std::generate(w155.begin(), w155.end(), std::ref(f16rng));
  std::fill(w156.begin(), w156.end(), 0);
  std::generate(w156.begin(), w156.end() - size_t(sparsity * w156.size()), std::ref(f16rng));
  std::shuffle(w156.begin(), w156.end(), rng);
  std::generate(w157.begin(), w157.end(), std::ref(f16rng));
  std::fill(w158.begin(), w158.end(), 0);
  std::generate(w158.begin(), w158.end() - size_t(sparsity * w158.size()), std::ref(f16rng));
  std::shuffle(w158.begin(), w158.end(), rng);
  std::generate(w159.begin(), w159.end(), std::ref(f16rng));
  std::fill(w160.begin(), w160.end(), 0);
  std::generate(w160.begin(), w160.end() - size_t(sparsity * w160.size()), std::ref(f16rng));
  std::shuffle(w160.begin(), w160.end(), rng);
  std::generate(w161.begin(), w161.end(), std::ref(f16rng));
  std::fill(w162.begin(), w162.end(), 0);
  std::generate(w162.begin(), w162.end() - size_t(sparsity * w162.size()), std::ref(f16rng));
  std::shuffle(w162.begin(), w162.end(), rng);
  std::generate(w163.begin(), w163.end(), std::ref(f16rng));
  std::generate(w164.begin(), w164.end(), std::ref(f16rng));
  std::generate(w165.begin(), w165.end(), std::ref(f16rng));
  std::fill(w166.begin(), w166.end(), 0);
  std::generate(w166.begin(), w166.end() - size_t(sparsity * w166.size()), std::ref(f16rng));
  std::shuffle(w166.begin(), w166.end(), rng);
  std::generate(w167.begin(), w167.end(), std::ref(f16rng));
  std::fill(w168.begin(), w168.end(), 0);
  std::generate(w168.begin(), w168.end() - size_t(sparsity * w168.size()), std::ref(f16rng));
  std::shuffle(w168.begin(), w168.end(), rng);
  std::generate(w169.begin(), w169.end(), std::ref(f16rng));
  std::generate(w170.begin(), w170.end(), std::ref(f16rng));
  std::generate(w171.begin(), w171.end(), std::ref(f16rng));
  std::fill(w172.begin(), w172.end(), 0);
  std::generate(w172.begin(), w172.end() - size_t(sparsity * w172.size()), std::ref(f16rng));
  std::shuffle(w172.begin(), w172.end(), rng);
  std::generate(w173.begin(), w173.end(), std::ref(f16rng));
  std::fill(w174.begin(), w174.end(), 0);
  std::generate(w174.begin(), w174.end() - size_t(sparsity * w174.size()), std::ref(f16rng));
  std::shuffle(w174.begin(), w174.end(), rng);
  std::generate(w175.begin(), w175.end(), std::ref(f16rng));
  std::generate(w176.begin(), w176.end(), std::ref(f16rng));
  std::generate(w177.begin(), w177.end(), std::ref(f16rng));
  std::fill(w178.begin(), w178.end(), 0);
  std::generate(w178.begin(), w178.end() - size_t(sparsity * w178.size()), std::ref(f16rng));
  std::shuffle(w178.begin(), w178.end(), rng);
  std::generate(w179.begin(), w179.end(), std::ref(f16rng));
  std::fill(w180.begin(), w180.end(), 0);
  std::generate(w180.begin(), w180.end() - size_t(sparsity * w180.size()), std::ref(f16rng));
  std::shuffle(w180.begin(), w180.end(), rng);
  std::generate(w181.begin(), w181.end(), std::ref(f16rng));
  std::generate(w182.begin(), w182.end(), std::ref(f16rng));
  std::generate(w183.begin(), w183.end(), std::ref(f16rng));
  std::fill(w184.begin(), w184.end(), 0);
  std::generate(w184.begin(), w184.end() - size_t(sparsity * w184.size()), std::ref(f16rng));
  std::shuffle(w184.begin(), w184.end(), rng);
  std::generate(w185.begin(), w185.end(), std::ref(f16rng));
  std::fill(w186.begin(), w186.end(), 0);
  std::generate(w186.begin(), w186.end() - size_t(sparsity * w186.size()), std::ref(f16rng));
  std::shuffle(w186.begin(), w186.end(), rng);
  std::generate(w187.begin(), w187.end(), std::ref(f16rng));
  std::generate(w188.begin(), w188.end(), std::ref(f16rng));
  std::generate(w189.begin(), w189.end(), std::ref(f16rng));
  std::fill(w190.begin(), w190.end(), 0);
  std::generate(w190.begin(), w190.end() - size_t(sparsity * w190.size()), std::ref(f16rng));
  std::shuffle(w190.begin(), w190.end(), rng);
  std::generate(w191.begin(), w191.end(), std::ref(f16rng));
  std::fill(w192.begin(), w192.end(), 0);
  std::generate(w192.begin(), w192.end() - size_t(sparsity * w192.size()), std::ref(f16rng));
  std::shuffle(w192.begin(), w192.end(), rng);
  std::generate(w193.begin(), w193.end(), std::ref(f16rng));
  std::fill(w194.begin(), w194.end(), 0);
  std::generate(w194.begin(), w194.end() - size_t(sparsity * w194.size()), std::ref(f16rng));
  std::shuffle(w194.begin(), w194.end(), rng);
  std::generate(w195.begin(), w195.end(), std::ref(f16rng));
  std::fill(w196.begin(), w196.end(), 0);
  std::generate(w196.begin(), w196.end() - size_t(sparsity * w196.size()), std::ref(f16rng));
  std::shuffle(w196.begin(), w196.end(), rng);
  std::generate(w197.begin(), w197.end(), std::ref(f16rng));
  std::generate(w198.begin(), w198.end(), std::ref(f16rng));
  std::generate(w199.begin(), w199.end(), std::ref(f16rng));
  std::fill(w200.begin(), w200.end(), 0);
  std::generate(w200.begin(), w200.end() - size_t(sparsity * w200.size()), std::ref(f16rng));
  std::shuffle(w200.begin(), w200.end(), rng);
  std::generate(w201.begin(), w201.end(), std::ref(f16rng));
  std::fill(w202.begin(), w202.end(), 0);
  std::generate(w202.begin(), w202.end() - size_t(sparsity * w202.size()), std::ref(f16rng));
  std::shuffle(w202.begin(), w202.end(), rng);
  std::generate(w203.begin(), w203.end(), std::ref(f16rng));
  std::fill(w204.begin(), w204.end(), 0);
  std::generate(w204.begin(), w204.end() - size_t(sparsity * w204.size()), std::ref(f16rng));
  std::shuffle(w204.begin(), w204.end(), rng);
  std::generate(w205.begin(), w205.end(), std::ref(f16rng));
  std::fill(w206.begin(), w206.end(), 0);
  std::generate(w206.begin(), w206.end() - size_t(sparsity * w206.size()), std::ref(f16rng));
  std::shuffle(w206.begin(), w206.end(), rng);
  std::generate(w207.begin(), w207.end(), std::ref(f16rng));
  std::generate(w208.begin(), w208.end(), std::ref(f16rng));
  std::generate(w209.begin(), w209.end(), std::ref(f16rng));
  std::fill(w210.begin(), w210.end(), 0);
  std::generate(w210.begin(), w210.end() - size_t(sparsity * w210.size()), std::ref(f16rng));
  std::shuffle(w210.begin(), w210.end(), rng);
  std::generate(w211.begin(), w211.end(), std::ref(f16rng));
  std::fill(w212.begin(), w212.end(), 0);
  std::generate(w212.begin(), w212.end() - size_t(sparsity * w212.size()), std::ref(f16rng));
  std::shuffle(w212.begin(), w212.end(), rng);
  std::generate(w213.begin(), w213.end(), std::ref(f16rng));
  std::fill(w214.begin(), w214.end(), 0);
  std::generate(w214.begin(), w214.end() - size_t(sparsity * w214.size()), std::ref(f16rng));
  std::shuffle(w214.begin(), w214.end(), rng);
  std::generate(w215.begin(), w215.end(), std::ref(f16rng));
  std::fill(w216.begin(), w216.end(), 0);
  std::generate(w216.begin(), w216.end() - size_t(sparsity * w216.size()), std::ref(f16rng));
  std::shuffle(w216.begin(), w216.end(), rng);
  std::generate(w217.begin(), w217.end(), std::ref(f16rng));
  std::generate(w218.begin(), w218.end(), std::ref(f16rng));
  std::generate(w219.begin(), w219.end(), std::ref(f16rng));
  std::fill(w220.begin(), w220.end(), 0);
  std::generate(w220.begin(), w220.end() - size_t(sparsity * w220.size()), std::ref(f16rng));
  std::shuffle(w220.begin(), w220.end(), rng);
  std::generate(w221.begin(), w221.end(), std::ref(f16rng));
  std::fill(w222.begin(), w222.end(), 0);
  std::generate(w222.begin(), w222.end() - size_t(sparsity * w222.size()), std::ref(f16rng));
  std::shuffle(w222.begin(), w222.end(), rng);
  std::generate(w223.begin(), w223.end(), std::ref(f16rng));
  std::fill(w224.begin(), w224.end(), 0);
  std::generate(w224.begin(), w224.end() - size_t(sparsity * w224.size()), std::ref(f16rng));
  std::shuffle(w224.begin(), w224.end(), rng);
  std::generate(w225.begin(), w225.end(), std::ref(f16rng));
  std::fill(w226.begin(), w226.end(), 0);
  std::generate(w226.begin(), w226.end() - size_t(sparsity * w226.size()), std::ref(f16rng));
  std::shuffle(w226.begin(), w226.end(), rng);
  std::generate(w227.begin(), w227.end(), std::ref(f16rng));
  std::generate(w228.begin(), w228.end(), std::ref(f16rng));
  std::generate(w229.begin(), w229.end(), std::ref(f16rng));
  std::fill(w230.begin(), w230.end(), 0);
  std::generate(w230.begin(), w230.end() - size_t(sparsity * w230.size()), std::ref(f16rng));
  std::shuffle(w230.begin(), w230.end(), rng);
  std::generate(w231.begin(), w231.end(), std::ref(f16rng));
  std::fill(w232.begin(), w232.end(), 0);
  std::generate(w232.begin(), w232.end() - size_t(sparsity * w232.size()), std::ref(f16rng));
  std::shuffle(w232.begin(), w232.end(), rng);
  std::generate(w233.begin(), w233.end(), std::ref(f16rng));
  std::fill(w234.begin(), w234.end(), 0);
  std::generate(w234.begin(), w234.end() - size_t(sparsity * w234.size()), std::ref(f16rng));
  std::shuffle(w234.begin(), w234.end(), rng);
  std::generate(w235.begin(), w235.end(), std::ref(f16rng));
  std::fill(w236.begin(), w236.end(), 0);
  std::generate(w236.begin(), w236.end() - size_t(sparsity * w236.size()), std::ref(f16rng));
  std::shuffle(w236.begin(), w236.end(), rng);
  std::generate(w237.begin(), w237.end(), std::ref(f16rng));
  std::generate(w238.begin(), w238.end(), std::ref(f16rng));
  std::generate(w239.begin(), w239.end(), std::ref(f16rng));
  std::generate(w240.begin(), w240.end(), std::ref(f16rng));
  std::generate(w241.begin(), w241.end(), std::ref(f16rng));

  Operators operators;
  xnn_status status;
  size_t max_workspace_size = 0;

  xnn_operator_t op0 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
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
    XNN_FLAG_INPUT_NHWC /* flags */,
    nullptr,
    nullptr,
    &op0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #0" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op0, xnn_delete_operator);

  xnn_operator_t op1 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #1" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op1, xnn_delete_operator);

  xnn_operator_t op2 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #2" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op2, xnn_delete_operator);

  xnn_operator_t op3 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #3" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op3, xnn_delete_operator);

  xnn_operator_t op4 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op4);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #4" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op4, xnn_delete_operator);

  xnn_operator_t op5 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #5" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op5, xnn_delete_operator);

  xnn_operator_t op6 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
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
    nullptr,
    nullptr,
    &op6);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #6" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op6, xnn_delete_operator);

  xnn_operator_t op7 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op7);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #7" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op7, xnn_delete_operator);

  xnn_operator_t op8 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #8" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op8, xnn_delete_operator);

  xnn_operator_t op9 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #9" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op9, xnn_delete_operator);

  xnn_operator_t op10 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #10" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op10, xnn_delete_operator);

  xnn_operator_t op11 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #11" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op11, xnn_delete_operator);

  xnn_operator_t op12 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #12" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op12, xnn_delete_operator);

  xnn_operator_t op13 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
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
    nullptr,
    nullptr,
    &op13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #13" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op13, xnn_delete_operator);

  xnn_operator_t op14 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f16(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op14);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #14" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op14, xnn_delete_operator);

  xnn_operator_t op15 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #15" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op15, xnn_delete_operator);

  xnn_operator_t op16 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #16" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op16, xnn_delete_operator);

  xnn_operator_t op17 = nullptr;
  status = xnn_create_multiply_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #17" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op17, xnn_delete_operator);

  xnn_operator_t op18 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op18);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #18" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op18, xnn_delete_operator);

  xnn_operator_t op19 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #19" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op19, xnn_delete_operator);

  xnn_operator_t op20 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op20);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #20" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op20, xnn_delete_operator);

  xnn_operator_t op21 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f16(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op21);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #21" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op21, xnn_delete_operator);

  xnn_operator_t op22 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op22);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #22" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op22, xnn_delete_operator);

  xnn_operator_t op23 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op23);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #23" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op23, xnn_delete_operator);

  xnn_operator_t op24 = nullptr;
  status = xnn_create_multiply_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #24" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op24, xnn_delete_operator);

  xnn_operator_t op25 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #25" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op25, xnn_delete_operator);

  xnn_operator_t op26 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #26" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op26, xnn_delete_operator);

  xnn_operator_t op27 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #27" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op27, xnn_delete_operator);

  xnn_operator_t op28 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #28" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op28, xnn_delete_operator);

  xnn_operator_t op29 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f16(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op29);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #29" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op29, xnn_delete_operator);

  xnn_operator_t op30 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op30);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #30" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op30, xnn_delete_operator);

  xnn_operator_t op31 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op31);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #31" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op31, xnn_delete_operator);

  xnn_operator_t op32 = nullptr;
  status = xnn_create_multiply_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #32" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op32, xnn_delete_operator);

  xnn_operator_t op33 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op33);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #33" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op33, xnn_delete_operator);

  xnn_operator_t op34 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op34);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #34" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op34, xnn_delete_operator);

  xnn_operator_t op35 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op35);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #35" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op35, xnn_delete_operator);

  xnn_operator_t op36 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op36);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #36" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op36, xnn_delete_operator);

  xnn_operator_t op37 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
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
    nullptr,
    nullptr,
    &op37);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #37" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op37, xnn_delete_operator);

  xnn_operator_t op38 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op38);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #38" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op38, xnn_delete_operator);

  xnn_operator_t op39 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op39);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #39" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op39, xnn_delete_operator);

  xnn_operator_t op40 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op40);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #40" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op40, xnn_delete_operator);

  xnn_operator_t op41 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op41);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #41" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op41, xnn_delete_operator);

  xnn_operator_t op42 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op42);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #42" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op42, xnn_delete_operator);

  xnn_operator_t op43 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op43);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #43" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op43, xnn_delete_operator);

  xnn_operator_t op44 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op44);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #44" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op44, xnn_delete_operator);

  xnn_operator_t op45 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op45);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #45" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op45, xnn_delete_operator);

  xnn_operator_t op46 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op46);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #46" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op46, xnn_delete_operator);

  xnn_operator_t op47 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op47);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #47" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op47, xnn_delete_operator);

  xnn_operator_t op48 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op48);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #48" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op48, xnn_delete_operator);

  xnn_operator_t op49 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op49);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #49" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op49, xnn_delete_operator);

  xnn_operator_t op50 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op50);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #50" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op50, xnn_delete_operator);

  xnn_operator_t op51 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op51);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #51" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op51, xnn_delete_operator);

  xnn_operator_t op52 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op52);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #52" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op52, xnn_delete_operator);

  xnn_operator_t op53 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op53);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #53" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op53, xnn_delete_operator);

  xnn_operator_t op54 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op54);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #54" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op54, xnn_delete_operator);

  xnn_operator_t op55 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op55);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #55" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op55, xnn_delete_operator);

  xnn_operator_t op56 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
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
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op58);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #58" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op58, xnn_delete_operator);

  xnn_operator_t op59 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op59);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #59" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op59, xnn_delete_operator);

  xnn_operator_t op60 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op60);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #60" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op60, xnn_delete_operator);

  xnn_operator_t op61 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op61);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #61" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op61, xnn_delete_operator);

  xnn_operator_t op62 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f16(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op62);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #62" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op62, xnn_delete_operator);

  xnn_operator_t op63 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op63);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #63" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op63, xnn_delete_operator);

  xnn_operator_t op64 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op64);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #64" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op64, xnn_delete_operator);

  xnn_operator_t op65 = nullptr;
  status = xnn_create_multiply_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op65);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #65" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op65, xnn_delete_operator);

  xnn_operator_t op66 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op66);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #66" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op66, xnn_delete_operator);

  xnn_operator_t op67 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op67);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #67" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op67, xnn_delete_operator);

  xnn_operator_t op68 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op68);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #68" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op68, xnn_delete_operator);

  xnn_operator_t op69 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op69);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #69" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op69, xnn_delete_operator);

  xnn_operator_t op70 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op70);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #70" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op70, xnn_delete_operator);

  xnn_operator_t op71 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f16(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op71);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #71" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op71, xnn_delete_operator);

  xnn_operator_t op72 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op72);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #72" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op72, xnn_delete_operator);

  xnn_operator_t op73 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op73);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #73" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op73, xnn_delete_operator);

  xnn_operator_t op74 = nullptr;
  status = xnn_create_multiply_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op74);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #74" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op74, xnn_delete_operator);

  xnn_operator_t op75 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op75);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #75" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op75, xnn_delete_operator);

  xnn_operator_t op76 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op76);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #76" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op76, xnn_delete_operator);

  xnn_operator_t op77 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op77);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #77" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op77, xnn_delete_operator);

  xnn_operator_t op78 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op78);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #78" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op78, xnn_delete_operator);

  xnn_operator_t op79 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
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
    nullptr,
    nullptr,
    &op79);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #79" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op79, xnn_delete_operator);

  xnn_operator_t op80 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op80);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #80" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op80, xnn_delete_operator);

  xnn_operator_t op81 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f16(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op81);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #81" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op81, xnn_delete_operator);

  xnn_operator_t op82 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op82);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #82" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op82, xnn_delete_operator);

  xnn_operator_t op83 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op83);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #83" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op83, xnn_delete_operator);

  xnn_operator_t op84 = nullptr;
  status = xnn_create_multiply_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op84);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #84" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op84, xnn_delete_operator);

  xnn_operator_t op85 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op85);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #85" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op85, xnn_delete_operator);

  xnn_operator_t op86 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op86);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #86" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op86, xnn_delete_operator);

  xnn_operator_t op87 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op87);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #87" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op87, xnn_delete_operator);

  xnn_operator_t op88 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op88);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #88" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op88, xnn_delete_operator);

  xnn_operator_t op89 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op89);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #89" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op89, xnn_delete_operator);

  xnn_operator_t op90 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f16(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op90);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #90" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op90, xnn_delete_operator);

  xnn_operator_t op91 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op91);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #91" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op91, xnn_delete_operator);

  xnn_operator_t op92 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op92);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #92" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op92, xnn_delete_operator);

  xnn_operator_t op93 = nullptr;
  status = xnn_create_multiply_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op93);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #93" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op93, xnn_delete_operator);

  xnn_operator_t op94 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op94);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #94" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op94, xnn_delete_operator);

  xnn_operator_t op95 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op95);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #95" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op95, xnn_delete_operator);

  xnn_operator_t op96 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op96);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #96" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op96, xnn_delete_operator);

  xnn_operator_t op97 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op97);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #97" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op97, xnn_delete_operator);

  xnn_operator_t op98 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op98);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #98" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op98, xnn_delete_operator);

  xnn_operator_t op99 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op99);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #99" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op99, xnn_delete_operator);

  xnn_operator_t op100 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f16(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op100);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #100" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op100, xnn_delete_operator);

  xnn_operator_t op101 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op101);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #101" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op101, xnn_delete_operator);

  xnn_operator_t op102 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op102);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #102" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op102, xnn_delete_operator);

  xnn_operator_t op103 = nullptr;
  status = xnn_create_multiply_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op103);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #103" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op103, xnn_delete_operator);

  xnn_operator_t op104 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op104);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #104" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op104, xnn_delete_operator);

  xnn_operator_t op105 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op105);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #105" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op105, xnn_delete_operator);

  xnn_operator_t op106 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    nullptr,
    nullptr,
    &op106);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #106" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op106, xnn_delete_operator);

  xnn_operator_t op107 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op107);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #107" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op107, xnn_delete_operator);

  xnn_operator_t op108 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f16(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op108);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #108" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op108, xnn_delete_operator);

  xnn_operator_t op109 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
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
    nullptr,
    nullptr,
    &op109);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #109" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op109, xnn_delete_operator);

  xnn_operator_t op110 = nullptr;
  status = xnn_create_hardswish_nc_f16(
    0 /* flags */,
    &op110);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #110" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op110, xnn_delete_operator);

  xnn_operator_t op111 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f16(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op111);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #111" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op111, xnn_delete_operator);

  xnn_operator_t op112 = nullptr;
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
    w240.data(), w241.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op112);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #112" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op112, xnn_delete_operator);

  status = xnn_reshape_convolution2d_nchw_f16(
    op0,
    /*batch_size=*/1, /*input_height=*/224, /*input_width=*/224,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
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

  status = xnn_reshape_convolution2d_nchw_f16(
    op2,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op3,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #3" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 16, 112, 112 };
    const size_t b_shape[] = { 1, 16, 112, 112 };
    status = xnn_reshape_add_nd_f16(
      op4,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op5,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #5" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op6,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op7,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op8,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #8" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op9,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op10,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #10" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 24, 56, 56 };
    const size_t b_shape[] = { 1, 24, 56, 56 };
    status = xnn_reshape_add_nd_f16(
      op11,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op12,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op13,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #13" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_global_average_pooling_ncw_f16(
    op14,
    /*batch_size=*/1, 784 /* width */,
    72 /* channels */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op15,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #15" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op16,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #16" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 72, 28, 28 };
    const size_t b_shape[] = { 1, 72, 1, 1 };
    status = xnn_reshape_multiply_nd_f16(
      op17,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op18,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op19,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #19" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op20,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_global_average_pooling_ncw_f16(
    op21,
    /*batch_size=*/1, 784 /* width */,
    120 /* channels */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #21" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op22,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op23,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #23" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 120, 28, 28 };
    const size_t b_shape[] = { 1, 120, 1, 1 };
    status = xnn_reshape_multiply_nd_f16(
      op24,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op25,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #25" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 40, 28, 28 };
    const size_t b_shape[] = { 1, 40, 28, 28 };
    status = xnn_reshape_add_nd_f16(
      op26,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #26" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op27,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op28,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #28" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_global_average_pooling_ncw_f16(
    op29,
    /*batch_size=*/1, 784 /* width */,
    120 /* channels */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #29" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op30,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #30" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op31,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #31" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 120, 28, 28 };
    const size_t b_shape[] = { 1, 120, 1, 1 };
    status = xnn_reshape_multiply_nd_f16(
      op32,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #32" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op33,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #33" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 40, 28, 28 };
    const size_t b_shape[] = { 1, 40, 28, 28 };
    status = xnn_reshape_add_nd_f16(
      op34,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #34" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op35,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #35" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op36,
    /*batch_size=*/784,
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #36" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op37,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #37" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op38,
    /*batch_size=*/196,
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #38" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op39,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #39" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op40,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #40" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op41,
    /*batch_size=*/196,
    200 /* channels */,
    200 /* input stride */,
    200 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #41" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op42,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #42" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op43,
    /*batch_size=*/196,
    200 /* channels */,
    200 /* input stride */,
    200 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #43" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op44,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #44" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 80, 14, 14 };
    const size_t b_shape[] = { 1, 80, 14, 14 };
    status = xnn_reshape_add_nd_f16(
      op45,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #45" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op46,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #46" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op47,
    /*batch_size=*/196,
    184 /* channels */,
    184 /* input stride */,
    184 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #47" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op48,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #48" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op49,
    /*batch_size=*/196,
    184 /* channels */,
    184 /* input stride */,
    184 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #49" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op50,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #50" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 80, 14, 14 };
    const size_t b_shape[] = { 1, 80, 14, 14 };
    status = xnn_reshape_add_nd_f16(
      op51,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #51" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op52,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #52" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op53,
    /*batch_size=*/196,
    184 /* channels */,
    184 /* input stride */,
    184 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #53" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op54,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #54" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op55,
    /*batch_size=*/196,
    480 /* channels */,
    480 /* input stride */,
    480 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #55" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op56,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #56" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 80, 14, 14 };
    const size_t b_shape[] = { 1, 80, 14, 14 };
    status = xnn_reshape_add_nd_f16(
      op57,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #57" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op58,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #58" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op59,
    /*batch_size=*/196,
    480 /* channels */,
    480 /* input stride */,
    480 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #59" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op60,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #60" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op61,
    /*batch_size=*/196,
    672 /* channels */,
    672 /* input stride */,
    672 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #61" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_global_average_pooling_ncw_f16(
    op62,
    /*batch_size=*/1, 196 /* width */,
    480 /* channels */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #62" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op63,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #63" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op64,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #64" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 480, 14, 14 };
    const size_t b_shape[] = { 1, 480, 1, 1 };
    status = xnn_reshape_multiply_nd_f16(
      op65,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #65" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op66,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #66" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op67,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #67" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op68,
    /*batch_size=*/196,
    672 /* channels */,
    672 /* input stride */,
    672 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #68" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op69,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #69" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op70,
    /*batch_size=*/196,
    672 /* channels */,
    672 /* input stride */,
    672 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #70" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_global_average_pooling_ncw_f16(
    op71,
    /*batch_size=*/1, 196 /* width */,
    672 /* channels */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #71" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op72,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #72" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op73,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #73" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 672, 14, 14 };
    const size_t b_shape[] = { 1, 672, 1, 1 };
    status = xnn_reshape_multiply_nd_f16(
      op74,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #74" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op75,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #75" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 112, 14, 14 };
    const size_t b_shape[] = { 1, 112, 14, 14 };
    status = xnn_reshape_add_nd_f16(
      op76,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #76" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op77,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #77" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op78,
    /*batch_size=*/196,
    672 /* channels */,
    672 /* input stride */,
    672 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #78" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op79,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #79" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op80,
    /*batch_size=*/49,
    672 /* channels */,
    672 /* input stride */,
    672 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #80" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_global_average_pooling_ncw_f16(
    op81,
    /*batch_size=*/1, 49 /* width */,
    672 /* channels */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #81" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op82,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #82" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op83,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #83" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 672, 7, 7 };
    const size_t b_shape[] = { 1, 672, 1, 1 };
    status = xnn_reshape_multiply_nd_f16(
      op84,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #84" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op85,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #85" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op86,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #86" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op87,
    /*batch_size=*/49,
    960 /* channels */,
    960 /* input stride */,
    960 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #87" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op88,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #88" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op89,
    /*batch_size=*/49,
    960 /* channels */,
    960 /* input stride */,
    960 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #89" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_global_average_pooling_ncw_f16(
    op90,
    /*batch_size=*/1, 49 /* width */,
    960 /* channels */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #90" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op91,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #91" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op92,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #92" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 960, 7, 7 };
    const size_t b_shape[] = { 1, 960, 1, 1 };
    status = xnn_reshape_multiply_nd_f16(
      op93,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #93" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op94,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #94" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 160, 7, 7 };
    const size_t b_shape[] = { 1, 160, 7, 7 };
    status = xnn_reshape_add_nd_f16(
      op95,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #95" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op96,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #96" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op97,
    /*batch_size=*/49,
    960 /* channels */,
    960 /* input stride */,
    960 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #97" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op98,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #98" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op99,
    /*batch_size=*/49,
    960 /* channels */,
    960 /* input stride */,
    960 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #99" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_global_average_pooling_ncw_f16(
    op100,
    /*batch_size=*/1, 49 /* width */,
    960 /* channels */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #100" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op101,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #101" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op102,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #102" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 960, 7, 7 };
    const size_t b_shape[] = { 1, 960, 1, 1 };
    status = xnn_reshape_multiply_nd_f16(
      op103,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #103" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op104,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #104" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 160, 7, 7 };
    const size_t b_shape[] = { 1, 160, 7, 7 };
    status = xnn_reshape_add_nd_f16(
      op105,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #105" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op106,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #106" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op107,
    /*batch_size=*/49,
    960 /* channels */,
    960 /* input stride */,
    960 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #107" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_global_average_pooling_ncw_f16(
    op108,
    /*batch_size=*/1, 49 /* width */,
    960 /* channels */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #108" << std::endl;
    return ExecutionPlan();
  }

  size_t op109_workspace_size = 0;
  size_t op109_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f16(
    op109,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op109_workspace_size, &op109_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op109_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #109" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f16(
    op110,
    /*batch_size=*/1,
    1280 /* channels */,
    1280 /* input stride */,
    1280 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #110" << std::endl;
    return ExecutionPlan();
  }

  size_t op111_workspace_size = 0;
  size_t op111_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_f16(
    op111,
    /*batch_size=*/1, 1 /* width */,
    1280 /* channels */, 1280 /* input stride */, 1280 /* output stride */,
    &op111_workspace_size, &op111_workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #111" << std::endl;
    return ExecutionPlan();
  }

  size_t op112_workspace_size = 0;
  size_t op112_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f16(
    op112,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op112_workspace_size, &op112_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op112_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #112" << std::endl;
    return ExecutionPlan();
  }

  Workspace workspace(max_workspace_size);

  status = xnn_setup_convolution2d_nchw_f16(
    op0,
    /*input=*/v0.data(), /*output=*/v1.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op1,
    /*input=*/v1.data(), /*output=*/v2.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #1" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op2,
    /*input=*/v2.data(), /*output=*/v3.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op3,
    /*input=*/v3.data(), /*output=*/v4.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #3" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f16(
    op4,
    v4.data() /* a */, v2.data() /* b */, /*output=*/v5.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op5,
    /*input=*/v5.data(), /*output=*/v6.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #5" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op6,
    /*input=*/v6.data(), /*output=*/v7.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op7,
    /*input=*/v7.data(), /*output=*/v8.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op8,
    /*input=*/v8.data(), /*output=*/v9.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #8" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op9,
    /*input=*/v9.data(), /*output=*/v10.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op10,
    /*input=*/v10.data(), /*output=*/v11.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #10" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f16(
    op11,
    v11.data() /* a */, v8.data() /* b */, /*output=*/v12.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op12,
    /*input=*/v12.data(), /*output=*/v13.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op13,
    /*input=*/v13.data(), /*output=*/v14.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #13" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f16(
    op14,
    /*input=*/v14.data(), /*output=*/v15.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op15,
    /*input=*/v15.data(), /*output=*/v16.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #15" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op16,
    /*input=*/v16.data(), /*output=*/v17.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #16" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_f16(
    op17,
    v14.data() /* a */, v17.data() /* b */, /*output=*/v18.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op18,
    /*input=*/v18.data(), /*output=*/v19.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op19,
    /*input=*/v19.data(), /*output=*/v20.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #19" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op20,
    /*input=*/v20.data(), /*output=*/v21.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f16(
    op21,
    /*input=*/v21.data(), /*output=*/v22.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #21" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op22,
    /*input=*/v22.data(), /*output=*/v23.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op23,
    /*input=*/v23.data(), /*output=*/v24.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #23" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_f16(
    op24,
    v21.data() /* a */, v24.data() /* b */, /*output=*/v25.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op25,
    /*input=*/v25.data(), /*output=*/v26.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #25" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f16(
    op26,
    v26.data() /* a */, v19.data() /* b */, /*output=*/v27.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #26" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op27,
    /*input=*/v27.data(), /*output=*/v28.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op28,
    /*input=*/v28.data(), /*output=*/v29.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #28" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f16(
    op29,
    /*input=*/v29.data(), /*output=*/v30.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #29" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op30,
    /*input=*/v30.data(), /*output=*/v31.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #30" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op31,
    /*input=*/v31.data(), /*output=*/v32.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #31" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_f16(
    op32,
    v29.data() /* a */, v32.data() /* b */, /*output=*/v33.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #32" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op33,
    /*input=*/v33.data(), /*output=*/v34.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #33" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f16(
    op34,
    v34.data() /* a */, v27.data() /* b */, /*output=*/v35.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #34" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op35,
    /*input=*/v35.data(), /*output=*/v36.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #35" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op36,
    /*input=*/v36.data(), /*output=*/v37.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #36" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op37,
    /*input=*/v37.data(), /*output=*/v38.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #37" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op38,
    /*input=*/v38.data(), /*output=*/v39.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #38" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op39,
    /*input=*/v39.data(), /*output=*/v40.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #39" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op40,
    /*input=*/v40.data(), /*output=*/v41.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #40" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op41,
    /*input=*/v41.data(), /*output=*/v42.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #41" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op42,
    /*input=*/v42.data(), /*output=*/v43.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #42" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op43,
    /*input=*/v43.data(), /*output=*/v44.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #43" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op44,
    /*input=*/v44.data(), /*output=*/v45.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #44" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f16(
    op45,
    v45.data() /* a */, v40.data() /* b */, /*output=*/v46.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #45" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op46,
    /*input=*/v46.data(), /*output=*/v47.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #46" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op47,
    /*input=*/v47.data(), /*output=*/v48.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #47" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op48,
    /*input=*/v48.data(), /*output=*/v49.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #48" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op49,
    /*input=*/v49.data(), /*output=*/v50.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #49" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op50,
    /*input=*/v50.data(), /*output=*/v51.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #50" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f16(
    op51,
    v51.data() /* a */, v46.data() /* b */, /*output=*/v52.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #51" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op52,
    /*input=*/v52.data(), /*output=*/v53.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #52" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op53,
    /*input=*/v53.data(), /*output=*/v54.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #53" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op54,
    /*input=*/v54.data(), /*output=*/v55.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #54" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op55,
    /*input=*/v55.data(), /*output=*/v56.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #55" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op56,
    /*input=*/v56.data(), /*output=*/v57.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #56" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f16(
    op57,
    v57.data() /* a */, v52.data() /* b */, /*output=*/v58.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #57" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op58,
    /*input=*/v58.data(), /*output=*/v59.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #58" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op59,
    /*input=*/v59.data(), /*output=*/v60.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #59" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op60,
    /*input=*/v60.data(), /*output=*/v61.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #60" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op61,
    /*input=*/v61.data(), /*output=*/v62.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #61" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f16(
    op62,
    /*input=*/v62.data(), /*output=*/v63.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #62" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op63,
    /*input=*/v63.data(), /*output=*/v64.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #63" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op64,
    /*input=*/v64.data(), /*output=*/v65.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #64" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_f16(
    op65,
    v62.data() /* a */, v65.data() /* b */, /*output=*/v66.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #65" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op66,
    /*input=*/v66.data(), /*output=*/v67.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #66" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op67,
    /*input=*/v67.data(), /*output=*/v68.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #67" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op68,
    /*input=*/v68.data(), /*output=*/v69.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #68" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op69,
    /*input=*/v69.data(), /*output=*/v70.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #69" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op70,
    /*input=*/v70.data(), /*output=*/v71.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #70" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f16(
    op71,
    /*input=*/v71.data(), /*output=*/v72.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #71" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op72,
    /*input=*/v72.data(), /*output=*/v73.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #72" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op73,
    /*input=*/v73.data(), /*output=*/v74.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #73" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_f16(
    op74,
    v71.data() /* a */, v74.data() /* b */, /*output=*/v75.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #74" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op75,
    /*input=*/v75.data(), /*output=*/v76.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #75" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f16(
    op76,
    v76.data() /* a */, v67.data() /* b */, /*output=*/v77.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #76" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op77,
    /*input=*/v77.data(), /*output=*/v78.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #77" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op78,
    /*input=*/v78.data(), /*output=*/v79.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #78" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op79,
    /*input=*/v79.data(), /*output=*/v80.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #79" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op80,
    /*input=*/v80.data(), /*output=*/v81.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #80" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f16(
    op81,
    /*input=*/v81.data(), /*output=*/v82.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #81" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op82,
    /*input=*/v82.data(), /*output=*/v83.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #82" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op83,
    /*input=*/v83.data(), /*output=*/v84.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #83" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_f16(
    op84,
    v81.data() /* a */, v84.data() /* b */, /*output=*/v85.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #84" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op85,
    /*input=*/v85.data(), /*output=*/v86.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #85" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op86,
    /*input=*/v86.data(), /*output=*/v87.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #86" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op87,
    /*input=*/v87.data(), /*output=*/v88.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #87" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op88,
    /*input=*/v88.data(), /*output=*/v89.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #88" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op89,
    /*input=*/v89.data(), /*output=*/v90.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #89" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f16(
    op90,
    /*input=*/v90.data(), /*output=*/v91.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #90" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op91,
    /*input=*/v91.data(), /*output=*/v92.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #91" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op92,
    /*input=*/v92.data(), /*output=*/v93.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #92" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_f16(
    op93,
    v90.data() /* a */, v93.data() /* b */, /*output=*/v94.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #93" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op94,
    /*input=*/v94.data(), /*output=*/v95.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #94" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f16(
    op95,
    v95.data() /* a */, v86.data() /* b */, /*output=*/v96.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #95" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op96,
    /*input=*/v96.data(), /*output=*/v97.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #96" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op97,
    /*input=*/v97.data(), /*output=*/v98.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #97" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op98,
    /*input=*/v98.data(), /*output=*/v99.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #98" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op99,
    /*input=*/v99.data(), /*output=*/v100.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #99" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f16(
    op100,
    /*input=*/v100.data(), /*output=*/v101.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #100" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op101,
    /*input=*/v101.data(), /*output=*/v102.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #101" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op102,
    /*input=*/v102.data(), /*output=*/v103.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #102" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_f16(
    op103,
    v100.data() /* a */, v103.data() /* b */, /*output=*/v104.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #103" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op104,
    /*input=*/v104.data(), /*output=*/v105.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #104" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f16(
    op105,
    v105.data() /* a */, v96.data() /* b */, /*output=*/v106.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #105" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op106,
    /*input=*/v106.data(), /*output=*/v107.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #106" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op107,
    /*input=*/v107.data(), /*output=*/v108.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #107" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f16(
    op108,
    /*input=*/v108.data(), /*output=*/v109.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #108" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op109,
    workspace.data(),
    /*input=*/v109.data(), /*output=*/v110.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #109" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f16(
    op110,
    /*input=*/v110.data(), /*output=*/v111.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #110" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f16(
    op111,
    workspace.data(),
    /*input=*/v111.data(), /*output=*/v112.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #111" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op112,
    workspace.data(),
    /*input=*/v112.data(), /*output=*/v113.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #112" << std::endl;
    return ExecutionPlan();
  }

  XNN_PRAGMA_CLANG("clang diagnostic push")
  XNN_PRAGMA_CLANG("clang diagnostic ignored \"-Wpessimizing-move\"")
  return ExecutionPlan{operators, workspace};
  XNN_PRAGMA_CLANG("clang diagnostic pop")
}

}  // namespace models
