// Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

#define fpack_lhs_packed_in      x0
#define fpack_rhs_packed_in      x1
#define output                   x2
#define k_dimension              x3
#define max_value                x4
#define min_value                x5

.macro load_data, reg1, reg2, reg3, reg4, pred, addr
ld1w \reg1, \pred/Z, [\addr]
ld1w \reg2, \pred/Z, [\addr, #1, MUL VL]
ld1w \reg3, \pred/Z, [\addr, #2, MUL VL]
ld1w \reg4, \pred/Z, [\addr, #3, MUL VL]
addvl \addr, \addr, #4
.endm

.macro load_data_2, reg1, reg2, pred, addr
ld1w \reg1, \pred/Z, [\addr]
ld1w \reg2, \pred/Z, [\addr, #1, MUL VL]
addvl \addr, \addr, #2
.endm

.macro store_data, reg1, reg2, reg3, reg4, pred, addr
st1w \reg1, \pred, [\addr]
st1w \reg2, \pred, [\addr, #1, MUL VL]
st1w \reg3, \pred, [\addr, #2, MUL VL]
st1w \reg4, \pred, [\addr, #3, MUL VL]
addvl \addr, \addr, #4
.endm

.macro move_vector_tile, tile, reg1, reg2, reg3, reg4, pred, addr
mova \tile[\addr, 0], \pred/M, \reg1
mova \tile[\addr, 1], \pred/M, \reg2
mova \tile[\addr, 2], \pred/M, \reg3
mova \tile[\addr, 3], \pred/M, \reg4
.endm

.macro move_tile_vector, tile, reg1, reg2, reg3, reg4, pred, addr
mova \reg1, \pred/M, \tile[\addr, 0] 
mova \reg2, \pred/M, \tile[\addr, 1] 
mova \reg3, \pred/M, \tile[\addr, 2] 
mova \reg4, \pred/M, \tile[\addr, 3] 
.endm

.macro clamp_float reg, max_val, min_val, pred
FMIN \reg, \pred/M, \reg, \max_val
FMAX \reg, \pred/M, \reg, \min_val
.endm

.macro load_data_pred_index_counter, reg1, reg2, reg3, reg4, predr, predtype, start_idx, end_idx, addr, word_length
cntw \word_length
mov temp_start, \start_idx
whilelt \predtype, temp_start, \end_idx
ld1w \reg1, \predr/Z, [\addr]
add temp_start, temp_start, \word_length
whilelt \predtype, temp_start, \end_idx
ld1w \reg2, \predr/Z, [\addr, #1, MUL VL]
add temp_start, temp_start, \word_length
whilelt \predtype, temp_start, \end_idx
ld1w \reg3, \predr/Z, [\addr, #2, MUL VL]
add temp_start, temp_start, \word_length
whilelt \predtype, temp_start, \end_idx
ld1w \reg4, \predr/Z, [\addr, #3, MUL VL]
.endm

.macro store_data_pred_index_counter, reg1, reg2, reg3, reg4, predr, predtype, start_idx, end_idx, addr, word_length
cntw \word_length
mov temp_start, \start_idx
whilelt \predtype, temp_start, \end_idx
st1w \reg1, \predr, [\addr]
add temp_start, temp_start, \word_length
whilelt \predtype, temp_start, \end_idx
st1w \reg2, \predr, [\addr, #1, MUL VL]
add temp_start, temp_start, \word_length
whilelt \predtype, temp_start, \end_idx
st1w \reg3, \predr, [\addr, #2, MUL VL]
add temp_start, temp_start, \word_length
whilelt \predtype, temp_start, \end_idx
st1w \reg4, \predr, [\addr, #3, MUL VL]
.endm

.macro load_data_pred_index_counter_2, reg1, reg2, predr, predtype, start_idx, end_idx, addr, word_length
cntw \word_length
whilelt \predtype, \start_idx, \end_idx
ld1w \reg1, \predr/Z, [\addr]
add \start_idx, \start_idx, \word_length
whilelt \predtype, \start_idx, \end_idx
ld1w \reg2, \predr/Z, [\addr, #1, MUL VL]
neg \word_length, \word_length
add \start_idx, \start_idx, \word_length
neg \word_length, \word_length
.endm

.macro store_data_pred_index_counter_2, reg1, reg2, predr, predtype, start_idx, end_idx, addr, word_length
cntw \word_length
whilelt \predtype, \start_idx, \end_idx
st1w \reg1, \predr, [\addr]
add \start_idx, \start_idx, \word_length
whilelt \predtype, \start_idx, \end_idx
st1w \reg2, \predr, [\addr, #1, MUL VL]
neg \word_length, \word_length
add \start_idx, \start_idx, \word_length
neg \word_length, \word_length
.endm
    
BEGIN_FUNCTION xnn_pf32_gemm_minmax__asm_aarch64_neonsme
        #Save the link register and frame pointer
        stp x29, x30, [sp, #-96]!
        mov x29, sp
        stp x23, x24, [sp, #16]
        stp x21, x22, [sp, #32]
        stp x19, x20, [sp, #48]
        stp x25, x26, [sp, #64]
        stp x27, x28, [sp, #80]

        #Access the flag argument from stack.
        ldr x8, [sp, #104]  
        mov x20, x8
        SMSTART
        ptrue p2.b
        ptrue p5.b
        #Access the accumulator buffer argument from stack.
        ldr x8, [sp, #96]  
        mov x11, x8
        mov x13, x8
        #Access the ldcb argument from stack.
        ldr x8, [sp, #112]  
        tbz x20, #0, Skip_init_accum_load
        mov x14, #0x0
        cntw x17
        #Loop for initial accumulator load from buffer
        Init_accum_load:
        load_data z19.s, z18.s, z17.s, z16.s, p5, x11
        load_data z3.s, z2.s, z1.s, z0.s, p5, x11
        load_data z15.s, z14.s, z13.s, z12.s, p5, x11
        load_data z27.s, z26.s, z25.s, z24.s, p5, x11
        move_vector_tile za0h.s, z19.s, z18.s, z17.s, z16.s, p5, w14
        move_vector_tile za1h.s, z3.s, z2.s, z1.s, z0.s, p5, w14
        move_vector_tile za2h.s, z15.s, z14.s, z13.s, z12.s, p5, w14
        move_vector_tile za3h.s, z27.s, z26.s, z25.s, z24.s, p5, w14
        add x14, x14, #0x4
        cmp x14, x17
        blt Init_accum_load
        #End of loop for initial accumulator load from buffer
        Skip_init_accum_load:
        mov w12, w6
        mov x15, #0x0
        mov x16, #0x0
        mov w28, w7
        mov x21, fpack_lhs_packed_in
        #Loop for iteration over M dimensions
        M_Loop:
        mov x10, fpack_rhs_packed_in
        #Loop for iteration over N dimensions
        N_Loop:
        mov x26, x21
        whilelt p4.s, x16, x28
        tbnz x20, #0, Skip_Bias_Load
        fmov z26.s, #1.0
        #Load the bias into vectors
        load_data_pred_index_counter_2 z21.s, z20.s, p4, p4.s, x16, x28, x10, x23
        zero { za }
        addvl x10, x10, #2
        fmopa za0.s, p2/M, p2/M, z26.s, z21.s
        fmopa za1.s, p2/M, p2/M, z26.s, z20.s
        fmopa za2.s, p2/M, p2/M, z26.s, z21.s
        fmopa za3.s, p2/M, p2/M, z26.s, z20.s
        #Check for last block
        Skip_Bias_Load:
        mov x17, x16
        mov x9, x15
        incw x17, ALL, MUL #2
        incw x9, ALL, MUL #2
        cmp x17, x28
        mov x17, x20
        csel x9, x15, x9, LT
        bfm x20, XZR, #0x0, #0x0 
        cmp x9, x12
        csel x20, x17, x20, LT
        mov x17, k_dimension
        lsr x9, x17, #0x2
        and x17, x17, #0x3
        cbz x9, K_oddments
        subs x9, x9, #0x1
        load_data_2 z6.s, z7.s, p5, x26
        load_data_2 z23.s, z0.s, p5, x10
        load_data_2 z22.s, z1.s, p5, x26
        load_data_2 z11.s, z10.s, p5, x10
        load_data_2 z13.s, z21.s, p5, x26
        load_data_2 z12.s, z20.s, p5, x10
        load_data_2 z14.s, z4.s, p5, x26
        load_data_2 z9.s, z8.s, p5, x10
        ble K_rem_elems
        #Loop for iteration over K dimensions
        K_Loop:
        fmopa za0.s, p2/M, p2/M, z6.s, z23.s
        subs x9, x9, #0x1
        fmopa za1.s, p2/M, p2/M, z6.s, z0.s
        fmopa za2.s, p2/M, p2/M, z7.s, z23.s
        fmopa za3.s, p2/M, p2/M, z7.s, z0.s
        load_data_2 z6.s, z7.s, p5, x26
        fmopa za0.s, p2/M, p2/M, z22.s, z11.s
        load_data_2 z23.s, z0.s, p5, x10
        fmopa za1.s, p2/M, p2/M, z22.s, z10.s
        fmopa za2.s, p2/M, p2/M, z1.s, z11.s
        fmopa za3.s, p2/M, p2/M, z1.s, z10.s
        load_data_2 z22.s, z1.s, p5, x26
        fmopa za0.s, p2/M, p2/M, z13.s, z12.s
        load_data_2 z11.s, z10.s, p5, x10
        fmopa za1.s, p2/M, p2/M, z13.s, z20.s
        fmopa za2.s, p2/M, p2/M, z21.s, z12.s
        fmopa za3.s, p2/M, p2/M, z21.s, z20.s
        load_data_2 z13.s, z21.s, p5, x26
        load_data_2 z12.s, z20.s, p5, x10
        fmopa za0.s, p2/M, p2/M, z14.s, z9.s
        fmopa za1.s, p2/M, p2/M, z14.s, z8.s
        fmopa za2.s, p2/M, p2/M, z4.s, z9.s
        fmopa za3.s, p2/M, p2/M, z4.s, z8.s
        load_data_2 z14.s, z4.s, p5, x26
        load_data_2 z9.s, z8.s, p5, x10
        bgt K_Loop
        #K loop remaining elements
        K_rem_elems:
        fmopa za0.s, p2/M, p2/M, z6.s, z23.s
        fmopa za1.s, p2/M, p2/M, z6.s, z0.s
        fmopa za2.s, p2/M, p2/M, z7.s, z23.s
        fmopa za3.s, p2/M, p2/M, z7.s, z0.s
        fmopa za0.s, p2/M, p2/M, z22.s, z11.s
        fmopa za1.s, p2/M, p2/M, z22.s, z10.s
        fmopa za2.s, p2/M, p2/M, z1.s, z11.s
        fmopa za3.s, p2/M, p2/M, z1.s, z10.s
        fmopa za0.s, p2/M, p2/M, z13.s, z12.s
        fmopa za1.s, p2/M, p2/M, z13.s, z20.s
        fmopa za2.s, p2/M, p2/M, z21.s, z12.s
        fmopa za3.s, p2/M, p2/M, z21.s, z20.s
        fmopa za0.s, p2/M, p2/M, z14.s, z9.s
        fmopa za1.s, p2/M, p2/M, z14.s, z8.s
        fmopa za2.s, p2/M, p2/M, z4.s, z9.s
        fmopa za3.s, p2/M, p2/M, z4.s, z8.s
        #Handler for K oddments
        K_oddments:
        cbz x17, Skip_k_oddments_loop
        #Loop for handling K oddments
        K_oddments_loop:
        load_data_2 z21.s, z20.s, p5, x26
        subs x17, x17, #0x1
        load_data_2 z1.s, z0.s, p5, x10
        fmopa za0.s, p2/M, p2/M, z21.s, z1.s
        fmopa za1.s, p2/M, p2/M, z21.s, z0.s
        fmopa za2.s, p2/M, p2/M, z20.s, z1.s
        fmopa za3.s, p2/M, p2/M, z20.s, z0.s
        bgt K_oddments_loop
        #End of loop for handling K oddments
        Skip_k_oddments_loop:
        tbz x20, #1, Store_out_array
        tbz x20, #0, Store_partial_buff
        mov x14, #0x0
        cntw x17
        #Store to partial result buffer (Store and refill loop)
        Store_refill_partial_res:
        load_data z15.s, z14.s, z13.s, z12.s, p5, x11
        move_tile_vector za0h.s, z11.s, z10.s, z6.s, z7.s, p5, w14
        move_tile_vector za1h.s, z30.s, z31.s, z28.s, z29.s, p5, w14
        load_data z9.s, z8.s, z22.s, z23.s, p5, x11
        move_tile_vector za2h.s, z5.s, z4.s, z21.s, z20.s, p5, w14
        move_tile_vector za3h.s, z3.s, z2.s, z1.s, z0.s, p5, w14
        load_data z27.s, z26.s, z25.s, z24.s, p5, x11
        load_data z19.s, z18.s, z17.s, z16.s, p5, x11
        move_vector_tile za0h.s, z15.s, z14.s, z13.s, z12.s, p5, w14
        move_vector_tile za1h.s, z9.s, z8.s, z22.s, z23.s, p5, w14
        store_data z11.s, z10.s, z6.s, z7.s, p5, x13
        move_vector_tile za2h.s, z27.s, z26.s, z25.s, z24.s, p5, w14
        store_data z30.s, z31.s, z28.s, z29.s, p5, x13
        move_vector_tile za3h.s, z19.s, z18.s, z17.s, z16.s, p5, w14
        add x14, x14, #0x4
        store_data z5.s, z4.s, z21.s, z20.s, p5, x13
        cmp x14, x17
        store_data z3.s, z2.s, z1.s, z0.s, p5, x13
        blt Store_refill_partial_res
        b End_block
        #Store to partial result buffer (Store only)
        Store_partial_buff:
        mov x14, #0x0
        cntw x17
        #Store to partial result buffer (Store only loop)
        Store_partial_buff_loop:
        move_tile_vector za0h.s, z15.s, z14.s, z13.s, z12.s, p5, w14
        move_tile_vector za1h.s, z27.s, z26.s, z25.s, z24.s, p5, w14
        move_tile_vector za2h.s, z30.s, z31.s, z28.s, z29.s, p5, w14
        move_tile_vector za3h.s, z11.s, z10.s, z6.s, z7.s, p5, w14
        store_data z15.s, z14.s, z13.s, z12.s, p5, x13
        add x14, x14, #0x4
        store_data z27.s, z26.s, z25.s, z24.s, p5, x13
        cmp x14, x17
        store_data z30.s, z31.s, z28.s, z29.s, p5, x13
        store_data z11.s, z10.s, z6.s, z7.s, p5, x13
        blt Store_partial_buff_loop
        b End_block
        #Store to output array
        Store_out_array:
        mov x27, output
        sub x25, x12, x15
        mov x24, x8
        add x27, x27, x16, LSL #2
        madd x27, x15, x24, x27
        tbz x20, #2, Store_out_arr_na_end
        cntw x23
        mov x14, #0x0
        cmp x25, x23
        csel x22, x25, x23, LT
        lsr x9, x22, #0x2
        and x17, x22, #0x3
        cbz x9, Store_out_arr_na_r0_oddments
        #Store to output array, no activation, accumulator row 0 loop
        Store_out_arr_na_r0_loop:
        move_tile_vector za0h.s, z9.s, z8.s, z22.s, z23.s, p5, w14
        move_tile_vector za1h.s, z3.s, z2.s, z1.s, z0.s, p5, w14
        store_data_pred_index_counter_2 z9.s, z3.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        add x14, x14, #0x4
        store_data_pred_index_counter_2 z8.s, z2.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        store_data_pred_index_counter_2 z22.s, z1.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        store_data_pred_index_counter_2 z23.s, z0.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        cmp x14, x9, LSL #2
        blt Store_out_arr_na_r0_loop
        #Store to output array, no activation, accumulator row 0 oddments
        Store_out_arr_na_r0_oddments:
        cbz x17, Store_out_arr_na_r0_oddments_end
        move_tile_vector za0h.s, z15.s, z14.s, z13.s, z12.s, p5, w14
        move_tile_vector za1h.s, z5.s, z4.s, z21.s, z20.s, p5, w14
        store_data_pred_index_counter_2 z15.s, z5.s, p4, p4.s, x16, x28, x27, x23
        subs x17, x17, #0x1
        add x27, x27, x24
        beq Store_out_arr_na_r0_oddments_end
        store_data_pred_index_counter_2 z14.s, z4.s, p4, p4.s, x16, x28, x27, x23
        subs x17, x17, #0x1
        add x27, x27, x24
        beq Store_out_arr_na_r0_oddments_end
        store_data_pred_index_counter_2 z13.s, z21.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        #Store to output array, no activation, accumulator row 0 oddments end 
        Store_out_arr_na_r0_oddments_end:
        subs x25, x25, x22
        beq Store_out_arr_na_end
        cmp x25, x23
        mov x14, #0x0
        csel x22, x25, x23, LT
        lsr x9, x22, #0x2
        and x17, x22, #0x3
        cbz x9, Store_out_arr_na_r1_oddments
        #Store to output array, no activation, accumulator row 1 loop
        Store_out_arr_na_r1_loop:
        move_tile_vector za2h.s, z9.s, z8.s, z22.s, z23.s, p5, w14
        move_tile_vector za3h.s, z3.s, z2.s, z1.s, z0.s, p5, w14
        store_data_pred_index_counter_2 z9.s, z3.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        add x14, x14, #0x4
        store_data_pred_index_counter_2 z8.s, z2.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        store_data_pred_index_counter_2 z22.s, z1.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        store_data_pred_index_counter_2 z23.s, z0.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        cmp x14, x9, LSL #2
        blt Store_out_arr_na_r1_loop
        #Store to output array, no activation, accumulator row 1 oddments
        Store_out_arr_na_r1_oddments:
        cbz x17, Store_out_arr_na_r1_oddments_end
        move_tile_vector za2h.s, z9.s, z8.s, z22.s, z23.s, p5, w14
        move_tile_vector za3h.s, z3.s, z2.s, z1.s, z0.s, p5, w14
        store_data_pred_index_counter_2 z9.s, z3.s, p4, p4.s, x16, x28, x27, x23
        subs x17, x17, #0x1
        add x27, x27, x24
        beq Store_out_arr_na_r1_oddments_end
        store_data_pred_index_counter_2 z8.s, z2.s, p4, p4.s, x16, x28, x27, x23
        subs x17, x17, #0x1
        add x27, x27, x24
        beq Store_out_arr_na_r1_oddments_end
        store_data_pred_index_counter_2 z22.s, z1.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        #Store to output array, no activation, accumulator row 1 oddments end
        Store_out_arr_na_r1_oddments_end:
        subs x25, x25, x22
        beq Store_out_arr_na_end
        b Store_out_array_end
        #Store to output array, no activation, end
        Store_out_arr_na_end:
        cntw x23
        ld1rw { z10.s }, p2/Z, [min_value]
        mov x14, #0x0
        cmp x25, x23
        ld1rw { z11.s }, p2/Z, [max_value]
        csel x22, x25, x23, LT
        lsr x9, x22, #0x2
        and x17, x22, #0x3
        cbz x9, Store_out_arr_r0_oddments
        #Store to output array, accumulator row 0 loop
        Store_out_arr_r0_loop:
        move_tile_vector za0h.s, z27.s, z26.s, z25.s, z24.s, p5, w14
        move_tile_vector za1h.s, z19.s, z18.s, z17.s, z16.s, p5, w14
        clamp_float z27.s, z11.s, z10.s, p5
        clamp_float z26.s, z11.s, z10.s, p5
        clamp_float z25.s, z11.s, z10.s, p5
        clamp_float z24.s, z11.s, z10.s, p5
        clamp_float z19.s, z11.s, z10.s, p5
        clamp_float z18.s, z11.s, z10.s, p5
        clamp_float z17.s, z11.s, z10.s, p5
        clamp_float z16.s, z11.s, z10.s, p5
        add x14, x14, #0x4
        store_data_pred_index_counter_2 z27.s, z19.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        store_data_pred_index_counter_2 z26.s, z18.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        store_data_pred_index_counter_2 z25.s, z17.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        store_data_pred_index_counter_2 z24.s, z16.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        cmp x14, x9, LSL #2
        blt Store_out_arr_r0_loop
        #Store to output array, accumulator row 0 oddments
        Store_out_arr_r0_oddments:
        cbz x17, Store_out_arr_r0_oddments_end
        move_tile_vector za0h.s, z27.s, z26.s, z25.s, z24.s, p5, w14
        move_tile_vector za1h.s, z19.s, z18.s, z17.s, z16.s, p5, w14
        clamp_float z27.s, z11.s, z10.s, p5
        clamp_float z26.s, z11.s, z10.s, p5
        clamp_float z25.s, z11.s, z10.s, p5
        clamp_float z24.s, z11.s, z10.s, p5
        clamp_float z19.s, z11.s, z10.s, p5
        clamp_float z18.s, z11.s, z10.s, p5
        clamp_float z17.s, z11.s, z10.s, p5
        clamp_float z16.s, z11.s, z10.s, p5
        store_data_pred_index_counter_2 z27.s, z19.s, p4, p4.s, x16, x28, x27, x23
        subs x17, x17, #0x1
        add x27, x27, x24
        beq Store_out_arr_r0_oddments_end
        store_data_pred_index_counter_2 z26.s, z18.s, p4, p4.s, x16, x28, x27, x23
        subs x17, x17, #0x1
        add x27, x27, x24
        beq Store_out_arr_r0_oddments_end
        store_data_pred_index_counter_2 z25.s, z17.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        #Store to output array, accumulator row 0 oddments, end
        Store_out_arr_r0_oddments_end:
        subs x25, x25, x22
        beq Store_out_array_end
        cmp x25, x23
        mov x14, #0x0
        csel x17, x25, x23, LT
        lsr x9, x17, #0x2
        and x17, x17, #0x3
        cbz x9, Store_out_arr_r1_oddments
        #Store to output array, accumulator row 1 loop
        Store_out_arr_r1_loop:
        move_tile_vector za2h.s, z15.s, z14.s, z13.s, z12.s, p5, w14
        move_tile_vector za3h.s, z5.s, z4.s, z21.s, z20.s, p5, w14
        clamp_float z15.s, z11.s, z10.s, p5
        clamp_float z14.s, z11.s, z10.s, p5
        clamp_float z13.s, z11.s, z10.s, p5
        clamp_float z12.s, z11.s, z10.s, p5
        clamp_float z5.s, z11.s, z10.s, p5
        clamp_float z4.s, z11.s, z10.s, p5
        clamp_float z21.s, z11.s, z10.s, p5
        clamp_float z20.s, z11.s, z10.s, p5
        add x14, x14, #0x4
        store_data_pred_index_counter_2 z15.s, z5.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        store_data_pred_index_counter_2 z14.s, z4.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        store_data_pred_index_counter_2 z13.s, z21.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        store_data_pred_index_counter_2 z12.s, z20.s, p4, p4.s, x16, x28, x27, x23
        add x27, x27, x24
        cmp x14, x9, LSL #2
        blt Store_out_arr_r1_loop
        #Store to output array, accumulator row 1 oddments
        Store_out_arr_r1_oddments:
        cbz x17, Store_out_arr_r1_oddments_end
        move_tile_vector za2h.s, z27.s, z26.s, z25.s, z24.s, p5, w14
        move_tile_vector za3h.s, z19.s, z18.s, z17.s, z16.s, p5, w14
        clamp_float z27.s, z11.s, z10.s, p5
        clamp_float z26.s, z11.s, z10.s, p5
        clamp_float z25.s, z11.s, z10.s, p5
        clamp_float z24.s, z11.s, z10.s, p5
        clamp_float z19.s, z11.s, z10.s, p5
        clamp_float z18.s, z11.s, z10.s, p5
        clamp_float z17.s, z11.s, z10.s, p5
        clamp_float z16.s, z11.s, z10.s, p5
        store_data_pred_index_counter_2 z27.s, z19.s, p4, p4.s, x16, x28, x27, x23
        subs x17, x17, #0x1
        add x27, x27, x24
        beq Store_out_arr_r1_oddments_end
        store_data_pred_index_counter_2 z26.s, z18.s, p4, p4.s, x16, x28, x27, x23
        subs x17, x17, #0x1
        add x27, x27, x24
        beq Store_out_arr_r1_oddments_end
        store_data_pred_index_counter_2 z25.s, z17.s, p4, p4.s, x16, x28, x27, x23
        #Store to output array, accumulator row 1 oddments, end
        Store_out_arr_r1_oddments_end:
        #Store to output array, end
        Store_out_array_end:
        tbz x20, #0, End_block
        mov x14, #0x0
        cntw x17
        #Store to output array, refill accumulators loop
        Store_out_arr_refill_acc_loop:
        load_data z5.s, z4.s, z21.s, z20.s, p5, x11
        load_data z15.s, z14.s, z13.s, z12.s, p5, x11
        load_data z9.s, z8.s, z22.s, z23.s, p5, x11
        load_data z3.s, z2.s, z1.s, z0.s, p5, x11
        move_vector_tile za0h.s, z5.s, z4.s, z21.s, z20.s, p5, w14
        move_vector_tile za1h.s, z15.s, z14.s, z13.s, z12.s, p5, w14
        move_vector_tile za2h.s, z9.s, z8.s, z22.s, z23.s, p5, w14
        move_vector_tile za3h.s, z3.s, z2.s, z1.s, z0.s, p5, w14
        add x14, x14, #0x4
        cmp x14, x17
        blt Store_out_arr_refill_acc_loop
        #End block
        End_block:
        incw x16, ALL, MUL #2
        cmp x16, x28
        blt N_Loop
        incw x15, ALL, MUL #2
        mov x16, #0x0
        cmp x15, x12
        mov x21, x26
        blt M_Loop
        SMSTOP
        #Restore the link register and frame pointer
        ldp x19, x20, [sp, #48]
        ldp x21, x22, [sp, #32]
        ldp x23, x24, [sp, #16]
        ldp x25, x26, [sp, #64]
        ldp x27, x28, [sp, #80]
        ldp x29, x30, [sp], #96
        ret
  
END_FUNCTION xnn_pf32_gemm_minmax__asm_aarch64_neonsme

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
