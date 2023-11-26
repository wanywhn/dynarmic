/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <mcl/stdint.hpp>

namespace Dynarmic::Backend::LoongArch64::NZCV {
    constexpr u32 arm_mask = 0xF000'0000;

    constexpr size_t arm_n_flag_mask = 1 << 3;
    constexpr size_t arm_z_flag_mask = 1 << 2;
    constexpr size_t arm_c_flag_mask = 1 << 1;
    constexpr size_t arm_v_flag_mask = 1 << 0;
    constexpr size_t arm_hi_flag_mask = arm_c_flag_mask;
    constexpr size_t arm_ls_flag_mask = arm_z_flag_mask;
    constexpr size_t arm_ge_flag_mask = arm_n_flag_mask | arm_v_flag_mask;
    constexpr size_t arm_gt_flag_mask1 = arm_ge_flag_mask| arm_z_flag_mask;
    constexpr size_t arm_le_flag_mask2 = arm_gt_flag_mask1;
    constexpr size_t arm_nzcv_shift = 28;


    inline u32 ToLoongArch64(u32 nzcv) {
        /* Naive implementation:
        u32 x64_flags = 0;
        x64_flags |= mcl::bit::get_bit<31>(cpsr) ? 1 << 15 : 0;
        x64_flags |= mcl::bit::get_bit<30>(cpsr) ? 1 << 14 : 0;
        x64_flags |= mcl::bit::get_bit<29>(cpsr) ? 1 << 8 : 0;
        x64_flags |= mcl::bit::get_bit<28>(cpsr) ? 1 : 0;
        return x64_flags;
        */
        return (nzcv >> 28);
    }

    inline u32 FromLoongArch64(u32 x64_flags) {
        /* Naive implementation:
        u32 nzcv = 0;
        nzcv |= mcl::bit::get_bit<15>(x64_flags) ? 1 << 31 : 0;
        nzcv |= mcl::bit::get_bit<14>(x64_flags) ? 1 << 30 : 0;
        nzcv |= mcl::bit::get_bit<8>(x64_flags) ? 1 << 29 : 0;
        nzcv |= mcl::bit::get_bit<0>(x64_flags) ? 1 << 28 : 0;
        return nzcv;
        */
        return x64_flags << 28;
    }
} // namespace Dynarmic::Backend::LoongArch64::NZCV
