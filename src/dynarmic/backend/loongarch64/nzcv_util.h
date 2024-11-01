/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <mcl/stdint.hpp>
#include "reg_alloc.h"

namespace Dynarmic::Backend::LoongArch64::NZCV {

    constexpr u32 arm_mask = 0xF000'0000;
    constexpr size_t arm_nzcv_shift = 28;

    constexpr size_t arm_n_flag_inner_sft = 3;
    constexpr size_t arm_z_flag_inner_sft = 2;
    constexpr size_t arm_c_flag_inner_sft = 1;
    constexpr size_t arm_v_flag_inner_sft = 0;

    constexpr size_t arm_n_flag_sft = 3 + arm_nzcv_shift;
    constexpr size_t arm_z_flag_sft = 2 + arm_nzcv_shift;
    constexpr size_t arm_c_flag_sft = 1 + arm_nzcv_shift;
    constexpr size_t arm_v_flag_sft = arm_nzcv_shift;

    constexpr size_t arm_n_flag_mask = 1 << 3;
    constexpr size_t arm_z_flag_mask = 1 << 2;
    constexpr size_t arm_c_flag_mask = 1 << 1;
    constexpr size_t arm_v_flag_mask = 1 << 0;
    constexpr size_t arm_hi_flag_mask = arm_c_flag_mask;
    constexpr size_t arm_ls_flag_mask = arm_z_flag_mask;
    constexpr size_t arm_ge_flag_mask = arm_n_flag_mask | arm_v_flag_mask;
    constexpr size_t arm_gt_flag_mask1 = arm_ge_flag_mask | arm_z_flag_mask;
    constexpr size_t arm_le_flag_mask2 = arm_gt_flag_mask1;


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

    enum class Op {
        Add,
        Sub,
    };

    template<Op op, size_t fsize, size_t wantGetMask, bool want_rst>
    void
    calNZCV(BlockOfCode &code, const Xbyak_loongarch64::XReg &Rresult, const Xbyak_loongarch64::XReg &Ra,
            const Xbyak_loongarch64::XReg &Rb, const Xbyak_loongarch64::XReg &nzcv) {
        code.xor_(nzcv, nzcv, nzcv);
        if constexpr (wantGetMask & NZCV::arm_c_flag_mask) {
            if constexpr (op == Op::Add) {
                code.nor(Wscratch0, Ra, code.zero);
                code.sltu(Wscratch1, Wscratch0, Rb);
            } else if constexpr (op == Op::Sub) {
                code.sltu(Wscratch1, Ra, Rb);
            }
            CODE_WD(bstrins_)(nzcv, Wscratch1, NZCV::arm_c_flag_inner_sft, NZCV::arm_c_flag_inner_sft);
        }
        if constexpr (want_rst == true || wantGetMask != NZCV::arm_c_flag_mask) {
            CODE_WD(add_)(Rresult, Ra, Rb);
        }
        if constexpr (wantGetMask & NZCV::arm_v_flag_mask) {
            code.xor_(Xscratch0, Ra, Rb);
            code.xor_(Xscratch1, Ra, Rresult);
            if constexpr (op == Op::Add) {
                code.andn(Xscratch2, Xscratch1, Xscratch0);
            } else if constexpr (op == Op::Sub) {
                code.and_(Xscratch2, Xscratch1, Xscratch0);
            }
            CODE_WD(srli_)(Xscratch2, Xscratch2, fsize - 1);
            CODE_WD(bstrins_)(nzcv, Xscratch2, NZCV::arm_v_flag_inner_sft, NZCV::arm_v_flag_inner_sft);
        }
        if constexpr (wantGetMask & NZCV::arm_n_flag_mask) {
            CODE_WD(srli_)(Rresult, Rresult, fsize - 1 - NZCV::arm_n_flag_inner_sft);
            CODE_WD(bstrins_)(nzcv, Rresult, NZCV::arm_n_flag_inner_sft, NZCV::arm_n_flag_inner_sft);
        }
        if constexpr (wantGetMask & NZCV::arm_z_flag_mask) {
            CODE_WD(addi_)(Wscratch0, code.zero, NZCV::arm_z_flag_mask);
            code.masknez(Rresult, Wscratch0, Rresult);
            code.or_(nzcv, nzcv, Rresult);
        }
    }
} // namespace Dynarmic::Backend::LoongArch64::NZCV
