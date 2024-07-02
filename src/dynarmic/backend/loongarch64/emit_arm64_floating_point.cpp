/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/loongarch64/a32_jitstate.h"
#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/emit_arm64.h"
#include "dynarmic/backend/loongarch64/emit_context.h"
#include "dynarmic/backend/loongarch64/fpsr_manager.h"
#include "dynarmic/backend/loongarch64/reg_alloc.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"
#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"
#include "nzcv_util.h"
#include "mcl/type_traits/integer_of_size.hpp"
#include "dynarmic/common/fp/op/FPRecipStepFused.h"
#include "dynarmic/common/lut_from_list.h"
#include "dynarmic/common/fp/op/FPToFixed.h"
#include "mcl/bit_cast.hpp"
#include <mcl/assert.hpp>
#include <mcl/mp/metavalue/lift_value.hpp>
#include <mcl/mp/typelist/cartesian_product.hpp>
#include <mcl/mp/typelist/get.hpp>
#include <mcl/mp/typelist/lift_sequence.hpp>
#include <mcl/mp/typelist/list.hpp>
#include <mcl/mp/typelist/lower_to_tuple.hpp>
#include <mcl/stdint.hpp>
#include <mcl/type_traits/integer_of_size.hpp>
#include "dynarmic/common/fp/op.h"

namespace Dynarmic::Backend::LoongArch64 {

using namespace Xbyak_loongarch64::util;

    constexpr u64 f32_negative_zero = 0x80000000u;
    constexpr u64 f32_nan = 0x7fc00000u;
    constexpr u64 f32_non_sign_mask = 0x7fffffffu;
    constexpr u64 f32_smallest_normal = 0x00800000u;

    constexpr u64 f64_negative_zero = 0x8000000000000000u;
    constexpr u64 f64_nan = 0x7ff8000000000000u;
    constexpr u64 f64_non_sign_mask = 0x7fffffffffffffffu;
    constexpr u64 f64_smallest_normal = 0x0010000000000000u;

    constexpr u64 f64_min_s16 = 0xc0e0000000000000u;      // -32768 as a double
    constexpr u64 f64_max_s16 = 0x40dfffc000000000u;      // 32767 as a double
    constexpr u64 f64_min_u16 = 0x0000000000000000u;      // 0 as a double
    constexpr u64 f64_max_u16 = 0x40efffe000000000u;      // 65535 as a double
    constexpr u64 f64_max_s32 = 0x41dfffffffc00000u;      // 2147483647 as a double
    constexpr u64 f64_max_u32 = 0x41efffffffe00000u;      // 4294967295 as a double
    constexpr u64 f64_max_s64_lim = 0x43e0000000000000u;  // 2^63 as a double (actual maximum unrepresentable)


#define FCODE(NAME)                  \
    [&code](auto... args) {          \
        if constexpr (fsize == 32) { \
            code.NAME##s(args...);   \
        } else {                     \
            code.NAME##d(args...);   \
        }                            \
    }
#define ICODE(NAME)                  \
    [&code](auto... args) {          \
        if constexpr (fsize == 32) { \
            code.NAME##d(args...);   \
        } else {                     \
            code.NAME##q(args...);   \
        }                            \
    }


    template<size_t fsize>
    void ForceToDefaultNaN(BlockOfCode& code, Xbyak_loongarch64::XReg result) {

        Xbyak_loongarch64::Label end;
        FCODE(fcmp_sun_)(0, result, result);
        code.movcf2gr(Xscratch0, 0);
        code.beqz(Xscratch0, end);
        code.add_imm(Xscratch0, code.zero, fsize == 32 ? f32_nan : f64_nan, Xscratch2);
        code.fmov_s(result, Xscratch0);
        code.L(end);

    }

template<size_t bitsize, typename EmitFn>
static void EmitTwoOp(BlockOfCode&, EmitContext& ctx, IR::Inst* inst, EmitFn emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteReg<bitsize>(inst);
    auto Voperand = ctx.reg_alloc.ReadReg<bitsize>(args[0]);
    RegAlloc::Realize(Vresult, Voperand);

    emit(Vresult, Voperand);
}

template<size_t bitsize, typename EmitFn>
static void EmitThreeOp(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst, EmitFn emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteReg<bitsize>(inst);
    auto Va = ctx.reg_alloc.ReadReg<bitsize>(args[0]);
    auto Vb = ctx.reg_alloc.ReadReg<bitsize>(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);

    if (ctx.FPCR().DN() || ctx.conf.HasOptimization(OptimizationFlag::Unsafe_InaccurateNaN)) {
            code.movfcsr2gr(Wscratch0, Fscratch1);
            code.bstrins_w(Wscratch0, code.zero, 4, 4);
            code.movgr2fcsr(Fscratch1, Wscratch0);
            emit(Vresult, Va, Vb);


        if (!ctx.conf.HasOptimization(OptimizationFlag::Unsafe_InaccurateNaN)) {
            // FIXME
            ForceToDefaultNaN<bitsize>(code, Vresult);
        }
        return;
    }
// FIXME see x86
    emit(Vresult, Va, Vb);
}

template<size_t bitsize, typename EmitFn>
static void EmitFourOp(BlockOfCode&, EmitContext& ctx, IR::Inst* inst, EmitFn emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteReg<bitsize>(inst);
    auto Va = ctx.reg_alloc.ReadReg<bitsize>(args[0]);
    auto Vb = ctx.reg_alloc.ReadReg<bitsize>(args[1]);
    auto Vc = ctx.reg_alloc.ReadReg<bitsize>(args[2]);
    RegAlloc::Realize(Vresult, Va, Vb, Vc);
    ctx.fpsr.Load();

    emit(Vresult, Va, Vb, Vc);
}

template<size_t bitsize_from, size_t bitsize_to, typename EmitFn>
static void EmitConvert(BlockOfCode&, EmitContext& ctx, IR::Inst* inst, EmitFn emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vto = ctx.reg_alloc.WriteVec<bitsize_to>(inst);
    auto Vfrom = ctx.reg_alloc.ReadVec<bitsize_from>(args[0]);
    const auto rounding_mode = static_cast<FP::RoundingMode>(args[1].GetImmediateU8());
    RegAlloc::Realize(Vto, Vfrom);
    ctx.fpsr.Load();

    ASSERT(rounding_mode == ctx.FPCR().RMode());

    emit(Vto, Vfrom);
}
namespace mp = mcl::mp;

template<size_t bitsize_from, size_t bitsize_to, bool is_signed>
static void EmitToFixed(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Rto = ctx.reg_alloc.WriteReg<std::max<size_t>(bitsize_to, 32)>(inst);
    auto Vfrom = ctx.reg_alloc.ReadReg<bitsize_from>(args[0]);
    const size_t fbits = args[1].GetImmediateU8();
    const auto rounding_mode = static_cast<FP::RoundingMode>(args[2].GetImmediateU8());
    RegAlloc::Realize(Rto, Vfrom);
    ctx.fpsr.Load();

    using fbits_list = mp::lift_sequence<std::make_index_sequence<bitsize_to + 1>>;
    using rounding_list = mp::list<
    mp::lift_value<FP::RoundingMode::ToNearest_TieEven>,
    mp::lift_value<FP::RoundingMode::TowardsPlusInfinity>,
    mp::lift_value<FP::RoundingMode::TowardsMinusInfinity>,
    mp::lift_value<FP::RoundingMode::TowardsZero>,
    mp::lift_value<FP::RoundingMode::ToNearest_TieAwayFromZero>>;

    static const auto lut = Common::GenerateLookupTableFromList(
            []<typename I>(I) {
                return std::pair{
                        mp::lower_to_tuple_v<I>,
                        Common::FptrCast(
                                [](u64 input, FP::FPSR& fpsr, FP::FPCR fpcr) {
                                    constexpr size_t fbits = mp::get<0, I>::value;
                                    constexpr FP::RoundingMode rounding_mode = mp::get<1, I>::value;
                                    using FPT = mcl::unsigned_integer_of_size<bitsize_from>;

                                    return FP::FPToFixed<FPT>(bitsize_to, static_cast<FPT>(input), fbits, is_signed, fpcr, rounding_mode, fpsr);
                                })};
            },
            mp::cartesian_product<fbits_list, rounding_list>{});
    ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
    code.movfr2gr_d(Wscratch0, Vfrom);
    code.add_d(code.a0, code.zero, Wscratch0);
    code.addi_d(code.a1, Xstate, code.GetJitStateInfo().offsetof_fpsr_exc);
    code.add_imm(code.a2, code.zero, ctx.FPCR().Value(), Xscratch2);
    code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(lut.at(std::make_tuple(fbits, rounding_mode))), Xscratch2);
    code.jirl(code.ra, Xscratch0, 0);
    code.add_d(Rto, code.zero, code.a0);
    ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
}

template<size_t bitsize_from, size_t bitsize_to, typename EmitFn>
static void EmitFromFixed(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst, EmitFn emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vto = ctx.reg_alloc.WriteReg<bitsize_to>(inst);
    auto Rfrom = ctx.reg_alloc.ReadReg<std::max<size_t>(bitsize_from, 32)>(args[0]);
    const size_t fbits = args[1].GetImmediateU8();
    const auto rounding_mode = static_cast<FP::RoundingMode>(args[2].GetImmediateU8());
    RegAlloc::Realize(Vto, Rfrom);
    ctx.fpsr.Load();

    if (rounding_mode == ctx.FPCR().RMode()) {
        emit(Vto, Rfrom, fbits);
    } else {
        FP::FPCR new_fpcr = ctx.FPCR();
        // FIXME convert to arch spefic rounding_mode mode ?
        new_fpcr.RMode(rounding_mode);

        if constexpr (bitsize_from != 32) {
            code.add_imm(Wscratch0, code.zero, new_fpcr.Value(), Xscratch2);
            code.movgr2fcsr(code.fcsr3, Wscratch0);
        } else {
            code.EnterStandardASIMD();
        }

        emit(Vto, Rfrom, fbits);
        if constexpr (bitsize_from != 32) {
            code.add_imm(Wscratch0, code.zero, ctx.FPCR().Value(), Xscratch2);
            code.movgr2fcsr(code.fcsr3, Wscratch0);
        } else {
            code.EnterStandardASIMD();
        }
    }
}

template<>
void EmitIR<IR::Opcode::FPAbs16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPAbs32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Soperand) { code.fabs_s(Sresult, Soperand); });
}

template<>
void EmitIR<IR::Opcode::FPAbs64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Doperand) { code.fabs_d(Dresult, Doperand); });
}

template<>
void EmitIR<IR::Opcode::FPAdd32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.fadd_s(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPAdd64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.fadd_d(Dresult, Da, Db); });
}


#define GetNZCV_W(Wscratch0, Wscratch1, Wscratch2, Va, Vb, s_u, d_w) \
    code.xor_(Wscratch1, code.zero, code.zero);\
    code.fcmp_##s_u##eq_##d_w(0, Va, Vb);\
    code.movcf2gr(Wscratch2, 0);\
    code.bstrins_w(Wscratch1, Wscratch2, NZCV::arm_z_flag_sft, NZCV::arm_z_flag_sft);\
    code.fcmp_##s_u##lt_##d_w(0, Va, Vb);\
    code.movcf2gr(Wscratch2, 0);\
    code.bstrins_w(Wscratch1, Wscratch2, NZCV::arm_n_flag_sft, NZCV::arm_n_flag_sft);\
    code.addi_w(Wscratch2, Wscratch2, 1);\
    code.bstrins_w(Wscratch1, Wscratch2, NZCV::arm_c_flag_sft, NZCV::arm_c_flag_sft);\
    code.fcmp_##s_u##un_##d_w(0, Va, Vb);\
    code.movcf2gr(Wscratch2, 0);\
    code.bstrins_w(Wscratch1, Wscratch2, NZCV::arm_v_flag_sft, NZCV::arm_v_flag_sft);

#define GetNZCV_D(Wscratch0, Wscratch1, Wscratch2, Va, Vb, s_u, d_w) \
    code.xor_(Wscratch1, code.zero, code.zero);\
    code.fcmp_##s_u##eq_##d_w(0, Va, Vb);\
    code.movcf2gr(Wscratch2, 0);\
    code.bstrins_d(Wscratch1, Wscratch2, NZCV::arm_z_flag_sft, NZCV::arm_z_flag_sft);\
    code.fcmp_##s_u##lt_##d_w(0, Va, Vb);\
    code.movcf2gr(Wscratch2, 0);\
    code.bstrins_d(Wscratch1, Wscratch2, NZCV::arm_n_flag_sft, NZCV::arm_n_flag_sft);\
    code.addi_d(Wscratch2, Wscratch2, 1);\
    code.bstrins_d(Wscratch1, Wscratch2, NZCV::arm_c_flag_sft, NZCV::arm_c_flag_sft);\
    code.fcmp_##s_u##un_##d_w(0, Va, Vb);\
    code.movcf2gr(Wscratch2, 0);\
    code.bstrins_d(Wscratch1, Wscratch2, NZCV::arm_v_flag_sft, NZCV::arm_v_flag_sft);


template<size_t size>
void EmitCompare(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto flags = ctx.reg_alloc.WriteReg<32>(inst);
    auto Va = ctx.reg_alloc.ReadReg<size>(args[0]);
    auto Vb = ctx.reg_alloc.ReadReg<size>(args[1]);

    const bool exc_on_qnan = args[2].GetImmediateU1();

    RegAlloc::Realize(flags, Va, Vb);
    if (exc_on_qnan) {
        if constexpr (size == 32) {
            GetNZCV_W(Wscratch0, flags, Wscratch2, Va, Vb, s, s);
        } else {
            GetNZCV_D(Wscratch0, flags, Wscratch2, Va, Vb, s, d);
        }
    } else {
        if constexpr (size == 32) {
            GetNZCV_W(Wscratch0, flags, Wscratch2, Va, Vb, c, s);
        } else {
            GetNZCV_D(Wscratch0, flags, Wscratch2, Va, Vb, c, d);
        }
    }


}

template<>
void EmitIR<IR::Opcode::FPCompare32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitCompare<32>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPCompare64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitCompare<64>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPDiv32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.fdiv_s(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPDiv64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.fdiv_d(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPMax32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.fmax_s(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPMax64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.fmax_d(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPMaxNumeric32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {// FIXME
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.fmax_s(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPMaxNumeric64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.fmax_d(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPMin32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.fmin_s(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPMin64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.fmin_d(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPMinNumeric32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {// FIXME
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.fmin_s(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPMinNumeric64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.fmin_d(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPMul32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.fmul_s(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPMul64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.fmul_d(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPMulAdd16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPMulAdd32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFourOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& S1, auto& S2) { code.fmadd_s(Sresult, S1, S2, Sa); });
}

template<>
void EmitIR<IR::Opcode::FPMulAdd64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFourOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& D1, auto& D2) { code.fmadd_d(Dresult, D1, D2, Da); });
}

template<>
void EmitIR<IR::Opcode::FPMulX32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) { // FIXME
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.fmul_s(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPMulX64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.fmul_d(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPNeg16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPNeg32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Soperand) { code.fneg_s(Sresult, Soperand); });
}

template<>
void EmitIR<IR::Opcode::FPNeg64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Doperand) { code.fneg_d(Dresult, Doperand); });
}

template<>
void EmitIR<IR::Opcode::FPRecipEstimate16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPRecipEstimate32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) { // FIXME frecipe_s
    EmitTwoOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Soperand) { code.frecip_s(Sresult, Soperand); });
}

template<>
void EmitIR<IR::Opcode::FPRecipEstimate64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Doperand) { code.frecip_d(Dresult, Doperand); });
}
    template<size_t fsize>
    static void EmitFPRecipExponent(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
        using FPT = mcl::unsigned_integer_of_size<fsize>;

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);


        auto Rto = ctx.reg_alloc.WriteReg<32>(inst);
        auto Vfrom = ctx.reg_alloc.ReadReg<64>(args[0]);
        RegAlloc::Realize(Rto, Vfrom);

        ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
        code.movfr2gr_d(Wscratch0, Vfrom);
        code.add_d(code.a0, code.zero, Wscratch0);
        code.add_imm(code.a1, code.zero, ctx.FPCR().Value(), Xscratch2);
        code.addi_d(code.a2, Xstate, code.GetJitStateInfo().offsetof_fpsr_exc);
        code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(&FP::FPRecipExponent<FPT>), Xscratch2);
        code.jirl(code.ra, Xscratch0, 0);
        code.add_d(Rto, code.zero, code.a0);
        ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
    }

template<>
void EmitIR<IR::Opcode::FPRecipExponent16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFPRecipExponent<16>(code, ctx, inst);

}

template<>
void EmitIR<IR::Opcode::FPRecipExponent32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) { // FIXME
    EmitFPRecipExponent<32>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPRecipExponent64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFPRecipExponent<64>(code, ctx, inst);
}
    template<size_t fsize>
    static void EmitFPRecipStepFused(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
        using FPT = mcl::unsigned_integer_of_size<fsize>;

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);


        auto Rto = ctx.reg_alloc.WriteReg<32>(inst);
        auto Vfrom = ctx.reg_alloc.ReadReg<64>(args[0]);
        auto arg1 = ctx.reg_alloc.ReadReg<64>(args[1]);

        RegAlloc::Realize(Rto, Vfrom);

        ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
        code.movfr2gr_d(Wscratch0, Vfrom);
        code.add_d(code.a0, code.zero, Wscratch0);
        code.add_d(code.a1, code.zero, arg1);
        code.add_imm(code.a2, code.zero, ctx.FPCR().Value(), Xscratch2);
        code.addi_d(code.a3, Xstate, code.GetJitStateInfo().offsetof_fpsr_exc);

        code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(&FP::FPRecipStepFused<FPT>), Xscratch2);
        code.jirl(code.ra, Xscratch0, 0);
        code.add_d(Rto, code.zero, code.a0);
        ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
    }


template<>
void EmitIR<IR::Opcode::FPRecipStepFused16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFPRecipStepFused<16>(code, ctx, inst);

}

template<>
void EmitIR<IR::Opcode::FPRecipStepFused32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFPRecipStepFused<32>(code, ctx, inst);

}

template<>
void EmitIR<IR::Opcode::FPRecipStepFused64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFPRecipStepFused<64>(code, ctx, inst);
}

    static void EmitFPRound(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst, size_t fsize) {
        const auto rounding_mode = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());
        const bool exact = inst->GetArg(2).GetU1();

        using fsize_list = mp::list<mp::lift_value<size_t(16)>,
                mp::lift_value<size_t(32)>,
                mp::lift_value<size_t(64)>>;
        using rounding_list = mp::list<
                mp::lift_value<FP::RoundingMode::ToNearest_TieEven>,
                mp::lift_value<FP::RoundingMode::TowardsPlusInfinity>,
                mp::lift_value<FP::RoundingMode::TowardsMinusInfinity>,
                mp::lift_value<FP::RoundingMode::TowardsZero>,
                mp::lift_value<FP::RoundingMode::ToNearest_TieAwayFromZero>>;
        using exact_list = mp::list<std::true_type, std::false_type>;

        static const auto lut = Common::GenerateLookupTableFromList(
                []<typename I>(I) {
                    return std::pair{
                            mp::lower_to_tuple_v<I>,
                            Common::FptrCast(
                                    [](u64 input, FP::FPSR& fpsr, FP::FPCR fpcr) {
                                        constexpr size_t fsize = mp::get<0, I>::value;
                                        constexpr FP::RoundingMode rounding_mode = mp::get<1, I>::value;
                                        constexpr bool exact = mp::get<2, I>::value;
                                        using InputSize = mcl::unsigned_integer_of_size<fsize>;

                                        return FP::FPRoundInt<InputSize>(static_cast<InputSize>(input), fpcr, rounding_mode, exact, fpsr);
                                    })};
                },
                mp::cartesian_product<fsize_list, rounding_list, exact_list>{});

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Rto = ctx.reg_alloc.WriteReg<32>(inst);
        auto Vfrom = ctx.reg_alloc.ReadReg<64>(args[0]);
        RegAlloc::Realize(Rto, Vfrom);

        ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
        code.movfr2gr_d(Wscratch0, Vfrom);
        code.add_d(code.a0, code.zero, Wscratch0);
        code.addi_d(code.a1, Xstate, code.GetJitStateInfo().offsetof_fpsr_exc);
        code.add_imm(code.a2, code.zero, ctx.FPCR().Value(), Xscratch2);

        code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(lut.at(std::make_tuple(fsize, rounding_mode, exact))), Xscratch2);
        code.jirl(code.ra, Xscratch0, 0);
        code.add_d(Rto, code.zero, code.a0);
        ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
    }

template<>
void EmitIR<IR::Opcode::FPRoundInt16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFPRound(code, ctx, inst, 16);

}

template<>
void EmitIR<IR::Opcode::FPRoundInt32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFPRound(code, ctx, inst, 32);

}

template<>
void EmitIR<IR::Opcode::FPRoundInt64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFPRound(code, ctx, inst, 64);
}
    template<size_t fsize>
    static void EmitFPRSqrtEstimate(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
        using FPT = mcl::unsigned_integer_of_size<fsize>;

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        auto Rto = ctx.reg_alloc.WriteReg<32>(inst);
        auto Vfrom = ctx.reg_alloc.ReadReg<64>(args[0]);
        RegAlloc::Realize(Rto, Vfrom);

        ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
        code.movfr2gr_d(Wscratch0, Vfrom);
        code.add_d(code.a0, code.zero, Wscratch0);
        code.add_imm(code.a1, code.zero, ctx.FPCR().Value(), Xscratch2);
        code.addi_d(code.a2, Xstate, code.GetJitStateInfo().offsetof_fpsr_exc);
        code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(&FP::FPRSqrtEstimate<FPT>), Xscratch2);
        code.jirl(code.ra, Xscratch0, 0);
        code.add_d(Rto, code.zero, code.a0);
        ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);


    }
template<>
void EmitIR<IR::Opcode::FPRSqrtEstimate16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFPRSqrtEstimate<16>(code,ctx, inst);

}

template<>
void EmitIR<IR::Opcode::FPRSqrtEstimate32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFPRSqrtEstimate<32>(code,ctx, inst);
}


template<>
void EmitIR<IR::Opcode::FPRSqrtEstimate64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFPRSqrtEstimate<64>(code,ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPRSqrtStepFused16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPRSqrtStepFused32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa) { code.frsqrt_s(Sresult, Sa); });
}

template<>
void EmitIR<IR::Opcode::FPRSqrtStepFused64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da) { code.frsqrt_d(Dresult, Da); });
}

template<>
void EmitIR<IR::Opcode::FPSqrt32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Soperand) { code.fsqrt_s(Sresult, Soperand); });
}

template<>
void EmitIR<IR::Opcode::FPSqrt64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitTwoOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Doperand) { code.fsqrt_d(Dresult, Doperand); });
}

template<>
void EmitIR<IR::Opcode::FPSub32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<32>(code, ctx, inst, [&](auto& Sresult, auto& Sa, auto& Sb) { code.fsub_s(Sresult, Sa, Sb); });
}

template<>
void EmitIR<IR::Opcode::FPSub64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitThreeOp<64>(code, ctx, inst, [&](auto& Dresult, auto& Da, auto& Db) { code.fsub_d(Dresult, Da, Db); });
}

template<>
void EmitIR<IR::Opcode::FPHalfToDouble>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    const auto rounding_mode = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Rto = ctx.reg_alloc.WriteReg<32>(inst);
    auto Vfrom = ctx.reg_alloc.ReadReg<64>(args[0]);
    RegAlloc::Realize(Rto, Vfrom);

    ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
    code.movfr2gr_d(Wscratch0, Vfrom);
    code.add_d(code.a0, code.zero, Wscratch0);
    code.add_imm(code.a1, code.zero, ctx.FPCR().Value(), Xscratch2);
    code.add_imm(code.a2, code.zero, static_cast<u32>(rounding_mode), Xscratch2);
    code.addi_d(code.a3, Xstate, code.GetJitStateInfo().offsetof_fpsr_exc);
    code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(&FP::FPConvert<u64, u16>), Xscratch2);
    code.jirl(code.ra, Xscratch0, 0);
    code.add_d(Rto, code.zero, code.a0);
    ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);

}

template<>
void EmitIR<IR::Opcode::FPHalfToSingle>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    const auto rounding_mode = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Rto = ctx.reg_alloc.WriteReg<32>(inst);
    auto Vfrom = ctx.reg_alloc.ReadReg<64>(args[0]);
    RegAlloc::Realize(Rto, Vfrom);

    ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
    code.movfr2gr_d(Wscratch0, Vfrom);
    code.add_d(code.a0, code.zero, Wscratch0);
    code.add_imm(code.a1, code.zero, ctx.FPCR().Value(), Xscratch2);
    code.add_imm(code.a2, code.zero, static_cast<u32>(rounding_mode), Xscratch2);
    code.addi_d(code.a3, Xstate, code.GetJitStateInfo().offsetof_fpsr_exc);
    code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(&FP::FPConvert<u32, u16>), Xscratch2);
    code.jirl(code.ra, Xscratch0, 0);
    code.add_d(Rto, code.zero, code.a0);
    ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
}

template<>
void EmitIR<IR::Opcode::FPSingleToDouble>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    const auto rounding_mode = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Rto = ctx.reg_alloc.WriteReg<32>(inst);
    auto Vfrom = ctx.reg_alloc.ReadReg<64>(args[0]);
    RegAlloc::Realize(Rto, Vfrom);

    // We special-case the non-IEEE-defined ToOdd rounding mode.
    if (rounding_mode == ctx.FPCR().RMode() && rounding_mode != FP::RoundingMode::ToOdd) {
        code.fcvt_s_d(Rto, Vfrom);
        if (ctx.FPCR().DN()) {
            ForceToDefaultNaN<64>(code, Rto);
        }
    } else {

        ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
        code.movfr2gr_d(Wscratch0, Vfrom);
        code.add_d(code.a0, code.zero, Wscratch0);
        code.add_imm(code.a1, code.zero, ctx.FPCR().Value(), Xscratch2);
        code.add_imm(code.a2, code.zero, static_cast<u32>(rounding_mode), Xscratch2);
        code.addi_d(code.a3, Xstate, code.GetJitStateInfo().offsetof_fpsr_exc);
        code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(&FP::FPConvert<u64, u32>), Xscratch2);
        code.jirl(code.ra, Xscratch0, 0);
        code.add_d(Rto, code.zero, code.a0);
        ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
    }}

template<>
void EmitIR<IR::Opcode::FPSingleToHalf>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    const auto rounding_mode = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Rto = ctx.reg_alloc.WriteReg<32>(inst);
    auto Vfrom = ctx.reg_alloc.ReadReg<64>(args[0]);
    RegAlloc::Realize(Rto, Vfrom);

    ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
    code.movfr2gr_d(Wscratch0, Vfrom);
    code.add_d(code.a0, code.zero, Wscratch0);
    code.add_imm(code.a1, code.zero, ctx.FPCR().Value(), Xscratch2);
    code.add_imm(code.a2, code.zero, static_cast<u32>(rounding_mode), Xscratch2);
    code.addi_d(code.a3, Xstate, code.GetJitStateInfo().offsetof_fpsr_exc);
    code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(&FP::FPConvert<u16, u32>), Xscratch2);
    code.jirl(code.ra, Xscratch0, 0);
    code.add_d(Rto, code.zero, code.a0);
    ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
}

template<>
void EmitIR<IR::Opcode::FPDoubleToHalf>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    const auto rounding_mode = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Rto = ctx.reg_alloc.WriteReg<32>(inst);
    auto Vfrom = ctx.reg_alloc.ReadReg<64>(args[0]);
    RegAlloc::Realize(Rto, Vfrom);

    ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
    code.movfr2gr_d(Wscratch0, Vfrom);
    code.add_d(code.a0, code.zero, Wscratch0);
    code.add_imm(code.a1, code.zero, ctx.FPCR().Value(), Xscratch2);
    code.add_imm(code.a2, code.zero, static_cast<u32>(rounding_mode), Xscratch2);
    code.addi_d(code.a3, Xstate, code.GetJitStateInfo().offsetof_fpsr_exc);
    code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(&FP::FPConvert<u16, u64>), Xscratch2);
    code.jirl(code.ra, Xscratch0, 0);
    code.add_d(Rto, code.zero, code.a0);
    ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
}

template<>
void EmitIR<IR::Opcode::FPDoubleToSingle>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    const auto rounding_mode = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);

    auto Rto = ctx.reg_alloc.WriteReg<32>(inst);
    auto Vfrom = ctx.reg_alloc.ReadReg<64>(args[0]);
    RegAlloc::Realize(Rto, Vfrom);

    // We special-case the non-IEEE-defined ToOdd rounding mode.
    if (rounding_mode == ctx.FPCR().RMode() && rounding_mode != FP::RoundingMode::ToOdd) {
        code.fcvt_d_s(Rto, Vfrom);
        if (ctx.FPCR().DN()) {
            ForceToDefaultNaN<32>(code, Rto);
        }
    } else {

        ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
        code.movfr2gr_d(Wscratch0, Vfrom);
        code.add_d(code.a0, code.zero, Wscratch0);
        code.add_imm(code.a1, code.zero, ctx.FPCR().Value(), Xscratch2);
        code.add_imm(code.a2, code.zero, static_cast<u32>(rounding_mode), Xscratch2);
        code.addi_d(code.a3, Xstate, code.GetJitStateInfo().offsetof_fpsr_exc);
        code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(&FP::FPConvert<u32, u64>), Xscratch2);
        code.jirl(code.ra, Xscratch0, 0);
        code.add_d(Rto, code.zero, code.a0);
        ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << Rto->getIdx()) & ~(1ull << Vfrom->getIdx()), 0);
    }
}

template<>
void EmitIR<IR::Opcode::FPDoubleToFixedS16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitToFixed<64, 16, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPDoubleToFixedS32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitToFixed<64, 32, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPDoubleToFixedS64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    // TODO: Consider fpr source
    EmitToFixed<64, 64, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPDoubleToFixedU16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitToFixed<64, 16, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPDoubleToFixedU32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitToFixed<64, 32, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPDoubleToFixedU64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    // TODO: Consider fpr source
    EmitToFixed<64, 64, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPHalfToFixedS16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPHalfToFixedS32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPHalfToFixedS64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPHalfToFixedU16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPHalfToFixedU32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPHalfToFixedU64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::FPSingleToFixedS16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitToFixed<32, 16, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPSingleToFixedS32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    // TODO: Consider fpr source
    EmitToFixed<32, 32, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPSingleToFixedS64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitToFixed<32, 64, true>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPSingleToFixedU16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitToFixed<32, 16, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPSingleToFixedU32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    // TODO: Consider fpr source
    EmitToFixed<32, 32, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPSingleToFixedU64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitToFixed<32, 64, false>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::FPFixedU16ToSingle>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<16, 32>(code, ctx, inst, [&](auto& Sto, auto& Wfrom, u8 fbits) {
        code.bstrpick_w(Wscratch0, Wfrom, 15, 0);
        code.ffint_s_w(Sto, Wscratch0);
        if (fbits != 0) {
            const u32 scale_factor = static_cast<u32>((127 - fbits) << 23);
            code.add_imm(Wscratch0, code.zero, scale_factor, Wscratch2);
            code.movgr2fr_d(Fscratch0, Xscratch0);

            code.fmul_s(Sto, Sto, Fscratch0);
        }
    });
}

template<>
void EmitIR<IR::Opcode::FPFixedS16ToSingle>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<16, 32>(code, ctx, inst, [&](auto& Sto, auto& Wfrom, u8 fbits) {
        code.add_w(Wscratch0, code.zero, Wfrom);
        code.ffint_s_w(Sto, Wscratch0);
        if (fbits != 0) {
            const u32 scale_factor = static_cast<u32>((127 - fbits) << 23);
            code.add_imm(Wscratch0, code.zero, scale_factor, Wscratch2);
            code.movgr2fr_d(Fscratch0, Xscratch0);

            code.fmul_s(Sto, Sto, Fscratch0);
        }
    });
}

template<>
void EmitIR<IR::Opcode::FPFixedU16ToDouble>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<16, 64>(code, ctx, inst, [&](auto& Dto, auto& Wfrom, u8 fbits) {
        code.bstrpick_w(Wscratch0, Wfrom, 15, 0);
        code.ffint_d_w(Dto, Wscratch0);
        if (fbits != 0) {
            const u32 scale_factor = static_cast<u32>((127 - fbits) << 23);
            code.add_imm(Wscratch0, code.zero, scale_factor, Wscratch2);
            code.movgr2fr_d(Fscratch0, Xscratch0);

            code.fmul_d(Dto, Dto, Fscratch0);
        }
    });
}

template<>
void EmitIR<IR::Opcode::FPFixedS16ToDouble>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<16, 64>(code, ctx, inst, [&](auto& Dto, auto& Wfrom, u8 fbits) {
        code.add_w(Wscratch0, code.zero, Wfrom);
        code.ffint_d_w(Dto, Wscratch0);
        if (fbits != 0) {
            const u32 scale_factor = static_cast<u32>((127 - fbits) << 23);
            code.add_imm(Wscratch0, code.zero, scale_factor, Wscratch2);
            code.fmul_d(Dto, Dto, Fscratch0);
        }
    });
}

template<>
void EmitIR<IR::Opcode::FPFixedU32ToSingle>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    // TODO: Consider fpr source
    EmitFromFixed<32, 32>(code, ctx, inst, [&](auto& Sto, auto& Wfrom, u8 fbits) {
        code.bstrpick_d(Xscratch0, Wfrom, 31, 0);
        code.ffint_s_w(Sto, Xscratch0);
        if (fbits != 0) {
            const u32 scale_factor = static_cast<u32>((127 - fbits) << 23);
            code.add_imm(Xscratch0, code.zero, scale_factor, Xscratch2);
            code.movgr2fr_d(Fscratch0, Xscratch0);

            code.fmul_s(Sto, Sto, Fscratch0);
        }
    });
}

template<>
void EmitIR<IR::Opcode::FPFixedS32ToSingle>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    // TODO: Consider fpr source
    EmitFromFixed<32, 32>(code, ctx, inst, [&](auto& Sto, auto& Wfrom, u8 fbits) {
        code.add_d(Wscratch0, code.zero, Wfrom);
        code.ffint_s_l(Sto, Wscratch0);
        if (fbits != 0) {
            const u32 scale_factor = static_cast<u32>((127 - fbits) << 23);
            code.add_imm(Wscratch0, code.zero, scale_factor, Wscratch2);
            code.movgr2fr_d(Fscratch0, Xscratch0);

            code.fmul_s(Sto, Sto, Fscratch0);
        }

    });
}

template<>
void EmitIR<IR::Opcode::FPFixedU32ToDouble>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<32, 64>(code, ctx, inst, [&](auto& Dto, auto& Wfrom, u8 fbits) {
        code.bstrpick_d(Xscratch0, Wfrom, 31, 0);
        code.ffint_d_w(Dto, Xscratch0);
        if (fbits != 0) {
            const u64 scale_factor = static_cast<u64>((1023ul - fbits) << 52);
            code.add_imm(Xscratch0, code.zero, scale_factor, Xscratch2);
            code.movgr2fr_d(Fscratch0, Xscratch0);
            code.fmul_d(Dto, Dto, Fscratch0);
        }

    });
}

template<>
void EmitIR<IR::Opcode::FPFixedS32ToDouble>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<32, 64>(code, ctx, inst, [&](auto& Dto, auto& Wfrom, u8 fbits) {
        code.add_d(Wscratch0, code.zero, Wfrom);
        code.ffint_d_w(Dto, Xscratch0);
        if (fbits != 0) {
            const u64 scale_factor = static_cast<u64>((1023ul - fbits) << 52);
            code.add_imm(Xscratch0, code.zero, scale_factor, Xscratch2);
            code.movgr2fr_d(Fscratch0, Xscratch0);
            code.fmul_d(Dto, Dto, Fscratch0);
        }
    });
}

template<>
void EmitIR<IR::Opcode::FPFixedU64ToDouble>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    // TODO: Consider fpr source
    EmitFromFixed<64, 64>(code, ctx, inst, [&](auto& Dto, auto& Xfrom, u8 fbits) {
        code.bstrpick_d(Xscratch0, Xfrom, 63, 0);
        code.ffint_d_l(Dto, Xscratch0);
        if (fbits != 0) {
            const u64 scale_factor = static_cast<u64>((1023ul - fbits) << 52);
            code.add_imm(Xscratch0, code.zero, scale_factor, Xscratch2);
            code.movgr2fr_d(Fscratch0, Xscratch0);
            code.fmul_d(Dto, Dto, Fscratch0);
        }
    });
}

template<>
void EmitIR<IR::Opcode::FPFixedU64ToSingle>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<64, 32>(code, ctx, inst, [&](auto& Sto, auto& Xfrom, u8 fbits) {
        code.bstrpick_d(Xscratch0, Xfrom, 63, 0);
        code.ffint_s_l(Sto, Xscratch0);
        if (fbits != 0) {
            const u64 scale_factor = static_cast<u64>((1023ul - fbits) << 52);
            code.add_imm(Xscratch0, code.zero, scale_factor, Xscratch2);
            code.movgr2fr_d(Fscratch0, Xscratch0);
            code.fmul_d(Sto, Sto, Fscratch0);
        }

    });
}

template<>
void EmitIR<IR::Opcode::FPFixedS64ToDouble>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    // TODO: Consider fpr source
    EmitFromFixed<64, 64>(code, ctx, inst, [&](auto& Dto, auto& Xfrom, u8 fbits) {
        code.add_d(Wscratch0, code.zero, Xfrom);
        code.ffint_d_w(Dto, Xscratch0);
        if (fbits != 0) {
            const u64 scale_factor = static_cast<u64>((1023ul - fbits) << 52);
            code.add_imm(Xscratch0, code.zero, scale_factor, Xscratch2);
            code.movgr2fr_d(Fscratch0, Xscratch0);
            code.fmul_d(Dto, Dto, Fscratch0);
        }
    });
}

template<>
void EmitIR<IR::Opcode::FPFixedS64ToSingle>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<64, 32>(code, ctx, inst, [&](auto& Sto, auto& Xfrom, u8 fbits) {
        code.add_d(Wscratch0, code.zero, Xfrom);
        code.ffint_s_w(Sto, Xscratch0);
        if (fbits != 0) {
            const u64 scale_factor = static_cast<u64>((1023ul - fbits) << 52);
            code.add_imm(Xscratch0, code.zero, scale_factor, Xscratch2);
            code.movgr2fr_d(Fscratch0, Xscratch0);
            code.fmul_s(Sto, Sto, Fscratch0);
        }
    });
}

}  // namespace Dynarmic::Backend::LoongArch64
