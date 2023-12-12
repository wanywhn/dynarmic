/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <mcl/bit_cast.hpp>
#include <mcl/mp/metavalue/lift_value.hpp>
#include <mcl/mp/typelist/cartesian_product.hpp>
#include <mcl/mp/typelist/get.hpp>
#include <mcl/mp/typelist/lift_sequence.hpp>
#include <mcl/mp/typelist/list.hpp>
#include <mcl/mp/typelist/lower_to_tuple.hpp>
#include <mcl/type_traits/function_info.hpp>
#include <mcl/type_traits/integer_of_size.hpp>

#include "dynarmic/backend/loongarch64/a32_jitstate.h"
#include "dynarmic/backend/loongarch64/a64_jitstate.h"
#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/emit_arm64.h"
#include "dynarmic/backend/loongarch64/emit_context.h"
#include "dynarmic/backend/loongarch64/fpsr_manager.h"
#include "dynarmic/backend/loongarch64/reg_alloc.h"
#include "dynarmic/common/always_false.h"
#include "dynarmic/common/cast_util.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/fp/fpsr.h"
#include "dynarmic/common/fp/info.h"
#include "dynarmic/common/fp/op.h"
#include "dynarmic/common/fp/rounding_mode.h"
#include "dynarmic/common/lut_from_list.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"
#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"

namespace Dynarmic::Backend::LoongArch64 {

    using namespace Xbyak_loongarch64::util;
    namespace mp = mcl::mp;

    using A64FullVectorWidth = std::integral_constant<size_t, 128>;

// Array alias that always sizes itself according to the given type T
// relative to the size of a vector register. e.g. T = u32 would result
// in a std::array<u32, 4>.
    template<typename T>
    using VectorArray = std::array<T, A64FullVectorWidth::value / mcl::bitsizeof<T>>;

    template<typename EmitFn>
    static void MaybeStandardFPSCRValue(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, bool fpcr_controlled,
                                        EmitFn emit) {
        if (ctx.FPCR(fpcr_controlled) != ctx.FPCR()) {
            code.add_imm(Wscratch0, code.zero, ctx.FPCR(fpcr_controlled).Value(), Wscratch1);
            code.MSR(Xbyak_loongarch64::SystemReg::FPCR, Xscratch0);
            emit();
            code.add_d(Wscratch0, ctx.FPCR().Value(), code.zero);
            code.MSR(Xbyak_loongarch64::SystemReg::FPCR, Xscratch0);
        } else {
            emit();
        }
    }

    template<typename EmitFn>
    static void EmitTwoOp(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        auto Qa = ctx.reg_alloc.ReadQ(args[0]);
        const bool fpcr_controlled = args[1].IsVoid() || args[1].GetImmediateU1();
        RegAlloc::Realize(Qresult, Qa);
        ctx.fpsr.Load();

        MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] { emit(Qresult, Qa); });
    }

    template<size_t size, typename EmitFn>
    static void
    EmitTwoOpArranged(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        EmitTwoOp(code, ctx, inst, [&](auto &Qresult, auto &Qa) {
            if constexpr (size == 16) {
                emit(Qresult->H8(), Qa->H8());
            } else if constexpr (size == 32) {
                emit(Qresult->S4(), Qa->S4());
            } else if constexpr (size == 64) {
                emit(Qresult->D2(), Qa->D2());
            } else {
                static_assert(Common::always_false_v<mcl::mp::lift_value<size>>);
            }
        });
    }

    template<typename EmitFn>
    static void EmitThreeOp(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        auto Qa = ctx.reg_alloc.ReadQ(args[0]);
        auto Qb = ctx.reg_alloc.ReadQ(args[1]);
        const bool fpcr_controlled = args[2].GetImmediateU1();
        RegAlloc::Realize(Qresult, Qa, Qb);
        ctx.fpsr.Load();

        MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] { emit(Qresult, Qa, Qb); });
    }

    template<size_t size, typename EmitFn>
    static void
    EmitThreeOpArranged(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        EmitThreeOp(code, ctx, inst, [&](auto &Qresult, auto &Qa, auto &Qb) {
            if constexpr (size == 16) {
                emit(Qresult->H8(), Qa->H8(), Qb->H8());
            } else if constexpr (size == 32) {
                emit(Qresult->S4(), Qa->S4(), Qb->S4());
            } else if constexpr (size == 64) {
                emit(Qresult->D2(), Qa->D2(), Qb->D2());
            } else {
                static_assert(Common::always_false_v<mcl::mp::lift_value<size>>);
            }
        });
    }

    template<size_t size, typename EmitFn>
    static void EmitFMA(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qresult = ctx.reg_alloc.ReadWriteQ(args[0], inst);
        auto Qm = ctx.reg_alloc.ReadQ(args[1]);
        auto Qn = ctx.reg_alloc.ReadQ(args[2]);
        const bool fpcr_controlled = args[3].GetImmediateU1();
        RegAlloc::Realize(Qresult, Qm, Qn);
        ctx.fpsr.Load();

        MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] {
            if constexpr (size == 16) {
                emit(Qresult->H8(), Qm->H8(), Qn->H8());
            } else if constexpr (size == 32) {
                emit(Qresult->S4(), Qm->S4(), Qn->S4());
            } else if constexpr (size == 64) {
                emit(Qresult->D2(), Qm->D2(), Qn->D2());
            } else {
                static_assert(Common::always_false_v<mcl::mp::lift_value<size>>);
            }
        });
    }

    template<size_t size, typename EmitFn>
    static void EmitFromFixed(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qto = ctx.reg_alloc.WriteQ(inst);
        auto Qfrom = ctx.reg_alloc.ReadQ(args[0]);
        const u8 fbits = args[1].GetImmediateU8();
        const FP::RoundingMode rounding_mode = static_cast<FP::RoundingMode>(args[2].GetImmediateU8());
        const bool fpcr_controlled = args[3].GetImmediateU1();
        ASSERT(rounding_mode == ctx.FPCR(fpcr_controlled).RMode());
        RegAlloc::Realize(Qto, Qfrom);

        MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] {
            if constexpr (size == 32) {
                emit(Qto->S4(), Qfrom->S4(), fbits);
            } else if constexpr (size == 64) {
                emit(Qto->D2(), Qfrom->D2(), fbits);
            } else {
                static_assert(Common::always_false_v<mcl::mp::lift_value<size>>);
            }
        });
    }

    template<size_t fsize, bool is_signed>
    void EmitToFixed(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qto = ctx.reg_alloc.WriteQ(inst);
        auto Qfrom = ctx.reg_alloc.ReadQ(args[0]);
        const size_t fbits = args[1].GetImmediateU8();
        const auto rounding_mode = static_cast<FP::RoundingMode>(args[2].GetImmediateU8());
        const bool fpcr_controlled = inst->GetArg(3).GetU1();
        RegAlloc::Realize(Qto, Qfrom);
        ctx.fpsr.Load();

        auto Vto = [&] {
            if constexpr (fsize == 32) {
                return Qto->S4();
            } else if constexpr (fsize == 64) {
                return Qto->D2();
            } else {
                static_assert(Common::always_false_v<mcl::mp::lift_value<fsize>>);
            }
        }();
        auto Vfrom = [&] {
            if constexpr (fsize == 32) {
                return Qfrom->S4();
            } else if constexpr (fsize == 64) {
                return Qfrom->D2();
            } else {
                static_assert(Common::always_false_v<mcl::mp::lift_value<fsize>>);
            }
        }();

        MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] {
            if (rounding_mode == FP::RoundingMode::TowardsZero) {
                if constexpr (is_signed) {
                    if (fbits) {
                        code.FCVTZS(Vto, Vfrom, fbits);
                    } else {
                        code.FCVTZS(Vto, Vfrom);
                    }
                } else {
                    if (fbits) {
                        code.FCVTZU(Vto, Vfrom, fbits);
                    } else {
                        code.FCVTZU(Vto, Vfrom);
                    }
                }
            } else {
                ASSERT(fbits == 0);
                if constexpr (is_signed) {
                    switch (rounding_mode) {
                        case FP::RoundingMode::ToNearest_TieEven:
                            code.FCVTNS(Vto, Vfrom);
                            break;
                        case FP::RoundingMode::TowardsPlusInfinity:
                            code.FCVTPS(Vto, Vfrom);
                            break;
                        case FP::RoundingMode::TowardsMinusInfinity:
                            code.FCVTMS(Vto, Vfrom);
                            break;
                        case FP::RoundingMode::TowardsZero:
                            code.FCVTZS(Vto, Vfrom);
                            break;
                        case FP::RoundingMode::ToNearest_TieAwayFromZero:
                            code.FCVTAS(Vto, Vfrom);
                            break;
                        case FP::RoundingMode::ToOdd:
                            ASSERT_FALSE("Unimplemented");
                            break;
                        default:
                            ASSERT_FALSE("Invalid RoundingMode");
                            break;
                    }
                } else {
                    switch (rounding_mode) {
                        case FP::RoundingMode::ToNearest_TieEven:
                            code.FCVTNU(Vto, Vfrom);
                            break;
                        case FP::RoundingMode::TowardsPlusInfinity:
                            code.FCVTPU(Vto, Vfrom);
                            break;
                        case FP::RoundingMode::TowardsMinusInfinity:
                            code.FCVTMU(Vto, Vfrom);
                            break;
                        case FP::RoundingMode::TowardsZero:
                            code.FCVTZU(Vto, Vfrom);
                            break;
                        case FP::RoundingMode::ToNearest_TieAwayFromZero:
                            code.FCVTAU(Vto, Vfrom);
                            break;
                        case FP::RoundingMode::ToOdd:
                            ASSERT_FALSE("Unimplemented");
                            break;
                        default:
                            ASSERT_FALSE("Invalid RoundingMode");
                            break;
                    }
                }
            }
        });
    }

    template<typename Lambda>
    static void EmitTwoOpFallbackWithoutRegAlloc(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                 Xbyak_loongarch64::VReg Qresult, Xbyak_loongarch64::VReg Qarg1,
                                                 Lambda lambda, bool fpcr_controlled) {
        const auto fn = static_cast<mcl::equivalent_function_type<Lambda> *>(lambda);

        const u32 fpcr = ctx.FPCR(fpcr_controlled).Value();
        constexpr u64 stack_size = sizeof(u64) * 4;  // sizeof(u128) * 2

        ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Qresult.getIdx()), stack_size);

        code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(fn), Xscratch1);
        code.add_imm(code.a0, code.sp, 0 * 16, Xscratch1);
        code.add_imm(code.a1, code.sp, 1 * 16, Xscratch1);
        code.add_d(code.a2, fpcr, code.zero);
        code.add_imm(X3, Xstate, ctx.conf.state_fpsr_offset, Xscratch1);
        code.stx_d(Qarg1, code.a1, code.zero);
        code.jirl(code.ra, Xscratch0, 0);
        code.ld_d(Qresult, SP, 0);

        ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << Qresult.getIdx()), stack_size);
    }

    template<size_t fpcr_controlled_arg_index = 1, typename Lambda>
    static void
    EmitTwoOpFallback(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, Lambda lambda) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qarg1 = ctx.reg_alloc.ReadQ(args[0]);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(Qarg1, Qresult);
        ctx.reg_alloc.SpillFlags();
        ctx.fpsr.Spill();

        const bool fpcr_controlled = args[fpcr_controlled_arg_index].GetImmediateU1();
        EmitTwoOpFallbackWithoutRegAlloc(code, ctx, Qresult, Qarg1, lambda, fpcr_controlled);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorAbs16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qresult = ctx.reg_alloc.ReadWriteQ(args[0], inst);
        RegAlloc::Realize(Qresult);

        code.vbitclri_h(Qresult, Qresult, 15);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorAbs32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va) { code.vbitclri_w(Vresult, Va, 31); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorAbs64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va) { code.vbitclri_d(Vresult, Va, 63); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorAdd32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfadd_s(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorAdd64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfadd_d(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorDiv32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfdiv_s(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorDiv64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfdiv_d(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorEqual16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorEqual32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfcmp_cond_s(Vresult, Va, Vb, 0x4); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorEqual64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfcmp_cond_d(Vresult, Va, Vb, 0x4); });
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorFromHalf32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        const auto rounding_mode = static_cast<FP::RoundingMode>(args[1].GetImmediateU8());
        ASSERT(rounding_mode == FP::RoundingMode::ToNearest_TieEven);
        const bool fpcr_controlled = args[2].GetImmediateU1();

        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        auto Doperand = ctx.reg_alloc.ReadD(args[0]);
        RegAlloc::Realize(Qresult, Doperand);
        ctx.fpsr.Load();

        MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] {
            code.FCVTL(Qresult->S4(), Doperand->H4());
        });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorFromSignedFixed32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitFromFixed<32>(code, ctx, inst, [&](auto Vto, auto Vfrom, u8 fbits) {
            fbits ? code.vffint_s_w(Vto, Vfrom, fbits) : code.SCVTF(Vto, Vfrom);
        });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorFromSignedFixed64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitFromFixed<64>(code, ctx, inst, [&](auto Vto, auto Vfrom, u8 fbits) {
            fbits ? code.SCVTF(Vto, Vfrom, fbits) : code.SCVTF(Vto, Vfrom);
        });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorFromUnsignedFixed32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitFromFixed<32>(code, ctx, inst, [&](auto Vto, auto Vfrom, u8 fbits) {
            fbits ? code.UCVTF(Vto, Vfrom, fbits) : code.UCVTF(Vto, Vfrom);
        });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorFromUnsignedFixed64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitFromFixed<64>(code, ctx, inst, [&](auto Vto, auto Vfrom, u8 fbits) {
            fbits ? code.UCVTF(Vto, Vfrom, fbits) : code.UCVTF(Vto, Vfrom);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorGreater32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfcmp_cond_s(Vresult, Vb, Va, 0x2); });
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorGreater64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfcmp_cond_d(Vresult, Vb, Va, 0x2); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorGreaterEqual32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfcmp_cond_s(Vresult, Vb, Va, 0x6); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorGreaterEqual64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfcmp_cond_d(Vresult, Vb, Va, 0x6); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorMax32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfmax_s(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorMax64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfmax_d(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorMaxNumeric32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfmaxa_s(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorMaxNumeric64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfmaxa_d(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorMin32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfmin_s(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorMin64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfmin_d(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorMinNumeric32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfmina_s(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorMinNumeric64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfmina_d(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorMul32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfmul_s(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorMul64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vfmul_d(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorMulAdd16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorMulAdd32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFMA<32>(code, ctx, inst, [&](auto Va, auto Vn, auto Vm) { code.vfmadd_s(Va, Vn, Vm); });
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorMulAdd64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFMA<64>(code, ctx, inst, [&](auto Va, auto Vn, auto Vm) { code.vfmadd_d(Va, Vn, Vm); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorMulX32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.FMULX(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorMulX64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.FMULX(Vresult, Va, Vb); });
    }

    template<size_t fsize>
    void FPVectorNeg(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qa = ctx.reg_alloc.ReadWriteQ(args[0], inst);
        using FPT = mcl::unsigned_integer_of_size<fsize>;
        constexpr FPT sign_mask = FP::FPInfo<FPT>::sign_mask;
        constexpr u64 sign_mask64 = mcl::bit::replicate_element<fsize, u64>(sign_mask);

        if constexpr (fsize == 64) {
            code.add_imm(Xscratch0, code.zero, sign_mask64, Xscratch1);
            code.vinsgr2vr_d(code.vr0, Xscratch0 ,0);
            code.vxor_v(Qa, Qa, code.vr0);
        } else if constexpr (fsize == 32) {
            code.add_imm(Xscratch0, code.zero, sign_mask64, Xscratch1);
            code.vinsgr2vr_w(code.vr0, Xscratch0 ,0);
            code.vxor_v(Qa, Qa, code.vr0);
        } else if constexpr (fsize == 16) {
            code.add_imm(Xscratch0, code.zero, sign_mask64, Xscratch1);
            code.vinsgr2vr_h(code.vr0, Xscratch0 ,0);
            code.vxor_v(Qa, Qa, code.vr0);
        }


//        ctx.reg_alloc.DefineValue(inst, a);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorNeg16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        FPVectorNeg<16>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorNeg32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        FPVectorNeg<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorNeg64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        FPVectorNeg<64>(code, ctx, inst);
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorPairedAdd32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_w(Vresult, Vb, Va);
            code.vpickod_w(Vc, Vb, Va);
            code.vfadd_w(Vresult, Vresult, Vc);
//            code.vinsgr2vr_d(Vresult, code.zero , 1);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorPairedAdd64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_d(Vresult, Vb, Va);
            code.vpickod_d(Vc, Vb, Va);
            code.vfadd_d(Vresult, Vresult, Vc);
//            code.vinsgr2vr_d(Vresult, code.zero , 1);
//            code.FADDP(Vresult, Va, Vb);
        });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorPairedAddLower32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto &Qresult, auto &Qa, auto &Qb) {
            code.vfadd_s(Qresult, Qa, Qb);
            code.vinsgr2vr_d(Qresult, code.zero ,1);
//            code.ZIP1(V0.D2(), Qa->D2(), Qb->D2());
//            code.MOVI(D1, Xbyak_loongarch64::RepImm{0});
//            code.FADDP(Qresult->S4(), V0.S4(), V1.S4());
        });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorPairedAddLower64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto &Qresult, auto &Qa, auto &Qb) {
            code.vfadd_d(Qresult, Qa, Qb);
            code.vinsgr2vr_d(Qresult, code.zero ,1);
//            code.ZIP1(V0.D2(), Qa->D2(), Qb->D2());
//            code.FADDP(Qresult->toD(), V0.D2());
        });
    }

    template<size_t fsize>
    static void EmitRecipEstimate(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
        using FPT = mcl::unsigned_integer_of_size<fsize>;

        if constexpr (fsize != 16) {
            if (ctx.HasOptimization(OptimizationFlag::Unsafe_ReducedErrorFP)) {
                auto args = ctx.reg_alloc.GetArgumentInfo(inst);
                const Xbyak::Xmm operand = ctx.reg_alloc.UseXmm(args[0]);
                const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();

                if (code.HasHostFeature(HostFeature::AVX512_OrthoFloat)) {
                    FCODE(vrcp14p)(result, operand);
                } else {
                    if constexpr (fsize == 32) {
                        code.rcpps(result, operand);
                    } else {
                        code.cvtpd2ps(result, operand);
                        code.rcpps(result, result);
                        code.cvtps2pd(result, result);
                    }
                }

                ctx.reg_alloc.DefineValue(inst, result);
                return;
            }
        }

        EmitTwoOpFallback(code, ctx, inst, [](VectorArray<FPT>& result, const VectorArray<FPT>& operand, FP::FPCR fpcr, FP::FPSR& fpsr) {
            for (size_t i = 0; i < result.size(); i++) {
                result[i] = FP::FPRecipEstimate<FPT>(operand[i], fpcr, fpsr);
            }
        });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRecipEstimate16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitRecipEstimate<16>(code, ctx, inst);

    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRecipEstimate32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitRecipEstimate<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRecipEstimate64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitRecipEstimate<64>(code, ctx, inst);
    }


    template<size_t fsize>
    static void EmitRecipStepFused(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
        using FPT = mcl::unsigned_integer_of_size<fsize>;

        const auto fallback_fn = [](VectorArray<FPT>& result, const VectorArray<FPT>& op1, const VectorArray<FPT>& op2, FP::FPCR fpcr, FP::FPSR& fpsr) {
            for (size_t i = 0; i < result.size(); i++) {
                result[i] = FP::FPRecipStepFused<FPT>(op1[i], op2[i], fpcr, fpsr);
            }
        };

        if constexpr (fsize != 16) {
            if (code.HasHostFeature(HostFeature::FMA | HostFeature::AVX) && ctx.HasOptimization(OptimizationFlag::Unsafe_InaccurateNaN)) {
                auto args = ctx.reg_alloc.GetArgumentInfo(inst);
                const bool fpcr_controlled = args[2].GetImmediateU1();

                const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
                const Xbyak::Xmm operand1 = ctx.reg_alloc.UseXmm(args[0]);
                const Xbyak::Xmm operand2 = ctx.reg_alloc.UseXmm(args[1]);

                MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] {
                    code.movaps(result, GetVectorOf<fsize, false, 0, 2>(code));
                    FCODE(vfnmadd231p)(result, operand1, operand2);
                });

                ctx.reg_alloc.DefineValue(inst, result);
                return;
            }

            if (code.HasHostFeature(HostFeature::FMA | HostFeature::AVX)) {
                auto args = ctx.reg_alloc.GetArgumentInfo(inst);
                const bool fpcr_controlled = args[2].GetImmediateU1();

                const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
                const Xbyak::Xmm operand1 = ctx.reg_alloc.UseXmm(args[0]);
                const Xbyak::Xmm operand2 = ctx.reg_alloc.UseXmm(args[1]);
                const Xbyak::Xmm tmp = ctx.reg_alloc.ScratchXmm();

                SharedLabel end = GenSharedLabel(), fallback = GenSharedLabel();

                MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] {
                    code.movaps(result, GetVectorOf<fsize, false, 0, 2>(code));
                    FCODE(vfnmadd231p)(result, operand1, operand2);

                    FCODE(vcmpunordp)(tmp, result, result);
                    code.vptest(tmp, tmp);
                    code.jnz(*fallback, code.T_NEAR);
                    code.L(*end);
                });

                ctx.deferred_emits.emplace_back([=, &code, &ctx] {
                    code.L(*fallback);
                    code.sub(rsp, 8);
                    ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLocXmmIdx(result.getIdx()));
                    EmitThreeOpFallbackWithoutRegAlloc(code, ctx, result, operand1, operand2, fallback_fn, fpcr_controlled);
                    ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLocXmmIdx(result.getIdx()));
                    code.add(rsp, 8);
                    code.jmp(*end, code.T_NEAR);
                });

                ctx.reg_alloc.DefineValue(inst, result);
                return;
            }

            if (ctx.HasOptimization(OptimizationFlag::Unsafe_UnfuseFMA)) {
                auto args = ctx.reg_alloc.GetArgumentInfo(inst);

                const Xbyak::Xmm operand1 = ctx.reg_alloc.UseScratchXmm(args[0]);
                const Xbyak::Xmm operand2 = ctx.reg_alloc.UseXmm(args[1]);
                const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();

                code.movaps(result, GetVectorOf<fsize, false, 0, 2>(code));
                FCODE(mulp)(operand1, operand2);
                FCODE(subp)(result, operand1);

                ctx.reg_alloc.DefineValue(inst, result);
                return;
            }
        }

        EmitThreeOpFallback(code, ctx, inst, fallback_fn);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRecipStepFused16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitRecipStepFused<16>(code, ctx, inst);

    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRecipStepFused32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitRecipStepFused<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRecipStepFused64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitRecipStepFused<64>(code, ctx, inst);
    }

    template<size_t fsize>
    void EmitFPVectorRoundInt(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
        const auto rounding = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());
        const bool exact = inst->GetArg(2).GetU1();

        if constexpr (fsize != 16) {
            if (code.HasHostFeature(HostFeature::SSE41) && rounding != FP::RoundingMode::ToNearest_TieAwayFromZero && !exact) {
                const u8 round_imm = [&]() -> u8 {
                    switch (rounding) {
                        case FP::RoundingMode::ToNearest_TieEven:
                            return 0b00;
                        case FP::RoundingMode::TowardsPlusInfinity:
                            return 0b10;
                        case FP::RoundingMode::TowardsMinusInfinity:
                            return 0b01;
                        case FP::RoundingMode::TowardsZero:
                            return 0b11;
                        default:
                            UNREACHABLE();
                    }
                }();

                EmitTwoOpVectorOperation<fsize, DefaultIndexer, 3>(code, ctx, inst, [&](const Xbyak::Xmm& result, const Xbyak::Xmm& xmm_a) {
                    FCODE(roundp)(result, xmm_a, round_imm);
                });

                return;
            }
        }

        using rounding_list = mp::list<
                mp::lift_value<FP::RoundingMode::ToNearest_TieEven>,
                mp::lift_value<FP::RoundingMode::TowardsPlusInfinity>,
                mp::lift_value<FP::RoundingMode::TowardsMinusInfinity>,
                mp::lift_value<FP::RoundingMode::TowardsZero>,
                mp::lift_value<FP::RoundingMode::ToNearest_TieAwayFromZero>>;
        using exact_list = mp::list<std::true_type, std::false_type>;

        static const auto lut = Common::GenerateLookupTableFromList(
                []<typename I>(I) {
                    using FPT = mcl::unsigned_integer_of_size<fsize>;  // WORKAROUND: For issue 678 on MSVC
                    return std::pair{
                            mp::lower_to_tuple_v<I>,
                            Common::FptrCast(
                                    [](VectorArray<FPT>& output, const VectorArray<FPT>& input, FP::FPCR fpcr, FP::FPSR& fpsr) {
                                        constexpr FP::RoundingMode rounding_mode = mp::get<0, I>::value;
                                        constexpr bool exact = mp::get<1, I>::value;

                                        for (size_t i = 0; i < output.size(); ++i) {
                                            output[i] = static_cast<FPT>(FP::FPRoundInt<FPT>(input[i], fpcr, rounding_mode, exact, fpsr));
                                        }
                                    })};
                },
                mp::cartesian_product<rounding_list, exact_list>{});

        EmitTwoOpFallback<3>(code, ctx, inst, lut.at(std::make_tuple(rounding, exact)));
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorRoundInt16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFPVectorRoundInt<16>(code, ctx, inst);

    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorRoundInt32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFPVectorRoundInt<32>(code, ctx, inst);

    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorRoundInt64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
//        const auto rounding_mode = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());
//        const bool exact = inst->GetArg(2).GetU1();
//        const bool fpcr_controlled = inst->GetArg(3).GetU1();
//
//        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
//        auto Qresult = ctx.reg_alloc.WriteQ(inst);
//        auto Qoperand = ctx.reg_alloc.ReadQ(args[0]);
//        RegAlloc::Realize(Qresult, Qoperand);
//        ctx.fpsr.Load();

        EmitFPVectorRoundInt<64>(code, ctx, inst);

    }

    template<size_t fsize>
    static void EmitRSqrtEstimate(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
        using FPT = mcl::unsigned_integer_of_size<fsize>;

        const auto fallback_fn = [](VectorArray<FPT>& result, const VectorArray<FPT>& operand, FP::FPCR fpcr, FP::FPSR& fpsr) {
            for (size_t i = 0; i < result.size(); i++) {
                result[i] = FP::FPRSqrtEstimate<FPT>(operand[i], fpcr, fpsr);
            }
        };

        if constexpr (fsize != 16) {
            if (ctx.HasOptimization(OptimizationFlag::Unsafe_ReducedErrorFP)) {
                auto args = ctx.reg_alloc.GetArgumentInfo(inst);
                const Xbyak::Xmm operand = ctx.reg_alloc.UseXmm(args[0]);
                const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();

                if (code.HasHostFeature(HostFeature::AVX512_OrthoFloat)) {
                    FCODE(vrsqrt14p)(result, operand);
                } else {
                    if constexpr (fsize == 32) {
                        code.rsqrtps(result, operand);
                    } else {
                        code.cvtpd2ps(result, operand);
                        code.rsqrtps(result, result);
                        code.cvtps2pd(result, result);
                    }
                }

                ctx.reg_alloc.DefineValue(inst, result);
                return;
            }

            if (code.HasHostFeature(HostFeature::AVX)) {
                auto args = ctx.reg_alloc.GetArgumentInfo(inst);
                const bool fpcr_controlled = args[1].GetImmediateU1();

                const Xbyak::Xmm operand = ctx.reg_alloc.UseXmm(args[0]);
                const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
                const Xbyak::Xmm value = ctx.reg_alloc.ScratchXmm();

                SharedLabel bad_values = GenSharedLabel(), end = GenSharedLabel();

                code.movaps(value, operand);

                code.movaps(xmm0, GetVectorOf<fsize, (fsize == 32 ? 0xFFFF8000 : 0xFFFF'F000'0000'0000)>(code));
                code.pand(value, xmm0);
                code.por(value, GetVectorOf<fsize, (fsize == 32 ? 0x00008000 : 0x0000'1000'0000'0000)>(code));

                // Detect NaNs, negatives, zeros, denormals and infinities
                FCODE(vcmpnge_uqp)(result, value, GetVectorOf<fsize, (FPT(1) << FP::FPInfo<FPT>::explicit_mantissa_width)>(code));
                code.vptest(result, result);
                code.jnz(*bad_values, code.T_NEAR);

                FCODE(sqrtp)(value, value);
                code.vmovaps(result, GetVectorOf<fsize, FP::FPValue<FPT, false, 0, 1>()>(code));
                FCODE(divp)(result, value);

                ICODE(padd)(result, GetVectorOf<fsize, (fsize == 32 ? 0x00004000 : 0x0000'0800'0000'0000)>(code));
                code.pand(result, xmm0);

                code.L(*end);

                ctx.deferred_emits.emplace_back([=, &code, &ctx] {
                    code.L(*bad_values);
                    code.sub(rsp, 8);
                    ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLocXmmIdx(result.getIdx()));
                    EmitTwoOpFallbackWithoutRegAlloc(code, ctx, result, operand, fallback_fn, fpcr_controlled);
                    ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLocXmmIdx(result.getIdx()));
                    code.add(rsp, 8);
                    code.jmp(*end, code.T_NEAR);
                });

                ctx.reg_alloc.DefineValue(inst, result);
                return;
            }
        }

        EmitTwoOpFallback(code, ctx, inst, fallback_fn);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRSqrtEstimate16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitRSqrtEstimate<16>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRSqrtEstimate32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitRSqrtEstimate<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRSqrtEstimate64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitRSqrtEstimate<64>(code, ctx, inst);
    }

    template<size_t fsize>
    static void EmitRSqrtStepFused(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
        using FPT = mcl::unsigned_integer_of_size<fsize>;

        const auto fallback_fn = [](VectorArray<FPT>& result, const VectorArray<FPT>& op1, const VectorArray<FPT>& op2, FP::FPCR fpcr, FP::FPSR& fpsr) {
            for (size_t i = 0; i < result.size(); i++) {
                result[i] = FP::FPRSqrtStepFused<FPT>(op1[i], op2[i], fpcr, fpsr);
            }
        };

        if constexpr (fsize != 16) {
            if (code.HasHostFeature(HostFeature::FMA | HostFeature::AVX) && ctx.HasOptimization(OptimizationFlag::Unsafe_InaccurateNaN)) {
                auto args = ctx.reg_alloc.GetArgumentInfo(inst);
                const bool fpcr_controlled = args[2].GetImmediateU1();

                const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
                const Xbyak::Xmm operand1 = ctx.reg_alloc.UseXmm(args[0]);
                const Xbyak::Xmm operand2 = ctx.reg_alloc.UseXmm(args[1]);

                MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] {
                    code.vmovaps(result, GetVectorOf<fsize, false, 0, 3>(code));
                    FCODE(vfnmadd231p)(result, operand1, operand2);
                    FCODE(vmulp)(result, result, GetVectorOf<fsize, false, -1, 1>(code));
                });

                ctx.reg_alloc.DefineValue(inst, result);
                return;
            }

            if (code.HasHostFeature(HostFeature::FMA | HostFeature::AVX)) {
                auto args = ctx.reg_alloc.GetArgumentInfo(inst);
                const bool fpcr_controlled = args[2].GetImmediateU1();

                const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
                const Xbyak::Xmm operand1 = ctx.reg_alloc.UseXmm(args[0]);
                const Xbyak::Xmm operand2 = ctx.reg_alloc.UseXmm(args[1]);
                const Xbyak::Xmm tmp = ctx.reg_alloc.ScratchXmm();
                const Xbyak::Xmm mask = ctx.reg_alloc.ScratchXmm();

                SharedLabel end = GenSharedLabel(), fallback = GenSharedLabel();

                MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] {
                    code.vmovaps(result, GetVectorOf<fsize, false, 0, 3>(code));
                    FCODE(vfnmadd231p)(result, operand1, operand2);

                    // An explanation for this is given in EmitFPRSqrtStepFused.
                    code.vmovaps(mask, GetVectorOf<fsize, (fsize == 32 ? 0x7f000000 : 0x7fe0000000000000)>(code));
                    FCODE(vandp)(tmp, result, mask);
                    ICODE(vpcmpeq)(tmp, tmp, mask);
                    code.ptest(tmp, tmp);
                    code.jnz(*fallback, code.T_NEAR);

                    FCODE(vmulp)(result, result, GetVectorOf<fsize, false, -1, 1>(code));
                    code.L(*end);
                });

                ctx.deferred_emits.emplace_back([=, &code, &ctx] {
                    code.L(*fallback);
                    code.sub(rsp, 8);
                    ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLocXmmIdx(result.getIdx()));
                    EmitThreeOpFallbackWithoutRegAlloc(code, ctx, result, operand1, operand2, fallback_fn, fpcr_controlled);
                    ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLocXmmIdx(result.getIdx()));
                    code.add(rsp, 8);
                    code.jmp(*end, code.T_NEAR);
                });

                ctx.reg_alloc.DefineValue(inst, result);
                return;
            }

            if (ctx.HasOptimization(OptimizationFlag::Unsafe_UnfuseFMA)) {
                auto args = ctx.reg_alloc.GetArgumentInfo(inst);

                const Xbyak::Xmm operand1 = ctx.reg_alloc.UseScratchXmm(args[0]);
                const Xbyak::Xmm operand2 = ctx.reg_alloc.UseXmm(args[1]);
                const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();

                code.movaps(result, GetVectorOf<fsize, false, 0, 3>(code));
                FCODE(mulp)(operand1, operand2);
                FCODE(subp)(result, operand1);
                FCODE(mulp)(result, GetVectorOf<fsize, false, -1, 1>(code));

                ctx.reg_alloc.DefineValue(inst, result);
                return;
            }
        }

        EmitThreeOpFallback(code, ctx, inst, fallback_fn);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRSqrtStepFused16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitRSqrtStepFused<16>(code, ctx, inst);

    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRSqrtStepFused32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitRSqrtStepFused<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRSqrtStepFused64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitRSqrtStepFused<64>(code, ctx, inst);

    }

    template<>
    void EmitIR<IR::Opcode::FPVectorSqrt32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va) { code.vfsqrt_s(Vresult, Va); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorSqrt64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va) { code.vfsqrt_d(Vresult, Va); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorSub32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vfsub_s(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorSub64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vfsub_d(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorToHalf32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        const auto rounding_mode = static_cast<FP::RoundingMode>(args[1].GetImmediateU8());
        ASSERT(rounding_mode == FP::RoundingMode::ToNearest_TieEven);
        const bool fpcr_controlled = args[2].GetImmediateU1();

//        const auto rounding_mode = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());
//        const bool fpcr_controlled = inst->GetArg(2).GetU1();

        if (code.HasHostFeature(HostFeature::F16C) && !ctx.FPCR().AHP() && !ctx.FPCR().FZ16()) {
            auto args = ctx.reg_alloc.GetArgumentInfo(inst);
            const auto round_imm = ConvertRoundingModeToX64Immediate(rounding_mode);

            const Xbyak::Xmm result = ctx.reg_alloc.UseScratchXmm(args[0]);

            ForceToDefaultNaN<32>(code, ctx.FPCR(fpcr_controlled), result);
            code.vcvtps2ph(result, result, static_cast<u8>(*round_imm));

            ctx.reg_alloc.DefineValue(inst, result);
            return;
        }

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
                                    [](VectorArray<u16>& output, const VectorArray<u32>& input, FP::FPCR fpcr, FP::FPSR& fpsr) {
                                        constexpr FP::RoundingMode rounding_mode = mp::get<0, I>::value;

                                        for (size_t i = 0; i < output.size(); ++i) {
                                            if (i < input.size()) {
                                                output[i] = FP::FPConvert<u16, u32>(input[i], fpcr, rounding_mode, fpsr);
                                            } else {
                                                output[i] = 0;
                                            }
                                        }
                                    })};
                },
                mp::cartesian_product<rounding_list>{});

        EmitTwoOpFallback<2>(code, ctx, inst, lut.at(std::make_tuple(rounding_mode)));
    }


    template<size_t fsize, bool unsigned_>
    void EmitFPVectorToFixed(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
        const size_t fbits = inst->GetArg(1).GetU8();
        const auto rounding = static_cast<FP::RoundingMode>(inst->GetArg(2).GetU8());
        [[maybe_unused]] const bool fpcr_controlled = inst->GetArg(3).GetU1();

        if constexpr (fsize != 16) {
            if (code.HasHostFeature(HostFeature::SSE41) && rounding != FP::RoundingMode::ToNearest_TieAwayFromZero) {
                auto args = ctx.reg_alloc.GetArgumentInfo(inst);

                const Xbyak::Xmm src = ctx.reg_alloc.UseScratchXmm(args[0]);

                MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] {
                    const int round_imm = [&] {
                        switch (rounding) {
                            case FP::RoundingMode::ToNearest_TieEven:
                            default:
                                return 0b00;
                            case FP::RoundingMode::TowardsPlusInfinity:
                                return 0b10;
                            case FP::RoundingMode::TowardsMinusInfinity:
                                return 0b01;
                            case FP::RoundingMode::TowardsZero:
                                return 0b11;
                        }
                    }();

                    const auto perform_conversion = [&code, &ctx](const Xbyak::Xmm& src) {
                        // MSVC doesn't allow us to use a [&] capture, so we have to do this instead.
                        (void)ctx;

                        if constexpr (fsize == 32) {
                            code.cvttps2dq(src, src);
                        } else {
                            if (code.HasHostFeature(HostFeature::AVX512_OrthoFloat)) {
                                code.vcvttpd2qq(src, src);
                            } else {
                                const Xbyak::Reg64 hi = ctx.reg_alloc.ScratchGpr();
                                const Xbyak::Reg64 lo = ctx.reg_alloc.ScratchGpr();

                                code.cvttsd2si(lo, src);
                                code.punpckhqdq(src, src);
                                code.cvttsd2si(hi, src);
                                code.movq(src, lo);
                                code.pinsrq(src, hi, 1);

                                ctx.reg_alloc.Release(hi);
                                ctx.reg_alloc.Release(lo);
                            }
                        }
                    };

                    if (fbits != 0) {
                        const u64 scale_factor = fsize == 32
                                                 ? static_cast<u64>(fbits + 127) << 23
                                                 : static_cast<u64>(fbits + 1023) << 52;
                        FCODE(mulp)(src, GetVectorOf<fsize>(code, scale_factor));
                    }

                    FCODE(roundp)(src, src, static_cast<u8>(round_imm));
                    ZeroIfNaN<fsize>(code, src);

                    constexpr u64 float_upper_limit_signed = fsize == 32 ? 0x4f000000 : 0x43e0000000000000;
                    [[maybe_unused]] constexpr u64 float_upper_limit_unsigned = fsize == 32 ? 0x4f800000 : 0x43f0000000000000;

                    if constexpr (unsigned_) {
                        if (code.HasHostFeature(HostFeature::AVX512_OrthoFloat)) {
                            // Mask positive values
                            code.xorps(xmm0, xmm0);
                            FCODE(vcmpp)(k1, src, xmm0, Cmp::GreaterEqual_OQ);

                            // Convert positive values to unsigned integers, write 0 anywhere else
                            // vcvttp*2u*q already saturates out-of-range values to (0xFFFF...)
                            if constexpr (fsize == 32) {
                                code.vcvttps2udq(src | k1 | T_z, src);
                            } else {
                                code.vcvttpd2uqq(src | k1 | T_z, src);
                            }
                        } else {
                            // Zero is minimum
                            code.xorps(xmm0, xmm0);
                            FCODE(cmplep)(xmm0, src);
                            FCODE(andp)(src, xmm0);

                            // Will we exceed unsigned range?
                            const Xbyak::Xmm exceed_unsigned = ctx.reg_alloc.ScratchXmm();
                            code.movaps(exceed_unsigned, GetVectorOf<fsize, float_upper_limit_unsigned>(code));
                            FCODE(cmplep)(exceed_unsigned, src);

                            // Will be exceed signed range?
                            const Xbyak::Xmm tmp = ctx.reg_alloc.ScratchXmm();
                            code.movaps(tmp, GetVectorOf<fsize, float_upper_limit_signed>(code));
                            code.movaps(xmm0, tmp);
                            FCODE(cmplep)(xmm0, src);
                            FCODE(andp)(tmp, xmm0);
                            FCODE(subp)(src, tmp);
                            perform_conversion(src);
                            ICODE(psll)(xmm0, static_cast<u8>(fsize - 1));
                            FCODE(orp)(src, xmm0);

                            // Saturate to max
                            FCODE(orp)(src, exceed_unsigned);
                        }
                    } else {
                        using FPT = mcl::unsigned_integer_of_size<fsize>;  // WORKAROUND: For issue 678 on MSVC
                        constexpr u64 integer_max = static_cast<FPT>(std::numeric_limits<std::conditional_t<unsigned_, FPT, std::make_signed_t<FPT>>>::max());

                        code.movaps(xmm0, GetVectorOf<fsize, float_upper_limit_signed>(code));
                        FCODE(cmplep)(xmm0, src);
                        perform_conversion(src);
                        FCODE(blendvp)(src, GetVectorOf<fsize, integer_max>(code));
                    }
                });

                ctx.reg_alloc.DefineValue(inst, src);
                return;
            }
        }

        using fbits_list = mp::lift_sequence<std::make_index_sequence<fsize + 1>>;
        using rounding_list = mp::list<
                mp::lift_value<FP::RoundingMode::ToNearest_TieEven>,
                mp::lift_value<FP::RoundingMode::TowardsPlusInfinity>,
                mp::lift_value<FP::RoundingMode::TowardsMinusInfinity>,
                mp::lift_value<FP::RoundingMode::TowardsZero>,
                mp::lift_value<FP::RoundingMode::ToNearest_TieAwayFromZero>>;

        static const auto lut = Common::GenerateLookupTableFromList(
                []<typename I>(I) {
                    using FPT = mcl::unsigned_integer_of_size<fsize>;  // WORKAROUND: For issue 678 on MSVC
                    return std::pair{
                            mp::lower_to_tuple_v<I>,
                            Common::FptrCast(
                                    [](VectorArray<FPT>& output, const VectorArray<FPT>& input, FP::FPCR fpcr, FP::FPSR& fpsr) {
                                        constexpr size_t fbits = mp::get<0, I>::value;
                                        constexpr FP::RoundingMode rounding_mode = mp::get<1, I>::value;

                                        for (size_t i = 0; i < output.size(); ++i) {
                                            output[i] = static_cast<FPT>(FP::FPToFixed<FPT>(fsize, input[i], fbits, unsigned_, fpcr, rounding_mode, fpsr));
                                        }
                                    })};
                },
                mp::cartesian_product<fbits_list, rounding_list>{});

        EmitTwoOpFallback<3>(code, ctx, inst, lut.at(std::make_tuple(fbits, rounding)));
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorToSignedFixed16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
//        EmitFPVectorToFixed<16, false>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorToSignedFixed32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitFPVectorToFixed<32, false>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorToSignedFixed64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitFPVectorToFixed<64, false>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorToUnsignedFixed16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
//        EmitFPVectorToFixed<16, true>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorToUnsignedFixed32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitFPVectorToFixed<32, true>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorToUnsignedFixed64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
//        EmitToFixed<64, false>(code, ctx, inst);
        EmitFPVectorToFixed<64, true>(code, ctx, inst);

    }

}  // namespace Dynarmic::Backend::LoongArch64
