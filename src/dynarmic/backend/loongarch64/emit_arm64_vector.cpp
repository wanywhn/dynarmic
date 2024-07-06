/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <mcl/mp/metavalue/lift_value.hpp>
#include <bitset>

#include "dynarmic/backend/loongarch64/a32_jitstate.h"
#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/emit_arm64.h"
#include "dynarmic/backend/loongarch64/emit_context.h"
#include "dynarmic/backend/loongarch64/fpsr_manager.h"
#include "dynarmic/backend/loongarch64/reg_alloc.h"
#include "dynarmic/common/always_false.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"
#include "xbyak_loongarch64.h"
#include "mcl/type_traits/function_info.hpp"
#include "dynarmic/common/math_util.h"
#include "mcl/bit_cast.hpp"

namespace Dynarmic::Backend::LoongArch64 {

    template<typename Lambda>
    static void
    EmitTwoArgumentFallbackWithSaturation(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst,
                                          Lambda lambda) {
        const auto fn = static_cast<mcl::equivalent_function_type<Lambda> *>(lambda);
        constexpr u32 stack_size = 3 * 16;
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Xscratch0.getIdx()), stack_size);


        auto arg1 = ctx.reg_alloc.ReadX(args[0]);
        auto arg2 = ctx.reg_alloc.ReadX(args[1]);


        code.add_d(code.a0, code.sp, code.zero);
        code.addi_d(code.a1, code.sp, 1 * 16);
        code.addi_d(code.a2, code.sp, 2 * 16);

        code.st_d(arg1, code.a1, 0);
        code.st_d(arg2, code.a2, 0);

        code.CallLambda(fn);

        code.ld_d(Xscratch0, code.sp, 0 * 16);
        ctx.reg_alloc.DefineAsRegister(inst, Xscratch0);

        code.ld_d(Xscratch1, code.sp, code.GetJitStateInfo().offsetof_fpsr_qc);
        code.or_(Xscratch1, Xscratch1, code.a0);
        code.st_d(Xscratch1, code.sp, code.GetJitStateInfo().offsetof_fpsr_qc);

        ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << Xscratch0.getIdx()), stack_size);
    }


    template<typename Lambda>
    static void
    EmitTwoArgumentFallbackWithSaturationAndImmediate(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst, Lambda lambda) {
        const auto fn = static_cast<mcl::equivalent_function_type<Lambda> *>(lambda);
        constexpr u32 stack_size = 2 * 16;
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        const u8 arg2 = args[1].GetImmediateU8();


        ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Xscratch0.getIdx()), stack_size);


        auto arg1 = ctx.reg_alloc.ReadX(args[0]);


        code.add_d(code.a0, code.sp, code.zero);
        code.addi_d(code.a1, code.sp, 1 * 16);
        code.addi_d(code.a2, code.zero, arg2);
        code.st_d(arg1, code.a1, 0);
        code.CallLambda(fn);

        code.ld_d(Xscratch0, code.sp, 0 * 16);
        ctx.reg_alloc.DefineAsRegister(inst, Xscratch0);

        code.ld_d(Xscratch1, code.sp, code.GetJitStateInfo().offsetof_fpsr_qc);
        code.or_(Xscratch1, Xscratch1, code.a0);
        code.st_d(Xscratch1, code.sp, code.GetJitStateInfo().offsetof_fpsr_qc);

        ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << Xscratch0.getIdx()), stack_size);

    }


    template<typename Lambda>
    static void
    EmitOneArgumentFallback(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst, Lambda lambda) {
        const auto fn = static_cast<mcl::equivalent_function_type<Lambda> *>(lambda);
        constexpr u32 stack_size = 2 * 16;
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto arg1 = ctx.reg_alloc.ReadX(args[0]);

        ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Xscratch0.getIdx()), stack_size);

        code.add_d(code.a0, code.sp, code.zero);
        code.addi_d(code.a1, code.sp, 1 * 16);
        code.st_d(arg1, code.a1, 0);
        code.CallLambda(fn);

        code.ld_d(Xscratch0, code.sp, 0 * 16);
        ctx.reg_alloc.DefineAsRegister(inst, Xscratch0);

        ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << Xscratch0.getIdx()), stack_size);
    }


    template<typename Lambda>
    static void
    EmitOneArgumentFallbackWithSaturation(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst,
                                          Lambda lambda) {
        const auto fn = static_cast<mcl::equivalent_function_type<Lambda> *>(lambda);
        constexpr u32 stack_size = 2 * 16;
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Xscratch0.getIdx()), stack_size);


        auto arg1 = ctx.reg_alloc.ReadX(args[0]);


        code.add_d(code.a0, code.sp, code.zero);
        code.addi_d(code.a1, code.sp, 1 * 16);

        code.st_d(arg1, code.a1, 0);

        code.CallLambda(fn);

        code.ld_d(Xscratch0, code.sp, 0 * 16);
        ctx.reg_alloc.DefineAsRegister(inst, Xscratch0);

        code.ld_d(Xscratch1, code.sp, code.GetJitStateInfo().offsetof_fpsr_qc);
        code.or_(Xscratch1, Xscratch1, code.a0);
        code.st_d(Xscratch1, code.sp, code.GetJitStateInfo().offsetof_fpsr_qc);

        ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << Xscratch0.getIdx()), stack_size);
    }

    template<typename T>
    static constexpr T VShift(T x, T y) {
        const s8 shift_amount = static_cast<s8>(static_cast<u8>(y));
        const s64 bit_size = static_cast<s64>(mcl::bitsizeof<T>);

        if constexpr (std::is_signed_v<T>) {
            if (shift_amount >= bit_size) {
                return 0;
            }

            if (shift_amount <= -bit_size) {
                // Parentheses necessary, as MSVC doesn't appear to consider cast parentheses
                // as a grouping in terms of precedence, causing warning C4554 to fire. See:
                // https://developercommunity.visualstudio.com/content/problem/144783/msvc-2017-does-not-understand-that-static-cast-cou.html
                return x >> (T(bit_size - 1));
            }
        } else if (shift_amount <= -bit_size || shift_amount >= bit_size) {
            return 0;
        }

        if (shift_amount < 0) {
            return x >> T(-shift_amount);
        }

        using unsigned_type = std::make_unsigned_t<T>;
        return static_cast<T>(static_cast<unsigned_type>(x) << static_cast<unsigned_type>(shift_amount));
    }

    template<typename Lambda>
    static void
    EmitTwoArgumentFallback(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst, Lambda lambda) {
        const auto fn = static_cast<mcl::equivalent_function_type<Lambda> *>(lambda);
        constexpr u32 stack_size = 3 * 16;
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);

        ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Xscratch0.getIdx()), stack_size);


        auto arg1 = ctx.reg_alloc.ReadX(args[0]);
        auto arg2 = ctx.reg_alloc.ReadX(args[1]);
        RegAlloc::Realize(arg1, arg2);

        code.add_d(code.a0, code.sp, code.zero);
        code.addi_d(code.a1, code.sp, 1 * 16);
        code.addi_d(code.a2, code.sp, 2 * 16);

        code.st_d(arg1, code.a1, 0);
        code.st_d(arg2, code.a2, 0);

        code.CallLambda(fn);

        code.ld_d(Xscratch0, code.sp, 0 * 16);
        ctx.reg_alloc.DefineAsRegister(inst, Xscratch0);

        ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << Xscratch0.getIdx()), stack_size);
    }

    template<typename EmitFn>
    static void EmitTwoOp(BlockOfCode &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        auto Qoperand = ctx.reg_alloc.ReadQ(args[0]);
        RegAlloc::Realize(Qresult, Qoperand);

        emit(Qresult, Qoperand);
    }

    template<typename EmitFn>
    static void EmitThreeOp(BlockOfCode &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        auto Qa = ctx.reg_alloc.ReadQ(args[0]);
        auto Qb = ctx.reg_alloc.ReadQ(args[1]);
        RegAlloc::Realize(Qresult, Qa, Qb);

        emit(*Qresult, *Qa, *Qb);
    }
    
    template<typename EmitFn>
    static void EmitImmShift(BlockOfCode &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        auto Qoperand = ctx.reg_alloc.ReadQ(args[0]);
        const u8 shift_amount = args[1].GetImmediateU8();
        RegAlloc::Realize(Qresult, Qoperand);


        emit(*Qresult, *Qoperand, shift_amount);

    }
    
    template<size_t size, typename EmitFn>
    static void EmitGetElement(BlockOfCode &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ASSERT(args[1].IsImmediate());
        const u8 index = args[1].GetImmediateU8();

        auto Rresult = ctx.reg_alloc.WriteReg<std::max<size_t>(32, size)>(inst);
        auto Qvalue = ctx.reg_alloc.ReadQ(args[0]);
        RegAlloc::Realize(Rresult, Qvalue);

        // TODO: fpr destination

        emit(Rresult, Qvalue, index);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorGetElement8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitGetElement<8>(code, ctx, inst,
                          [&](auto &Wresult, auto &Qvalue, u8 index) {
                              code.vpickve2gr_bu(Wresult, Qvalue, index);
                          });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorGetElement16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitGetElement<16>(code, ctx, inst,
                           [&](auto &Wresult, auto &Qvalue, u8 index) {
                               code.vpickve2gr_hu(Wresult, Qvalue, index);
                           });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorGetElement32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitGetElement<32>(code, ctx, inst,
                           [&](auto &Wresult, auto &Qvalue, u8 index) {
                               code.vpickve2gr_wu(Wresult, Qvalue, index);
                           });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorGetElement64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitGetElement<64>(code, ctx, inst,
                           [&](auto &Xresult, auto &Qvalue, u8 index) {
                               code.vpickve2gr_du(Xresult, Qvalue, index);
                           });
    }

    template<size_t size, typename EmitFn>
    static void EmitSetElement(BlockOfCode &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        ASSERT(args[1].IsImmediate());
        const u8 index = args[1].GetImmediateU8();

        auto Qvector = ctx.reg_alloc.ReadWriteQ(args[0], inst);
        auto Rvalue = ctx.reg_alloc.ReadReg<std::max<size_t>(32, size)>(args[2]);
        RegAlloc::Realize(Qvector, Rvalue);

        // TODO: fpr source

        emit(Qvector, Rvalue, index);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSetElement8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitSetElement<8>(
                code, ctx, inst,
                [&](auto &Qvector, auto &Wvalue, u8 index) {
                    code.vinsgr2vr_b(Qvector, Wvalue, index);
                });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSetElement16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitSetElement<16>(
                code, ctx, inst,
                [&](auto &Qvector, auto &Wvalue, u8 index) {
                    code.vinsgr2vr_h(Qvector, Wvalue, index);
                });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSetElement32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitSetElement<32>(
                code, ctx, inst,
                [&](auto &Qvector, auto &Wvalue, u8 index) {
                    code.vinsgr2vr_w(Qvector, Wvalue, index);
                });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSetElement64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitSetElement<64>(
                code, ctx, inst,
                [&](auto &Qvector, auto &Wvalue, u8 index) {
                    code.vinsgr2vr_d(Qvector, Wvalue, index);
                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAbs8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) {
            code.vreplgr2vr_b(Vscratch0, code.zero);
            code.vabsd_b(Vresult, Voperand, Vscratch0);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAbs16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) {
            code.vreplgr2vr_b(Vscratch0, code.zero);
            code.vabsd_h(Vresult, Voperand, Vscratch0);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAbs32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) {
            code.vreplgr2vr_b(Vscratch0, code.zero);
            code.vabsd_w(Vresult, Voperand, Vscratch0);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAbs64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) {
            code.vreplgr2vr_b(Vscratch0, code.zero);
            code.vabsd_d(Vresult, Voperand, Vscratch0);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAdd8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vadd_b(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAdd16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vadd_h(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAdd32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vadd_w(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAdd64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vadd_d(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAnd>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vand_v(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAndNot>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vandn_v(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorArithmeticShiftRight8>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitImmShift(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.vsrai_b(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorArithmeticShiftRight16>(BlockOfCode &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitImmShift(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.vsrai_h(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorArithmeticShiftRight32>(BlockOfCode &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitImmShift(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.vsrai_w(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorArithmeticShiftRight64>(BlockOfCode &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitImmShift(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.vsrai_d(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorArithmeticVShift8>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<s8> &result, const VectorArray<s8> &a, const VectorArray<s8> &b) {
                                    std::transform(a.begin(), a.end(), b.begin(), result.begin(), VShift<s8>);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorArithmeticVShift16>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<s16> &result, const VectorArray<s16> &a, const VectorArray<s16> &b) {
                                    std::transform(a.begin(), a.end(), b.begin(), result.begin(), VShift<s16>);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorArithmeticVShift32>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<s32> &result, const VectorArray<s32> &a, const VectorArray<s32> &b) {
                                    std::transform(a.begin(), a.end(), b.begin(), result.begin(), VShift<s32>);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorArithmeticVShift64>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<s64> &result, const VectorArray<s64> &a, const VectorArray<s64> &b) {
                                    std::transform(a.begin(), a.end(), b.begin(), result.begin(), VShift<s64>);
                                });
    }

    template<size_t size, typename EmitFn>
    static void EmitBroadcast(BlockOfCode &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qvector = ctx.reg_alloc.WriteQ(inst);
        auto Rvalue = ctx.reg_alloc.ReadReg<std::max<size_t>(32, size)>(args[0]);
        RegAlloc::Realize(Qvector, Rvalue);

        // TODO: fpr source

        emit(Qvector, Rvalue);
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastLower8>(BlockOfCode &code, EmitContext &ctx,
                                                   IR::Inst *inst) {
        EmitBroadcast<8>(code, ctx, inst, [&](auto &Qvector, auto &Wvalue) {
            code.vreplgr2vr_b(Qvector, Wvalue);
            code.vinsgr2vr_d(Qvector, code.zero, 1);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastLower16>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitBroadcast<16>(code, ctx, inst, [&](auto &Qvector, auto &Wvalue) {
            code.vreplgr2vr_h(Qvector, Wvalue);
            code.vinsgr2vr_d(Qvector, code.zero, 1);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastLower32>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitBroadcast<32>(code, ctx, inst, [&](auto &Qvector, auto &Wvalue) {
//            code.vxor_v(Vscratch0, Vscratch0, Vscratch0);
//            code.vreplgr2vr_w(Qvector, Wvalue);
//            code.vpickev_b(Qvector, Vscratch0, Qvector);
//            code.vpickod_b(Qvector, Vscratch0, Qvector);
            code.vreplgr2vr_w(Qvector, Wvalue);
            code.vinsgr2vr_d(Qvector, code.zero, 1);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorBroadcast8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitBroadcast<8>(code, ctx, inst, [&](auto &Qvector, auto &Wvalue) { code.vreplgr2vr_b(Qvector, Wvalue); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorBroadcast16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitBroadcast<16>(code, ctx, inst, [&](auto &Qvector, auto &Wvalue) { code.vreplgr2vr_h(Qvector, Wvalue); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorBroadcast32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitBroadcast<32>(code, ctx, inst, [&](auto &Qvector, auto &Wvalue) { code.vreplgr2vr_w(Qvector, Wvalue); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorBroadcast64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitBroadcast<64>(code, ctx, inst, [&](auto &Qvector, auto &Xvalue) { code.vreplgr2vr_d(Qvector, Xvalue); });
    }

    template<size_t size, typename EmitFn>
    static void
    EmitBroadcastElement(BlockOfCode &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qvector = ctx.reg_alloc.WriteQ(inst);
        auto Qvalue = ctx.reg_alloc.ReadQ(args[0]);
        const u8 index = args[1].GetImmediateU8();
        RegAlloc::Realize(Qvector, Qvalue);

        emit(Qvector, Qvalue, index);
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastElementLower8>(BlockOfCode &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitBroadcastElement<8>(code, ctx, inst, [&](auto &Qvector, auto &Qvalue, u8 index) {
            code.vreplvei_b(Qvector, Qvalue, index);
            code.vinsgr2vr_d(Qvector, code.zero, 1);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastElementLower16>(BlockOfCode &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitBroadcastElement<16>(code, ctx, inst, [&](auto &Qvector, auto &Qvalue, u8 index) {
            code.vreplvei_h(Qvector, Qvalue, index);
            code.vinsgr2vr_d(Qvector, code.zero, 1);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastElementLower32>(BlockOfCode &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitBroadcastElement<32>(code, ctx, inst, [&](auto &Qvector, auto &Qvalue, u8 index) {
//            code.vreplvei_w(Qvector, Qvector, index);
            code.vreplvei_w(Qvector, Qvalue, index);
            code.vinsgr2vr_d(Qvector, code.zero, 1);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastElement8>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitBroadcastElement<8>(code, ctx, inst, [&](auto &Qvector, auto &Qvalue, u8 index) {
            code.vreplvei_b(Qvector, Qvalue, index);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastElement16>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitBroadcastElement<16>(code, ctx, inst, [&](auto &Qvector, auto &Qvalue, u8 index) {
            code.vreplvei_h(Qvector, Qvalue, index);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastElement32>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitBroadcastElement<32>(code, ctx, inst, [&](auto &Qvector, auto &Qvalue, u8 index) {
            code.vreplvei_w(Qvector, Qvalue, index);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastElement64>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitBroadcastElement<64>(code, ctx, inst, [&](auto &Qvector, auto &Qvalue, u8 index) {
            code.vreplvei_d(Qvector, Qvalue, index);
//            code.DUP(Qvector->D2(), Qvalue->Delem()[index]);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorCountLeadingZeros8>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) { code.vclz_b(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorCountLeadingZeros16>(BlockOfCode &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) { code.vclz_h(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorCountLeadingZeros32>(BlockOfCode &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) { code.vclz_w(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveEven8>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                               [&](auto Vresult, auto Va, auto Vb) { code.vpickev_b(Vresult, Vb, Va); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveEven16>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vpickev_h(Vresult, Vb, Va); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveEven32>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vpickev_w(Vresult, Vb, Va); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveEven64>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vpickev_d(Vresult, Vb, Va); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveEvenLower8>(BlockOfCode &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                    [&](auto Vresult, auto Va, auto Vb) {
                                        code.vpickev_b(Vresult, Vb, Va);
                                        code.vinsgr2vr_d(Vresult, code.zero, 1);

                                    });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveEvenLower16>(BlockOfCode &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) {
                                         code.vpickev_h(Vresult, Vb, Va);
                                         code.vinsgr2vr_d(Vresult, code.zero, 1);

                                     });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveEvenLower32>(BlockOfCode &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) {
                                         code.vpickev_w(Vresult, Vb, Va);
                                         code.vinsgr2vr_d(Vresult, code.zero, 1);

                                     });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveOdd8>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                               [&](auto Vresult, auto Va, auto Vb) { code.vpickod_b(Vresult, Vb, Va); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveOdd16>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vpickod_h(Vresult, Vb, Va); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveOdd32>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vpickod_w(Vresult, Vb, Va); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveOdd64>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vpickod_d(Vresult, Vb, Va); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveOddLower8>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                    [&](auto Vresult, auto Va, auto Vb) {
                                        code.vpickod_b(Vresult, Vb, Va);
                                        code.vinsgr2vr_d(Vresult, code.zero, 1);

                                    });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveOddLower16>(BlockOfCode &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) {
                                         code.vpickod_h(Vresult, Vb, Va);
                                         code.vinsgr2vr_d(Vresult, code.zero, 1);

                                     });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveOddLower32>(BlockOfCode &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) {
                                         code.vpickod_w(Vresult, Vb, Va);
                                         code.vinsgr2vr_d(Vresult, code.zero, 1);

                                     });
    }

    template<>
    void EmitIR<IR::Opcode::VectorEor>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vxor_v(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorEqual8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vseq_b(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorEqual16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vseq_h(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorEqual32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vseq_w(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorEqual64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vseq_d(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorEqual128>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::VectorExtract>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        auto Qa = ctx.reg_alloc.ReadQ(args[0]);
        auto Qb = ctx.reg_alloc.ReadQ(args[1]);
        const u8 position = args[2].GetImmediateU8();
        ASSERT(position % 8 == 0);
        RegAlloc::Realize(Qresult, Qa, Qb);

        if (position != 0) {
            code.vilvh_b(Qresult, Qb, Qa);
            code.vsrli_b(Qa, Qresult, position / 8);
        }
        code.vxor_v(Qresult, Qresult, Qresult);
        code.vor_v(Qresult, Qresult, Qa);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorExtractLower>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Dresult = ctx.reg_alloc.WriteD(inst);
        auto Da = ctx.reg_alloc.ReadD(args[0]);
        auto Db = ctx.reg_alloc.ReadD(args[1]);
        const u8 position = args[2].GetImmediateU8();
        ASSERT(position % 8 == 0);
        RegAlloc::Realize(Dresult, Da, Db);
        if (position != 0) {
            code.vilvl_b(Dresult, Db, Da);
            code.vsrli_b(Da, Dresult, position / 8);
        }
        code.vxor_v(Dresult, Dresult, Dresult);
        code.vor_v(Dresult, Dresult, Da);
    }

    template<>
    void EmitIR<IR::Opcode::VectorGreaterS8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vsle_b(Vresult, Vb, Va);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorGreaterS16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vsle_h(Vresult, Vb, Va);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorGreaterS32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vsle_w(Vresult, Vb, Va);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorGreaterS64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vsle_d(Vresult, Vb, Va);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingAddS8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vavg_b(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingAddS16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vavg_h(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingAddS32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vavg_w(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingAddU8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vavg_bu(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingAddU16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vavg_hu(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingAddU32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vavg_wu(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingSubS8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vsub_b(Vresult, Vresult, Vb);
            code.vavg_b(Vresult, Va, Vresult);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingSubS16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vsub_h(Vresult, Vresult, Vb);
            code.vavg_h(Vresult, Va, Vresult);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingSubS32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vsub_w(Vresult, Vresult, Vb);
            code.vavg_w(Vresult, Va, Vresult);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingSubU8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vsub_b(Vresult, Vresult, Vb);
            code.vavg_bu(Vresult, Va, Vresult);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingSubU16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vsub_h(Vresult, Vresult, Vb);
            code.vavg_hu(Vresult, Va, Vresult);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingSubU32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vsub_w(Vresult, Vresult, Vb);
            code.vavg_wu(Vresult, Va, Vresult);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorInterleaveLower8>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vilvl_b(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorInterleaveLower16>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vilvl_h(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorInterleaveLower32>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vilvl_w(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorInterleaveLower64>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vilvl_d(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorInterleaveUpper8>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vilvh_b(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorInterleaveUpper16>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vilvh_h(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorInterleaveUpper32>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vilvh_w(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorInterleaveUpper64>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vilvh_d(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalShiftLeft8>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitImmShift(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.vslli_b(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalShiftLeft16>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitImmShift(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.vslli_h(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalShiftLeft32>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitImmShift(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.vslli_w(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalShiftLeft64>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitImmShift(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.vslli_d(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalShiftRight8>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitImmShift(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.vsrli_b(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalShiftRight16>(BlockOfCode &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitImmShift(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.vsrli_h(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalShiftRight32>(BlockOfCode &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitImmShift(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.vsrli_w(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalShiftRight64>(BlockOfCode &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitImmShift(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.vsrli_d(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorLogicalVShift8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsll_b(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalVShift16>(BlockOfCode &code, EmitContext &ctx,
                                                   IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsll_h(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalVShift32>(BlockOfCode &code, EmitContext &ctx,
                                                   IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsll_w(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalVShift64>(BlockOfCode &code, EmitContext &ctx,
                                                   IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsll_d(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMaxS8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vmax_b(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMaxS16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vmax_h(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMaxS32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vmax_w(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMaxS64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vmax_d(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMaxU8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vmax_bu(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMaxU16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vmax_hu(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMaxU32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vmax_wu(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMaxU64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vmax_du(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMinS8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vmin_b(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMinS16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vmin_h(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMinS32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vmin_w(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMinS64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vmin_d(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMinU8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vmin_bu(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMinU16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vmin_hu(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMinU32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vmin_wu(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMinU64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vmin_du(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMultiply8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vmul_b(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorMultiply16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vmul_h(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorMultiply32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vmul_w(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorMultiply64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vmul_d(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMultiplySignedWiden8>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vmulwod_h_b(Vresult, Va, Vb);
            code.vmulwev_h_b(Vscratch0, Va, Vb);
            code.vilvl_h(Vresult, Vresult, Vscratch0);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMultiplySignedWiden16>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vmulwod_w_h(Vresult, Va, Vb);
            code.vmulwev_w_h(Vscratch0, Va, Vb);
            code.vilvl_w(Vresult, Vresult, Vscratch0);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMultiplySignedWiden32>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vmulwod_d_w(Vresult, Va, Vb);
            code.vmulwev_d_w(Vscratch0, Va, Vb);
            code.vilvl_d(Vresult, Vresult, Vscratch0);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMultiplyUnsignedWiden8>(BlockOfCode &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vmulwod_h_bu(Vresult, Va, Vb);
            code.vmulwev_h_bu(Vscratch0, Va, Vb);
            code.vilvl_h(Vresult, Vresult, Vscratch0);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMultiplyUnsignedWiden16>(BlockOfCode &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vmulwod_w_hu(Vresult, Va, Vb);
            code.vmulwev_w_hu(Vscratch0, Va, Vb);
            code.vilvl_w(Vresult, Vresult, Vscratch0);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMultiplyUnsignedWiden32>(BlockOfCode &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vmulwod_d_wu(Vresult, Va, Vb);
            code.vmulwev_d_wu(Vscratch0, Va, Vb);
            code.vilvl_d(Vresult, Vresult, Vscratch0);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorNarrow16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto Va = ctx.reg_alloc.ReadQ(ctx.reg_alloc.GetArgumentInfo(inst)[0]);
        auto Vresult = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(Va, Vresult);

        code.vxor_v(Vresult, Vresult, Vresult);
        code.vsrlni_b_h(Vresult, Va, 0);
    }

    template<>
    void EmitIR<IR::Opcode::VectorNarrow32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto Va = ctx.reg_alloc.ReadQ(ctx.reg_alloc.GetArgumentInfo(inst)[0]);
        auto Vresult = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(Va, Vresult);

        code.vxor_v(Vresult, Vresult, Vresult);
        code.vsrlni_h_w(Vresult, Va, 0);
    }

    template<>
    void EmitIR<IR::Opcode::VectorNarrow64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto Va = ctx.reg_alloc.ReadQ(ctx.reg_alloc.GetArgumentInfo(inst)[0]);
        auto Vresult = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(Va, Vresult);

        code.vxor_v(Vresult, Vresult, Vresult);
        code.vsrlni_w_d(Vresult, Va, 0);
    }

    template<>
    void EmitIR<IR::Opcode::VectorNot>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) {
            code.vorn_v(Vscratch0, Vscratch0, Vscratch0);
            code.vxor_v(Vresult, Voperand, Vscratch0);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorOr>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vor_v(Vresult, Va, Vb); });
    }


    template<typename T, typename Function>
    static void PairedOperation(VectorArray<T> &result, const VectorArray<T> &x, const VectorArray<T> &y, Function fn) {
        const size_t range = x.size() / 2;

        for (size_t i = 0; i < range; i++) {
            result[i] = fn(x[2 * i], x[2 * i + 1]);
        }

        for (size_t i = 0; i < range; i++) {
            result[range + i] = fn(y[2 * i], y[2 * i + 1]);
        }
    }

    template<typename T, typename Function>
    static void
    LowerPairedOperation(VectorArray<T> &result, const VectorArray<T> &x, const VectorArray<T> &y, Function fn) {
        const size_t range = x.size() / 4;

        for (size_t i = 0; i < range; i++) {
            result[i] = fn(x[2 * i], x[2 * i + 1]);
        }

        for (size_t i = 0; i < range; i++) {
            result[range + i] = fn(y[2 * i], y[2 * i + 1]);
        }
    }

    template<typename T>
    static void PairedMax(VectorArray<T> &result, const VectorArray<T> &x, const VectorArray<T> &y) {
        PairedOperation(result, x, y, [](auto a, auto b) { return std::max(a, b); });
    }

    template<typename T>
    static void PairedMin(VectorArray<T> &result, const VectorArray<T> &x, const VectorArray<T> &y) {
        PairedOperation(result, x, y, [](auto a, auto b) { return std::min(a, b); });
    }

    template<typename T>
    static void LowerPairedMax(VectorArray<T> &result, const VectorArray<T> &x, const VectorArray<T> &y) {
        LowerPairedOperation(result, x, y, [](auto a, auto b) { return std::max(a, b); });
    }

    template<typename T>
    static void LowerPairedMin(VectorArray<T> &result, const VectorArray<T> &x, const VectorArray<T> &y) {
        LowerPairedOperation(result, x, y, [](auto a, auto b) { return std::min(a, b); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddLower8>(BlockOfCode &code, EmitContext &ctx,
                                                   IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                    [&](auto Vresult, auto Va, auto Vb) {
                        code.vpickev_b(Vresult, Va, Va);
                        code.vpickod_b(Vscratch0, Va, Va);
                        code.vadd_b(Vresult, Vresult, Vscratch0);
                        code.vpickev_b(Vscratch1, Vb, Vb);
                        code.vpickod_b(Vscratch0, Vb, Vb);
                        code.vadd_b(Vscratch1, Vscratch1, Vscratch0);
                        code.vextrins_w(Vresult, Vscratch1, 0x12);
                    });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddLower16>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                    [&](auto Vresult, auto Va, auto Vb) {
                        code.vpickev_h(Vresult, Va, Va);
                        code.vpickod_h(Vscratch0, Va, Va);
                        code.vadd_h(Vresult, Vresult, Vscratch0);
                        code.vpickev_h(Vscratch1, Vb, Vb);
                        code.vpickod_h(Vscratch0, Vb, Vb);
                        code.vadd_h(Vscratch1, Vscratch1, Vscratch0);
                        code.vextrins_w(Vresult, Vscratch1, 0x12);
                    });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddLower32>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_w(Vresult, Va, Va);
            code.vpickod_w(Vscratch0, Va, Va);
            code.vadd_w(Vresult, Vresult, Vscratch0);
            code.vpickev_w(Vscratch1, Vb, Vb);
            code.vpickod_w(Vscratch0, Vb, Vb);
            code.vadd_w(Vscratch1, Vscratch1, Vscratch0);
            code.vextrins_w(Vresult, Vscratch1, 0x12);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddSignedWiden8>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                                      [&](auto &Vresult, auto &Voperand) {
                                          code.vbsrl_v(Vresult, Voperand, 1);
                                          code.vaddwev_h_b(Vresult, Vresult, Voperand);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddSignedWiden16>(BlockOfCode &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                                       [&](auto &Vresult, auto &Voperand) {
                                           code.vbsrl_v(Vresult, Voperand, 2);
                                           code.vaddwev_w_h(Vresult, Vresult, Voperand);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddSignedWiden32>(BlockOfCode &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                                       [&](auto &Vresult, auto &Voperand) {
                                           code.vbsrl_v(Vresult, Voperand, 4);
                                           code.vaddwev_d_w(Vresult, Vresult, Voperand);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddUnsignedWiden8>(BlockOfCode &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                                      [&](auto &Vresult, auto &Voperand) {
                                          code.vbsrl_v(Vresult, Voperand, 1);
                                          code.vaddwev_h_bu(Vresult, Vresult, Voperand);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddUnsignedWiden16>(BlockOfCode &code, EmitContext &ctx,
                                                            IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                                       [&](auto &Vresult, auto &Voperand) {
                                           code.vbsrl_v(Vresult, Voperand, 2);
                                           code.vaddwev_w_hu(Vresult, Vresult, Voperand);        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddUnsignedWiden32>(BlockOfCode &code, EmitContext &ctx,
                                                            IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                                       [&](auto &Vresult, auto &Voperand) {
            code.vbsrl_v(Vresult, Voperand, 4);
            code.vaddwev_d_wu(Vresult, Vresult, Voperand);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedAdd8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_b(Vresult, Va, Vb);
            code.vpickod_b(Vscratch2,Va, Vb);
            code.vadd_b(Vresult, Vresult, Vscratch2);        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedAdd16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_h(Vresult, Va, Vb);
            code.vpickod_h(Vscratch2,Va, Vb);
            code.vadd_h(Vresult, Vresult, Vscratch2);        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedAdd32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_w(Vresult, Va, Vb);
            code.vpickod_w(Vscratch2,Va, Vb);
            code.vadd_w(Vresult, Vresult, Vscratch2);        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedAdd64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_d(Vresult, Va, Vb);
            code.vpickod_d(Vscratch2,Va, Vb);
            code.vadd_d(Vresult, Vresult, Vscratch2);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMaxS8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_d(Vresult, Va, Vb);
            code.vpickod_d(Vscratch2,Va, Vb);
            code.vmax_b(Vresult, Vresult, Vscratch2);
        });
//        EmitTwoArgumentFallback(code, ctx, inst,
//                                [](VectorArray<s8> &result, const VectorArray<s8> &a, const VectorArray<s8> &b) {
//                                    PairedMax(result, a, b);
//                                });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMaxS16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_h(Vresult, Va, Vb);
            code.vpickod_h(Vscratch2,Va, Vb);
            code.vmax_h(Vresult, Vresult, Vscratch2);
        });
//        EmitTwoArgumentFallback(code, ctx, inst,
//                                [](VectorArray<s16> &result, const VectorArray<s16> &a, const VectorArray<s16> &b) {
//                                    PairedMax(result, a, b);
//                                });
    }

// TODO is this right
    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMaxS32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_w(Vresult, Va, Vb);
            code.vpickod_w(Vscratch2,Va, Vb);
            code.vmax_w(Vresult, Vresult, Vscratch2);
        });
//        EmitTwoArgumentFallback(code, ctx, inst,
//                                [](VectorArray<s32> &result, const VectorArray<s32> &a, const VectorArray<s32> &b) {
//                                    PairedMax(result, a, b);
//                                });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMaxU8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_b(Vresult, Va, Vb);
            code.vpickod_b(Vscratch2,Va, Vb);
            code.vmax_bu(Vresult, Vresult, Vscratch2);
        });
//        EmitTwoArgumentFallback(code, ctx, inst,
//                                [](VectorArray<u8> &result, const VectorArray<u8> &a, const VectorArray<u8> &b) {
//                                    PairedMax(result, a, b);
//                                });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMaxU16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_h(Vresult, Va, Vb);
            code.vpickod_h(Vscratch2,Va, Vb);
            code.vmax_hu(Vresult, Vresult, Vscratch2);
        });
//        EmitTwoArgumentFallback(code, ctx, inst,
//                                [](VectorArray<u16> &result, const VectorArray<u16> &a, const VectorArray<u16> &b) {
//                                    PairedMax(result, a, b);
//                                });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMaxU32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_w(Vresult, Va, Vb);
            code.vpickod_w(Vscratch2,Va, Vb);
            code.vmax_wu(Vresult, Vresult, Vscratch2);
        });
//        EmitTwoArgumentFallback(code, ctx, inst,
//                                [](VectorArray<u32> &result, const VectorArray<u32> &a, const VectorArray<u32> &b) {
//                                    PairedMax(result, a, b);
//                                });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMinS8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_b(Vresult, Va, Vb);
            code.vpickod_b(Vscratch2,Va, Vb);
            code.vmin_b(Vresult, Vresult, Vscratch2);
        });
//        EmitTwoArgumentFallback(code, ctx, inst,
//                                [](VectorArray<s8> &result, const VectorArray<s8> &a, const VectorArray<s8> &b) {
//                                    PairedMin(result, a, b);
//                                });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMinS16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_h(Vresult, Va, Vb);
            code.vpickod_h(Vscratch2,Va, Vb);
            code.vmin_h(Vresult, Vresult, Vscratch2);
        });
//        EmitTwoArgumentFallback(code, ctx, inst,
//                                [](VectorArray<s16> &result, const VectorArray<s16> &a, const VectorArray<s16> &b) {
//                                    PairedMin(result, a, b);
//                                });
    }

// TODO is this right
    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMinS32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_w(Vresult, Va, Vb);
            code.vpickod_w(Vscratch2,Va, Vb);
            code.vmin_w(Vresult, Vresult, Vscratch2);
        });
//        EmitTwoArgumentFallback(code, ctx, inst,
//                                [](VectorArray<u32> &result, const VectorArray<u32> &a, const VectorArray<u32> &b) {
//                                    PairedMin(result, a, b);
//                                });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMinU8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_b(Vresult, Va, Vb);
            code.vpickod_b(Vscratch2,Va, Vb);
            code.vmin_bu(Vresult, Vresult, Vscratch2);
        });
//        EmitTwoArgumentFallback(code, ctx, inst,
//                                [](VectorArray<u8> &result, const VectorArray<u8> &a, const VectorArray<u8> &b) {
//                                    PairedMin(result, a, b);
//                                });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMinU16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_h(Vresult, Va, Vb);
            code.vpickod_h(Vscratch2,Va, Vb);
            code.vmin_hu(Vresult, Vresult, Vscratch2);
        });
//        EmitTwoArgumentFallback(code, ctx, inst,
//                                [](VectorArray<u16> &result, const VectorArray<u16> &a, const VectorArray<u16> &b) {
//                                    PairedMin(result, a, b);
//                                });
    }

// TODO is this right
    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMinU32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vpickev_w(Vresult, Va, Vb);
            code.vpickod_w(Vscratch2,Va, Vb);
            code.vmin_wu(Vresult, Vresult, Vscratch2);
        });
//        EmitTwoArgumentFallback(code, ctx, inst,
//                                [](VectorArray<u32> &result, const VectorArray<u32> &a, const VectorArray<u32> &b) {
//                                    PairedMin(result, a, b);
//                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMaxLowerS8>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<s8> &result, const VectorArray<s8> &a, const VectorArray<s8> &b) {
                                    LowerPairedMax(result, a, b);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMaxLowerS16>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<s16> &result, const VectorArray<s16> &a, const VectorArray<s16> &b) {
                                    LowerPairedMax(result, a, b);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMaxLowerS32>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<s32> &result, const VectorArray<s32> &a, const VectorArray<s32> &b) {
                                    LowerPairedMax(result, a, b);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMaxLowerU8>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<u8> &result, const VectorArray<u8> &a, const VectorArray<u8> &b) {
                                    LowerPairedMax(result, a, b);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMaxLowerU16>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<u16> &result, const VectorArray<u16> &a, const VectorArray<u16> &b) {
                                    LowerPairedMax(result, a, b);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMaxLowerU32>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<u32> &result, const VectorArray<u32> &a, const VectorArray<u32> &b) {
                                    LowerPairedMax(result, a, b);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMinLowerS8>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<s8> &result, const VectorArray<s8> &a, const VectorArray<s8> &b) {
                                    LowerPairedMin(result, a, b);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMinLowerS16>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<s16> &result, const VectorArray<s16> &a, const VectorArray<s16> &b) {
                                    LowerPairedMin(result, a, b);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMinLowerS32>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<s32> &result, const VectorArray<s32> &a, const VectorArray<s32> &b) {
                                    LowerPairedMin(result, a, b);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMinLowerU8>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<u8> &result, const VectorArray<u8> &a, const VectorArray<u8> &b) {
                                    LowerPairedMin(result, a, b);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMinLowerU16>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<u16> &result, const VectorArray<u16> &a, const VectorArray<u16> &b) {
                                    LowerPairedMin(result, a, b);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMinLowerU32>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<u32> &result, const VectorArray<u32> &a, const VectorArray<u32> &b) {
                                    LowerPairedMin(result, a, b);
                                });
    }
    template<typename D, typename T>
    static D PolynomialMultiply(T lhs, T rhs) {
        constexpr size_t bit_size = mcl::bitsizeof<T>;
        const std::bitset<bit_size> operand(lhs);

        D res = 0;
        for (size_t i = 0; i < bit_size; i++) {
            if (operand[i]) {
                res ^= rhs << i;
            }
        }

        return res;
    }
    template<>
    void EmitIR<IR::Opcode::VectorPolynomialMultiply8>(BlockOfCode &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst, [](VectorArray<u8>& result, const VectorArray<u8>& a, const VectorArray<u8>& b) {
            std::transform(a.begin(), a.end(), b.begin(), result.begin(), PolynomialMultiply<u8, u8>);
        });    }

    template<>
    void EmitIR<IR::Opcode::VectorPolynomialMultiplyLong8>(BlockOfCode &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst, [](VectorArray<u16>& result, const VectorArray<u8>& a, const VectorArray<u8>& b) {
            for (size_t i = 0; i < result.size(); i++) {
                result[i] = PolynomialMultiply<u16, u8>(a[i], b[i]);
            }
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPolynomialMultiplyLong64>(BlockOfCode &code, EmitContext &ctx,
                                                            IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst, [](VectorArray<u64>& result, const VectorArray<u64>& a, const VectorArray<u64>& b) {
            const auto handle_high_bits = [](u64 lhs, u64 rhs) {
                constexpr size_t bit_size = mcl::bitsizeof<u64>;
                u64 result = 0;

                for (size_t i = 1; i < bit_size; i++) {
                    if (mcl::bit::get_bit(i, lhs)) {
                        result ^= rhs >> (bit_size - i);
                    }
                }

                return result;
            };

            result[0] = PolynomialMultiply<u64, u64>(a[0], b[0]);
            result[1] = handle_high_bits(a[0], b[0]);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPopulationCount>(BlockOfCode &code, EmitContext &ctx,
                                                   IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) { code.vpcnt_b(Vresult, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReverseBits>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) {
            code.vpickve2gr_du(Xscratch0, Voperand, 0);
            code.vpickve2gr_du(Xscratch1, Voperand, 1);
            code.bitrev_d(Xscratch0, Xscratch0);
            code.bitrev_d(Xscratch1, Xscratch1);
            code.vinsgr2vr_d(Vresult, Xscratch0, 1);
            code.vinsgr2vr_d(Vresult, Xscratch1, 0);

        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReverseElementsInHalfGroups8>(BlockOfCode &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) {
            code.vshuf4i_h(Vresult, Voperand, 0b00011011);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReverseElementsInWordGroups8>(BlockOfCode &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) {
            code.vshuf4i_w(Vresult, Voperand, 0b00011011);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReverseElementsInWordGroups16>(BlockOfCode &code, EmitContext &ctx,
                                                            IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) {
            code.vshuf4i_w(Vresult, Voperand, 0b00011011);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReverseElementsInLongGroups8>(BlockOfCode &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) {
            code.vshuf4i_d(Vresult, Voperand, 0b00011011);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReverseElementsInLongGroups16>(BlockOfCode &code, EmitContext &ctx,
                                                            IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) {
            code.vshuf4i_d(Vresult, Voperand, 0b00011011);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReverseElementsInLongGroups32>(BlockOfCode &code, EmitContext &ctx,
                                                            IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) {
            code.vshuf4i_d(Vresult, Voperand, 0b01001110);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReduceAdd8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                      [&](auto &Bresult, auto &Voperand) {
            code.vhaddw_h_b(Bresult, Voperand, Voperand);
            code.vhaddw_w_h(Bresult, Bresult, Bresult);
            code.vhaddw_d_w(Bresult, Bresult, Bresult);
            code.vhaddw_q_d(Bresult, Bresult, Bresult);

        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReduceAdd16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                       [&](auto &Hresult, auto &Voperand) {
                           code.vhaddw_w_h(Hresult, Voperand, Voperand);
                           code.vhaddw_d_w(Hresult, Hresult, Hresult);
                           code.vhaddw_q_d(Hresult, Hresult, Hresult);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReduceAdd32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                       [&](auto &Sresult, auto &Voperand) {
                           code.vhaddw_d_w(Sresult, Voperand, Voperand);
                           code.vhaddw_q_d(Sresult, Sresult, Sresult);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReduceAdd64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                       [&](auto &Dresult, auto &Voperand) {
            code.vhaddw_q_d(Dresult, Voperand, Voperand);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRotateWholeVectorRight>(BlockOfCode &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitImmShift(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            ASSERT(shift_amount % 8 == 0);
            const u8 ext_imm = (shift_amount % 128) / 8;
            code.vbsrl_v(Vresult, Voperand, ext_imm);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingHalvingAddS8>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vavgr_b(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingHalvingAddS16>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vavgr_h(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingHalvingAddS32>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vavgr_w(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingHalvingAddU8>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                               [&](auto Vresult, auto Va, auto Vb) { code.vavgr_bu(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingHalvingAddU16>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vavgr_hu(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingHalvingAddU32>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vavgr_wu(Vresult, Va, Vb); });
    }


    template<typename T, typename U>
    static void RoundingShiftLeft(VectorArray<T> &out, const VectorArray<T> &lhs, const VectorArray<U> &rhs) {
        using signed_type = std::make_signed_t<T>;
        using unsigned_type = std::make_unsigned_t<T>;

        constexpr auto bit_size = static_cast<s64>(mcl::bitsizeof<T>);

        for (size_t i = 0; i < out.size(); i++) {
            const s64 extended_shift = static_cast<s64>(mcl::bit::sign_extend<8, u64>(rhs[i] & 0xFF));

            if (extended_shift >= 0) {
                if (extended_shift >= bit_size) {
                    out[i] = 0;
                } else {
                    out[i] = static_cast<T>(static_cast<unsigned_type>(lhs[i]) << extended_shift);
                }
            } else {
                if ((std::is_unsigned_v<T> && extended_shift < -bit_size) ||
                    (std::is_signed_v<T> && extended_shift <= -bit_size)) {
                    out[i] = 0;
                } else {
                    const s64 shift_value = -extended_shift - 1;
                    const T shifted = (lhs[i] & (static_cast<signed_type>(1) << shift_value)) >> shift_value;

                    if (extended_shift == -bit_size) {
                        out[i] = shifted;
                    } else {
                        out[i] = (lhs[i] >> -extended_shift) + shifted;
                    }
                }
            }
        }
    }


    template<>
    void EmitIR<IR::Opcode::VectorRoundingShiftLeftS8>(BlockOfCode &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<s8> &result, const VectorArray<s8> &lhs, const VectorArray<s8> &rhs) {
                                    RoundingShiftLeft(result, lhs, rhs);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingShiftLeftS16>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<s16> &result, const VectorArray<s16> &lhs, const VectorArray<s16> &rhs) {
                                    RoundingShiftLeft(result, lhs, rhs);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingShiftLeftS32>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<s32> &result, const VectorArray<s32> &lhs, const VectorArray<s32> &rhs) {
                                    RoundingShiftLeft(result, lhs, rhs);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingShiftLeftS64>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<s64> &result, const VectorArray<s64> &lhs, const VectorArray<s64> &rhs) {
                                    RoundingShiftLeft(result, lhs, rhs);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingShiftLeftU8>(BlockOfCode &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<u8> &result, const VectorArray<u8> &lhs, const VectorArray<s8> &rhs) {
                                    RoundingShiftLeft(result, lhs, rhs);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingShiftLeftU16>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst,
                                [](VectorArray<u16> &result, const VectorArray<u16> &lhs, const VectorArray<s16> &rhs) {
                                    RoundingShiftLeft(result, lhs, rhs);
                                });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingShiftLeftU32>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst, [](VectorArray<u32> &result, const VectorArray<u32> &lhs,
                                                    const VectorArray<s32> &rhs) {
            RoundingShiftLeft(result, lhs, rhs);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingShiftLeftU64>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitTwoArgumentFallback(code, ctx, inst, [](VectorArray<u64> &result, const VectorArray<u64> &lhs,
                                                    const VectorArray<s64> &rhs) {
            RoundingShiftLeft(result, lhs, rhs);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignExtend8>(BlockOfCode &code, EmitContext &ctx,
                                          IR::Inst *inst) {
        ASSERT_FALSE("Unimplemented");
        (void)code;
        (void)ctx;
        (void)inst;
//        EmitTwoOp(code, ctx, inst,
//                                  [&](auto &Vresult, auto &Voperand) { code.SXTL(Vresult, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignExtend16>(BlockOfCode &code, EmitContext &ctx,
                                           IR::Inst *inst) {
        ASSERT_FALSE("Unimplemented");
        (void)code;
        (void)ctx;
        (void)inst;
//        EmitTwoOp(code, ctx, inst,
//                                   [&](auto &Vresult, auto &Voperand) { code.SXTL(Vresult, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignExtend32>(BlockOfCode &code, EmitContext &ctx,
                                           IR::Inst *inst) {
        ASSERT_FALSE("Unimplemented");
        (void)code;
        (void)ctx;
        (void)inst;
//        EmitTwoOp(code, ctx, inst,
//                                   [&](auto &Vresult, auto &Voperand) { code.SXTL(Vresult, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignExtend64>(BlockOfCode &code, EmitContext &ctx,
                                           IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignedAbsoluteDifference8>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                               [&](auto Vresult, auto Va, auto Vb) { code.vabsd_b(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignedAbsoluteDifference16>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vabsd_h(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignedAbsoluteDifference32>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vabsd_w(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedMultiply16>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedMultiply32>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAbs8>(BlockOfCode &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitOneArgumentFallbackWithSaturation(code, ctx, inst, [](VectorArray<s8>& result, const VectorArray<s8>& data) {
            bool qc_flag = false;

            for (size_t i = 0; i < result.size(); i++) {
                if (static_cast<u16>(data[i]) == 0x80) {
                    result[i] = 0x7F;
                    qc_flag = true;
                } else {
                    result[i] = std::abs(data[i]);
                }
            }

            return qc_flag;
        });
//        EmitTwoOp(code, ctx, inst,
//                                      [&](auto Vresult, auto Voperand) { code.SQABS(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAbs16>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitOneArgumentFallbackWithSaturation(code, ctx, inst, [](VectorArray<s16>& result, const VectorArray<s16>& data) {
            bool qc_flag = false;

            for (size_t i = 0; i < result.size(); i++) {
                if (static_cast<u16>(data[i]) == 0x8000) {
                    result[i] = 0x7FFF;
                    qc_flag = true;
                } else {
                    result[i] = std::abs(data[i]);
                }
            }

            return qc_flag;
        });
//        EmitTwoOp(code, ctx, inst,
//                                       [&](auto Vresult, auto Voperand) { code.SQABS(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAbs32>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitOneArgumentFallbackWithSaturation(code, ctx, inst, [](VectorArray<s32>& result, const VectorArray<s32>& data) {
            bool qc_flag = false;

            for (size_t i = 0; i < result.size(); i++) {
                if (static_cast<u32>(data[i]) == 0x80000000) {
                    result[i] = 0x7FFFFFFF;
                    qc_flag = true;
                } else {
                    result[i] = std::abs(data[i]);
                }
            }

            return qc_flag;
        });
//        EmitTwoOp(code, ctx, inst,
//                                       [&](auto Vresult, auto Voperand) { code.SQABS(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAbs64>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitOneArgumentFallbackWithSaturation(code, ctx, inst, [](VectorArray<s64>& result, const VectorArray<s64>& data) {
            bool qc_flag = false;

            for (size_t i = 0; i < result.size(); i++) {
                if (static_cast<u64>(data[i]) == 0x8000000000000000) {
                    result[i] = 0x7FFFFFFFFFFFFFFF;
                    qc_flag = true;
                } else {
                    result[i] = std::abs(data[i]);
                }
            }

            return qc_flag;
        });
//        EmitTwoOp(code, ctx, inst,
//                                       [&](auto Vresult, auto Voperand) { code.SQABS(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAccumulateUnsigned8>(BlockOfCode &code,
                                                                      EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                                   [&](auto &Vaccumulator, auto &Voperand) {
            // FIXME write to offsetof_fpsr_qc
            code.vsadd_bu(Vaccumulator, Vaccumulator, Voperand);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAccumulateUnsigned16>(BlockOfCode &code,
                                                                       EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                                    [&](auto &Vaccumulator, auto &Voperand) {
                                        code.vsadd_hu(Vaccumulator, Vaccumulator, Voperand);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAccumulateUnsigned32>(BlockOfCode &code,
                                                                       EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                                    [&](auto &Vaccumulator, auto &Voperand) {
                                        code.vsadd_wu(Vaccumulator, Vaccumulator, Voperand);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAccumulateUnsigned64>(BlockOfCode &code,
                                                                       EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                                    [&](auto &Vaccumulator, auto &Voperand) {
                                        code.vsadd_du(Vaccumulator, Vaccumulator, Voperand);
        });
    }
    template<bool is_rounding>
    static void EmitVectorSignedSaturatedDoublingMultiply16(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto x = ctx.reg_alloc.ReadQ(args[0]);
        auto y = ctx.reg_alloc.ReadQ(args[1]);
        auto upper_tmp = Vscratch0;
        auto lower_tmp = Vscratch1;
        auto result = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(x, y, result);

        code.vmul_h(upper_tmp, x, y);
        code.vmul_h(lower_tmp, x, y);

        //x, y released
        auto vtmp1 = *x;
        if constexpr (is_rounding) {
            code.vsrli_h(lower_tmp, lower_tmp, 14);
            code.add_imm(Xscratch0, code.zero, 0x0001000100010001, Xscratch1);
            code.vreplgr2vr_d(vtmp1, Xscratch0);
            code.vadd_h(lower_tmp, lower_tmp, vtmp1);
            code.vsrli_h(lower_tmp, lower_tmp, 1);
        } else {
            code.vsrli_h(lower_tmp, lower_tmp, 15);
        }
        code.vadd_h(upper_tmp, upper_tmp, upper_tmp);
        code.vadd_h(result, upper_tmp, lower_tmp);
        code.add_imm(Xscratch0, code.zero, 0x8000800080008000, Xscratch1);
        code.vreplgr2vr_d(vtmp1, Xscratch0);
        code.vseq_h(upper_tmp, result, vtmp1);
        code.vxor_v(result, result, upper_tmp);

        code.vmskltz_b(vtmp1, upper_tmp);

        code.vpickve2gr_wu(Xscratch1, vtmp1,0);

        code.ld_w(Xscratch0, Xstate, code.GetJitStateInfo().offsetof_fpsr_qc);
        code.or_(Xscratch0, Xscratch0, Xscratch1);
        code.st_w(Xscratch0, Xstate, code.GetJitStateInfo().offsetof_fpsr_qc);
    }
    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedDoublingMultiplyHigh16>(BlockOfCode &code,
                                                                         EmitContext &ctx, IR::Inst *inst) {
        EmitVectorSignedSaturatedDoublingMultiply16<false>(code, ctx, inst);
    }

    template<bool is_rounding>
    static void EmitVectorSignedSaturatedDoublingMultiply32(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto x = ctx.reg_alloc.ReadQ(args[0]);
        auto y = ctx.reg_alloc.ReadQ(args[1]);
        auto odds = Vscratch0;
        auto even = Vscratch1;
        auto result = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(x, y, result);

        code.vmulwod_d_w(odds, x, y);
        code.vmulwev_d_w(even, x, y);
        code.vadd_d(odds, odds, odds);
        code.vadd_d(even, even, even);

        if constexpr (is_rounding) {
            code.add_imm(Xscratch0, code.zero, 0x0000000080000000ULL, Xscratch1);
            code.vinsgr2vr_d(result, Xscratch0, 0);
            code.vinsgr2vr_d(result, Xscratch0, 1);
            code.vadd_d(odds, odds, result);
            code.vadd_d(even, even, result);
        }
        code.vsrli_d(result, odds, 32);

        // TODO change with vpermi_w
        code.vbsrl_v(even, even, 4);
        code.vpackev_w(result, result, even);

        auto mask = *x;
        auto bit = Xscratch2;
        // here Vscratch0 and Vscratch1 is released
        code.add_imm(Xscratch1, code.zero, 0x8000000080000000ULL, Xscratch2);
        code.vreplgr2vr_d(Vscratch0, Xscratch1);
        code.vseq_w(mask, result, Vscratch0);

        code.vxor_v(result, result, mask);
        code.vmskltz_b(Vscratch2, mask);
        code.vpickve2gr_wu(bit, Vscratch2,0);

        code.ld_w(Xscratch0, Xstate, code.GetJitStateInfo().offsetof_fpsr_qc);
        code.or_(Xscratch0, Xscratch0, bit);
        code.st_w(Xscratch0, Xstate, code.GetJitStateInfo().offsetof_fpsr_qc);

    }
    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedDoublingMultiplyHigh32>(BlockOfCode &code,
                                                                         EmitContext &ctx, IR::Inst *inst) {
        EmitVectorSignedSaturatedDoublingMultiply32<false>(code, ctx, inst);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignedSaturatedDoublingMultiplyHighRounding16>(BlockOfCode &code,
                                                                            EmitContext &ctx, IR::Inst *inst) {
        EmitVectorSignedSaturatedDoublingMultiply16<true>(code, ctx, inst);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignedSaturatedDoublingMultiplyHighRounding32>(BlockOfCode &code,
                                                                            EmitContext &ctx, IR::Inst *inst) {
        EmitVectorSignedSaturatedDoublingMultiply32<true>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedDoublingMultiplyLong16>(BlockOfCode &code,
                                                                         EmitContext &ctx, IR::Inst *inst) {
        ASSERT_FALSE("Unimplemented");
        (void)code;
        (void)ctx;
        (void)inst;
//        EmitThreeOp(code, ctx, inst,
//                                              [&](auto Vresult, auto Va, auto Vb) {
//                                                  code.SQDMULL(Vresult, Va, Vb);
//                                              });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedDoublingMultiplyLong32>(BlockOfCode &code,
                                                                         EmitContext &ctx, IR::Inst *inst) {
        ASSERT_FALSE("Unimplemented");
        (void)code;
        (void)ctx;
        (void)inst;
//        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
//                                                  code.SQDMULL(Vresult, Va, Vb);
//        });
    }
#define VSSNTU(op, tosize) \
    EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) {\
    Xbyak_loongarch64::Label set_qc, label_end;\
    code.vsat_##op##u(Vresult, Voperand, tosize);\
    code.vxor_v(Vscratch2, Vscratch2, Vscratch2);\
    code.vslt_##op(Vscratch0, Vresult, Vscratch2);\
    code.vnor_v(Vscratch0, Vscratch0, Vscratch2);\
    code.vmax_##op(Vresult, Vresult, Vscratch0);\
    code.vseq_##op(Vscratch1, Vresult, Voperand);\
    code.vnor_v(Vscratch1, Vscratch1, Vscratch2);\
    code.vsetnez_v(0 , Vscratch1);\
    code.bcnez(0, set_qc);\
    code.st_b(code.zero, Xstate, code.GetJitStateInfo().offsetof_fpsr_qc);\
    code.b(label_end);\
    code.L(set_qc);\
    code.addi_w(Xscratch0, code.zero, 1);\
    code.st_b(Xscratch0, Xstate, code.GetJitStateInfo().offsetof_fpsr_qc);\
    code.L(label_end);\
});
#define VSSNT(op, tosize) \
    EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) {\
    Xbyak_loongarch64::Label set_qc, label_end;\
    code.vsat_##op(Vresult, Voperand, tosize);\
    code.vxor_v(Vscratch2, Vscratch2, Vscratch2);\
    code.vslt_##op(Vscratch0, Vresult, Vscratch2);\
    code.vnor_v(Vscratch0, Vscratch0, Vscratch2);\
    code.vmax_##op(Vresult, Vresult, Vscratch0);\
    code.vseq_##op(Vscratch1, Vresult, Voperand);\
    code.vnor_v(Vscratch1, Vscratch1, Vscratch2);\
    code.vsetnez_v(0 , Vscratch1);\
    code.bcnez(0, set_qc);\
    code.st_b(code.zero, Xstate, code.GetJitStateInfo().offsetof_fpsr_qc);\
    code.b(label_end);\
    code.L(set_qc);\
    code.addi_w(Xscratch0, code.zero, 1);\
    code.st_b(Xscratch0, Xstate, code.GetJitStateInfo().offsetof_fpsr_qc);\
    code.L(label_end);\
});
    template<>
    void
    EmitIR<IR::Opcode::VectorSignedSaturatedNarrowToSigned16>(BlockOfCode &code,
                                                              EmitContext &ctx,
                                                              IR::Inst *inst) {
        VSSNT(h, 8);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignedSaturatedNarrowToSigned32>(BlockOfCode &code,
                                                              EmitContext &ctx,
                                                              IR::Inst *inst) {
        VSSNT(w, 16);

    }


    template<>
    void
    EmitIR<IR::Opcode::VectorSignedSaturatedNarrowToSigned64>(BlockOfCode &code,
                                                              EmitContext &ctx,
                                                              IR::Inst *inst) {
        VSSNT(d, 32);

    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedNarrowToUnsigned16>(BlockOfCode &code,
                                                                     EmitContext &ctx, IR::Inst *inst) {
        VSSNTU(h, 8);

    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedNarrowToUnsigned32>(BlockOfCode &code,
                                                                     EmitContext &ctx, IR::Inst *inst) {
        VSSNTU(w, 16);
    }



    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedNarrowToUnsigned64>(BlockOfCode &code,
                                                                     EmitContext &ctx, IR::Inst *inst) {
        VSSNTU(d, 32);
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedNeg8>(BlockOfCode &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                                       [&](auto &Vresult, auto &Voperand) {
                                           code.vxor_v(Vscratch2, Vscratch2, Vscratch2);
                                           code.vssub_b(Vresult, Vscratch2, Voperand);
                                       });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedNeg16>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                                       [&](auto &Vresult, auto &Voperand) {
                                           code.vxor_v(Vscratch2, Vscratch2, Vscratch2);
                                           code.vssub_h(Vresult, Vscratch2, Voperand);
                                       });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedNeg32>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                                       [&](auto &Vresult, auto &Voperand) {
                                           code.vxor_v(Vscratch2, Vscratch2, Vscratch2);
                                           code.vssub_w(Vresult, Vscratch2, Voperand);
                                       });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedNeg64>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst,
                                       [&](auto &Vresult, auto &Voperand) {
            code.vxor_v(Vscratch2, Vscratch2, Vscratch2);
            code.vssub_d(Vresult, Vscratch2, Voperand);
        });
    }

    template<typename T, typename U = std::make_unsigned_t<T>>
    static bool VectorSignedSaturatedShiftLeft(VectorArray<T> &dst, const VectorArray<T> &data,
                                               const VectorArray<T> &shift_values) {
        static_assert(std::is_signed_v<T>, "T must be signed.");

        bool qc_flag = false;

        constexpr size_t bit_size_minus_one = mcl::bitsizeof<T> - 1;

        const auto saturate = [bit_size_minus_one](T value) {
            return static_cast<T>((static_cast<U>(value) >> bit_size_minus_one) + (U{1} << bit_size_minus_one) - 1);
        };

        for (size_t i = 0; i < dst.size(); i++) {
            const T element = data[i];
            const T shift = std::clamp<T>(
                    static_cast<T>(mcl::bit::sign_extend<8>(static_cast<U>(shift_values[i] & 0xFF))),
                    -static_cast<T>(bit_size_minus_one), std::numeric_limits<T>::max());

            if (element == 0) {
                dst[i] = 0;
            } else if (shift < 0) {
                dst[i] = static_cast<T>(element >> -shift);
            } else if (static_cast<U>(shift) > bit_size_minus_one) {
                dst[i] = saturate(element);
                qc_flag = true;
            } else {
                const T shifted = T(U(element) << shift);

                if ((shifted >> shift) != element) {
                    dst[i] = saturate(element);
                    qc_flag = true;
                } else {
                    dst[i] = shifted;
                }
            }
        }

        return qc_flag;
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignedSaturatedShiftLeft8>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, VectorSignedSaturatedShiftLeft<s8>);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignedSaturatedShiftLeft16>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, VectorSignedSaturatedShiftLeft<s16>);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignedSaturatedShiftLeft32>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, VectorSignedSaturatedShiftLeft<s32>);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignedSaturatedShiftLeft64>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, VectorSignedSaturatedShiftLeft<s64>);
    }

    template<typename T, typename U = std::make_unsigned_t<T>>
    static bool
    VectorSignedSaturatedShiftLeftUnsigned(VectorArray<T> &dst, const VectorArray<T> &data, u8 shift_amount) {
        static_assert(std::is_signed_v<T>, "T must be signed.");

        bool qc_flag = false;
        for (size_t i = 0; i < dst.size(); i++) {
            const T element = data[i];
            const T shift = static_cast<T>(shift_amount);

            if (element == 0) {
                dst[i] = 0;
            } else if (element < 0) {
                dst[i] = 0;
                qc_flag = true;
            } else {
                const U shifted = static_cast<U>(element) << static_cast<U>(shift);
                const U shifted_test = shifted >> static_cast<U>(shift);

                if (shifted_test != static_cast<U>(element)) {
                    dst[i] = static_cast<T>(std::numeric_limits<U>::max());
                    qc_flag = true;
                } else {
                    dst[i] = shifted;
                }
            }
        }

        return qc_flag;
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedShiftLeftUnsigned8>(BlockOfCode &code,
                                                                     EmitContext &ctx, IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturationAndImmediate(code, ctx, inst,
                                                          VectorSignedSaturatedShiftLeftUnsigned<s8>);
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedShiftLeftUnsigned16>(BlockOfCode &code,
                                                                      EmitContext &ctx, IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturationAndImmediate(code, ctx, inst,
                                                          VectorSignedSaturatedShiftLeftUnsigned<s16>);
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedShiftLeftUnsigned32>(BlockOfCode &code,
                                                                      EmitContext &ctx, IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturationAndImmediate(code, ctx, inst,
                                                          VectorSignedSaturatedShiftLeftUnsigned<s32>);
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedShiftLeftUnsigned64>(BlockOfCode &code,
                                                                      EmitContext &ctx, IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturationAndImmediate(code, ctx, inst,
                                                          VectorSignedSaturatedShiftLeftUnsigned<s64>);
    }

    template<>
    void EmitIR<IR::Opcode::VectorSub8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(
                code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsub_b(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSub16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(
                code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsub_h(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSub32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(
                code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsub_w(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSub64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(
                code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsub_d(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorTable>(BlockOfCode &, EmitContext &, IR::Inst *inst) {
        // Do nothing. We *want* to hold on to the refcount for our arguments, so VectorTableLookup can use our arguments.
        ASSERT_MSG(inst->UseCount() == 1, "Table cannot be used multiple times");
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorTableLookup64>(BlockOfCode &code, EmitContext &ctx,
                                            IR::Inst *inst) {
        ASSERT(inst->GetArg(1).GetInst()->GetOpcode() == IR::Opcode::VectorTable);

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto table = ctx.reg_alloc.GetArgumentInfo(inst->GetArg(1).GetInst());
        const size_t table_size = std::count_if(table.begin(), table.end(),
                                                [](const auto &elem) { return !elem.IsVoid(); });
//        const bool is_defaults_zero = inst->GetArg(0).IsZero();

        auto lambda = [](const HalfVectorArray<u8>* table, HalfVectorArray<u8>& result, const HalfVectorArray<u8>& indicies, size_t table_size) {
            for (size_t i = 0; i < result.size(); ++i) {
                const size_t index = indicies[i] / table[0].size();
                const size_t elem = indicies[i] % table[0].size();
                if (index < table_size) {
                    result[i] = table[index][elem];
                }
            }
        };

        u64 stack_size = static_cast<u32>(6 * 8);

        auto result = ctx.reg_alloc.WriteQ(inst);
        auto defaults = ctx.reg_alloc.ReadQ(args[0]);
        auto indic = ctx.reg_alloc.ReadQ(args[2]);
        auto indicies = Vscratch0;
        RegAlloc::Realize(result);
        RegAlloc::Realize(defaults);
        if ( inst->GetArg(0).IsImmediate() || (!inst->GetArg(0).IsImmediate() && (inst->GetArg(0).GetInst()->GetName() != inst->GetArg(2).GetInst()->GetName()))) {
            RegAlloc::Realize(indic);
            indicies = indic;
        } else {
            indicies = defaults;
        }

        ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << Xscratch0.getIdx()), stack_size);

        for (size_t i = 0; i < table_size; ++i) {
            auto tmp_value = ctx.reg_alloc.ReadQ(table[i]);
            RegAlloc::Realize(tmp_value);
            code.vpickve2gr_d(Xscratch0, tmp_value, 0);
            code.st_d(Xscratch0, code.sp, i * 8);
        }

        code.vpickve2gr_d(Xscratch0, defaults, 0);
        code.st_d(Xscratch0, code.sp , 4 * 8);

        code.vpickve2gr_d(Xscratch0, indicies, 0);
        code.st_d(Xscratch0, code.sp , 5 * 8);


        code.add_d(code.a0, code.sp, code.zero);
        code.addi_d(code.a1, code.sp , 4 * 8);
        code.addi_d(code.a2, code.sp , 5 * 8);
        code.add_imm(code.a3, code.zero, table_size, Wscratch2);
        code.CallLambda(lambda);

        code.ld_d(Xscratch0, code.sp, 4 * 8);

        ABI_PopRegisters(code, ABI_CALLER_SAVE & ~( 1ull << Xscratch0.getIdx()) , stack_size);

        code.vinsgr2vr_d(result, Xscratch0, 0);



    }

    template<>
    void
    EmitIR<IR::Opcode::VectorTableLookup128>(BlockOfCode &code, EmitContext &ctx,
                                             IR::Inst *inst) {
        ASSERT(inst->GetArg(1).GetInst()->GetOpcode() == IR::Opcode::VectorTable);

        auto result = ctx.reg_alloc.WriteQ(inst);
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto table = ctx.reg_alloc.GetArgumentInfo(inst->GetArg(1).GetInst());

        const size_t table_size = std::count_if(table.begin(), table.end(),
                                                [](const auto &elem) { return !elem.IsVoid(); });

        auto lambda = [](const VectorArray<u8>* table, VectorArray<u8>& result, const VectorArray<u8>& indicies, size_t table_size) {
            for (size_t i = 0; i < result.size(); ++i) {
                const size_t index = indicies[i] / table[0].size();
                const size_t elem = indicies[i] % table[0].size();
                if (index < table_size) {
                    result[i] = table[index][elem];
                }
            }
        };
        auto defaults = ctx.reg_alloc.ReadQ(args[0]);
        auto indicies = ctx.reg_alloc.ReadQ(args[2]);
        RegAlloc::Realize(result, defaults, indicies);

        u64 stack_size = static_cast<u32>((table_size + 2) * 16);
        ABI_PushRegisters(code, ABI_CALLER_SAVE & ~(1ull << (result->getIdx() + 32)), stack_size);

        for (size_t i = 0; i < table_size; ++i) {
            auto tmp_value = ctx.reg_alloc.ReadQ(table[i]);
            RegAlloc::Realize(tmp_value);
            code.vst(tmp_value, code.sp, i * 16);
        }

        code.vst(defaults, code.sp , (table_size + 0) * 16);
        code.vst(indicies, code.sp , (table_size + 1) * 16);

        code.add_d(code.a0, code.sp, code.zero);
        code.addi_d(code.a1, code.sp , (table_size + 0) * 16);
        code.addi_d(code.a2, code.sp , (table_size + 1) * 16);
        code.add_imm(code.a3, code.zero, table_size, Wscratch2);
        code.CallLambda(lambda);
        code.vld(result, code.sp, (table_size + 0) * 16);

        ABI_PopRegisters(code, ABI_CALLER_SAVE & ~(1ull << (result->getIdx() + 32)), stack_size);

    }

    template<>
    void
    EmitIR<IR::Opcode::VectorTranspose8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const bool part = inst->GetArg(2).GetU1();
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            !part ? code.vpackev_b(Vresult, Vb, Va) : code.vpackod_b(Vresult, Vb, Va);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorTranspose16>(BlockOfCode &code, EmitContext &ctx,
                                          IR::Inst *inst) {
        const bool part = inst->GetArg(2).GetU1();
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            !part ? code.vpackev_h(Vresult, Vb, Va) : code.vpackod_h(Vresult, Vb, Va);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorTranspose32>(BlockOfCode &code, EmitContext &ctx,
                                          IR::Inst *inst) {
        const bool part = inst->GetArg(2).GetU1();
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            !part ? code.vpackev_w(Vresult, Vb, Va) : code.vpackod_w(Vresult, Vb, Va);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorTranspose64>(BlockOfCode &code, EmitContext &ctx,
                                          IR::Inst *inst) {
        const bool part = inst->GetArg(2).GetU1();
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            !part ? code.vpackev_d(Vresult, Vb, Va) : code.vpackod_d(Vresult, Vb, Va);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorUnsignedAbsoluteDifference8>(BlockOfCode &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                               [&](auto Vresult, auto Va, auto Vb) { code.vabsd_b(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorUnsignedAbsoluteDifference16>(BlockOfCode &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vabsd_h(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorUnsignedAbsoluteDifference32>(BlockOfCode &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vabsd_w(Vresult, Va, Vb);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedMultiply16>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedMultiply32>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedRecipEstimate>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitOneArgumentFallback(code, ctx, inst, [](VectorArray<u32> &result, const VectorArray<u32> &a) {
            for (size_t i = 0; i < result.size(); i++) {
                if ((a[i] & 0x80000000) == 0) {
                    result[i] = 0xFFFFFFFF;
                    continue;
                }

                const u32 input = mcl::bit::get_bits<23, 31>(a[i]);
                const u32 estimate = Common::RecipEstimate(input);

                result[i] = (0b100000000 | estimate) << 23;
            }
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorUnsignedRecipSqrtEstimate>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitOneArgumentFallback(code, ctx, inst, [](VectorArray<u32> &result, const VectorArray<u32> &a) {
            for (size_t i = 0; i < result.size(); i++) {
                if ((a[i] & 0xC0000000) == 0) {
                    result[i] = 0xFFFFFFFF;
                    continue;
                }

                const u32 input = mcl::bit::get_bits<23, 31>(a[i]);
                const u32 estimate = Common::RecipSqrtEstimate(input);

                result[i] = (0b100000000 | estimate) << 23;
            }
        });
    }


// Simple generic case for 8, 16, and 32-bit values. 64-bit values
// will need to be special-cased as we can't simply use a larger integral size.
    template<typename T, typename U = std::make_unsigned_t<T>>
    static bool EmitVectorUnsignedSaturatedAccumulateSigned(VectorArray<U> &result, const VectorArray<T> &lhs,
                                                            const VectorArray<T> &rhs) {
        static_assert(std::is_signed_v<T>, "T must be signed.");
        static_assert(mcl::bitsizeof<T> < 64, "T must be less than 64 bits in size.");

        bool qc_flag = false;

        for (size_t i = 0; i < result.size(); i++) {
            // We treat rhs' members as unsigned, so cast to unsigned before signed to inhibit sign-extension.
            // We use the unsigned equivalent of T, as we want zero-extension to occur, rather than a plain move.
            const s64 x = s64{lhs[i]};
            const s64 y = static_cast<s64>(static_cast<std::make_unsigned_t<U>>(rhs[i]));
            const s64 sum = x + y;

            if (sum > std::numeric_limits<U>::max()) {
                result[i] = std::numeric_limits<U>::max();
                qc_flag = true;
            } else if (sum < 0) {
                result[i] = std::numeric_limits<U>::min();
                qc_flag = true;
            } else {
                result[i] = static_cast<U>(sum);
            }
        }

        return qc_flag;
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedAccumulateSigned8>(BlockOfCode &code,
                                                                      EmitContext &ctx, IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, EmitVectorUnsignedSaturatedAccumulateSigned<s8>);
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedAccumulateSigned16>(BlockOfCode &code,
                                                                       EmitContext &ctx, IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst,
                                              EmitVectorUnsignedSaturatedAccumulateSigned<s16>);
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedAccumulateSigned32>(BlockOfCode &code,
                                                                       EmitContext &ctx, IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst,
                                              EmitVectorUnsignedSaturatedAccumulateSigned<s32>);
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedAccumulateSigned64>(BlockOfCode &code,
                                                                       EmitContext &ctx, IR::Inst *inst) {
        // TODO use vsadd?
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst,
                                              [](VectorArray<u64> &result, const VectorArray<u64> &lhs,
                                                 const VectorArray<u64> &rhs) {
                                                  bool qc_flag = false;

                                                  for (size_t i = 0; i < result.size(); i++) {
                                                      const u64 x = lhs[i];
                                                      const u64 y = rhs[i];
                                                      const u64 res = x + y;

                                                      // Check sign bits to determine if an overflow occurred.
                                                      if ((~x & y & ~res) & 0x8000000000000000) {
                                                          result[i] = UINT64_MAX;
                                                          qc_flag = true;
                                                      } else if ((x & ~y & res) & 0x8000000000000000) {
                                                          result[i] = 0;
                                                          qc_flag = true;
                                                      } else {
                                                          result[i] = res;
                                                      }
                                                  }

                                                  return qc_flag;
                                              });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorUnsignedSaturatedNarrow16>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitOneArgumentFallbackWithSaturation(code, ctx, inst,
                                              [](VectorArray<u8> &result, const VectorArray<u16> &a) {
                                                  result = {};
                                                  bool qc_flag = false;
                                                  for (size_t i = 0; i < a.size(); ++i) {
                                                      const u16 saturated = std::clamp<u16>(a[i], 0, 0xFF);
                                                      result[i] = static_cast<u8>(saturated);
                                                      qc_flag |= saturated != a[i];
                                                  }
                                                  return qc_flag;
                                              });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorUnsignedSaturatedNarrow32>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitOneArgumentFallbackWithSaturation(code, ctx, inst,
                                              [](VectorArray<u16> &result, const VectorArray<u32> &a) {
                                                  result = {};
                                                  bool qc_flag = false;
                                                  for (size_t i = 0; i < a.size(); ++i) {
                                                      const u32 saturated = std::clamp<u32>(a[i], 0, 0xFFFF);
                                                      result[i] = static_cast<u16>(saturated);
                                                      qc_flag |= saturated != a[i];
                                                  }
                                                  return qc_flag;
                                              });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorUnsignedSaturatedNarrow64>(BlockOfCode &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitOneArgumentFallbackWithSaturation(code, ctx, inst,
                                              [](VectorArray<u32> &result, const VectorArray<u64> &a) {
                                                  result = {};
                                                  bool qc_flag = false;
                                                  for (size_t i = 0; i < a.size(); ++i) {
                                                      const u64 saturated = std::clamp<u64>(a[i], 0, 0xFFFFFFFF);
                                                      result[i] = static_cast<u32>(saturated);
                                                      qc_flag |= saturated != a[i];
                                                  }
                                                  return qc_flag;
                                              });
    }

    template<typename T, typename S = std::make_signed_t<T>>
    static bool VectorUnsignedSaturatedShiftLeft(VectorArray<T> &dst, const VectorArray<T> &data,
                                                 const VectorArray<T> &shift_values) {
        static_assert(std::is_unsigned_v<T>, "T must be an unsigned type.");

        bool qc_flag = false;

        constexpr size_t bit_size = mcl::bitsizeof<T>;
        constexpr S negative_bit_size = -static_cast<S>(bit_size);

        for (size_t i = 0; i < dst.size(); i++) {
            const T element = data[i];
            const S shift = std::clamp(
                    static_cast<S>(mcl::bit::sign_extend<8>(static_cast<T>(shift_values[i] & 0xFF))),
                    negative_bit_size, std::numeric_limits<S>::max());

            if (element == 0 || shift <= negative_bit_size) {
                dst[i] = 0;
            } else if (shift < 0) {
                dst[i] = static_cast<T>(element >> -shift);
            } else if (shift >= static_cast<S>(bit_size)) {
                dst[i] = std::numeric_limits<T>::max();
                qc_flag = true;
            } else {
                const T shifted = element << shift;

                if ((shifted >> shift) != element) {
                    dst[i] = std::numeric_limits<T>::max();
                    qc_flag = true;
                } else {
                    dst[i] = shifted;
                }
            }
        }

        return qc_flag;
    }


    template<>
    void
    EmitIR<IR::Opcode::VectorUnsignedSaturatedShiftLeft8>(BlockOfCode &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, VectorUnsignedSaturatedShiftLeft<u8>);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorUnsignedSaturatedShiftLeft16>(BlockOfCode &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, VectorUnsignedSaturatedShiftLeft<u16>);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorUnsignedSaturatedShiftLeft32>(BlockOfCode &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, VectorUnsignedSaturatedShiftLeft<u32>);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorUnsignedSaturatedShiftLeft64>(BlockOfCode &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
//    EmitThreeOp(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UQSHL(Vresult, Va, Vb); });
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, VectorUnsignedSaturatedShiftLeft<u64>);

    }

    template<>
    void
    EmitIR<IR::Opcode::VectorZeroExtend8>(BlockOfCode &code, EmitContext &ctx,
                                          IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vilvl_b(Vresult, Vresult, Voperand);
//        code.UXTL(Vresult, Voperand);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorZeroExtend16>(BlockOfCode &code, EmitContext &ctx,
                                           IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vilvl_h(Vresult, Vresult, Voperand);
//        code.UXTL(Vresult, Voperand);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorZeroExtend32>(BlockOfCode &code, EmitContext &ctx,
                                           IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Voperand) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vilvl_w(Vresult, Vresult, Voperand);
            // FIXME use which?
//        code.vsllwil_hu_bu(Vresult, Voperand, 0);?
//        code.UXTL(Vresult, Voperand);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorZeroExtend64>(BlockOfCode &code, EmitContext &ctx,
                                           IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Qresult, auto &Qoperand) {
            code.vor_v(Qresult, Qoperand, Qoperand);
//        code.FMOV(Qresult->toD(), Qoperand->toD()); });
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorZeroUpper>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Qresult, auto &Qoperand) {
            code.vbsll_v(Qoperand, Qoperand, 8);
            code.vbsrl_v(Qresult, Qoperand, 8);
        });
    }

    template<>
    void EmitIR<IR::Opcode::ZeroVector>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(Qresult);
        code.vreplgr2vr_d(Qresult, code.zero);
    }

}  // namespace Dynarmic::Backend::LoongArch64
