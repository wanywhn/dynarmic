/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/loongarch64/a32_jitstate.h"
#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/emit_loongarch64.h"
#include "dynarmic/backend/loongarch64/emit_context.h"
#include "dynarmic/backend/loongarch64/reg_alloc.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"
#include <mcl/type_traits/integer_of_size.hpp>
#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"
#include "nzcv_util.h"

namespace Dynarmic::Backend::LoongArch64 {

    using namespace Xbyak_loongarch64::util;

    namespace {

//    template<Op op, size_t size, bool has_overflow_inst = false>
//    void EmitSignedSaturatedOp(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
//        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
//
//        Xbyak::Reg result = ctx.reg_alloc.UseScratchGpr(args[0]).changeBit(size);
//        Xbyak::Reg addend = ctx.reg_alloc.UseGpr(args[1]).changeBit(size);
//        Xbyak::Reg overflow = ctx.reg_alloc.ScratchGpr().changeBit(size);
//
//        constexpr u64 int_max = static_cast<u64>(std::numeric_limits<mcl::signed_integer_of_size<size>>::max());
//        if constexpr (size < 64) {
//            code.xor_(overflow.cvt32(), overflow.cvt32());
//            code.bt(result.cvt32(), size - 1);
//            code.adc(overflow.cvt32(), int_max);
//        } else {
//            code.mov(overflow, int_max);
//            code.bt(result, 63);
//            code.adc(overflow, 0);
//        }
//
//        // overflow now contains 0x7F... if a was positive, or 0x80... if a was negative
//
//        if constexpr (op == Op::Add) {
//            code.add(result, addend);
//        } else {
//            code.sub(result, addend);
//        }
//
//        if constexpr (size == 8) {
//            code.cmovo(result.cvt32(), overflow.cvt32());
//        } else {
//            code.cmovo(result, overflow);
//        }
//
//        code.seto(overflow.cvt8());
//        if constexpr (has_overflow_inst) {
//            if (const auto overflow_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetOverflowFromOp)) {
//                ctx.reg_alloc.DefineValue(overflow_inst, overflow);
//            }
//        } else {
//            code.or_(code.byte[code.r15 + code.GetJitStateInfo().offsetof_fpsr_qc], overflow.cvt8());
//        }
//
//        ctx.reg_alloc.DefineValue(inst, result);
//    }

        template<NZCV::Op op, size_t size>
        void EmitUnsignedSaturatedOp(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
            auto args = ctx.reg_alloc.GetArgumentInfo(inst);

            auto Wresult = ctx.reg_alloc.WriteX(inst);
            auto Wa = ctx.reg_alloc.ReadX(args[0]);
            auto Wb = ctx.reg_alloc.ReadX(args[1]);
            auto overflow = ctx.reg_alloc.AllocScratchGpr();
            Xbyak_loongarch64::Label skipSaturate;

            RegAlloc::Realize(Wresult, Wa, Wb, overflow);

            constexpr u64 boundary =
                    op == NZCV::Op::Add ? std::numeric_limits<mcl::unsigned_integer_of_size<size>>::max() : 0;

            NZCV::calNZCV<op, size, NZCV::arm_v_flag_mask, true>(code, Wresult, Wa, Wb, overflow);

            code.beqz(overflow, skipSaturate);
            code.add_imm(Wresult, code.zero, boundary, Xscratch0);
            code.L(skipSaturate);
            code.st_w(overflow, code.sp, code.GetJitStateInfo().offsetof_fpsr_qc);

        }

    }  // anonymous namespace

    template<>
    void EmitIR<IR::Opcode::SignedSaturatedAddWithFlag32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const auto overflow_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetOverflowFromOp);
        ASSERT(overflow_inst);

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Wa = ctx.reg_alloc.ReadW(args[0]);
        auto Wb = ctx.reg_alloc.ReadW(args[1]);
        auto Woverflow = ctx.reg_alloc.WriteW(overflow_inst);
        RegAlloc::Realize(Wresult, Wa, Wb, Woverflow);
        ctx.reg_alloc.SpillFlags();


        Xbyak_loongarch64::Label overflowLable;
        code.add_w(Wresult, *Wa, Wb);
        code.xor_(Wscratch1, Wresult, Wa);
        code.xor_(Wscratch2, Wresult, Wb);
        code.and_(Wscratch0, Wscratch1, Wscratch2);

        code.slt(Wscratch2, Wscratch0, code.zero);
        code.maskeqz(Woverflow, Wscratch2, Wscratch2);

        code.srai_w(Wscratch0, Wresult, 31);
        code.add_imm(Wscratch1, code.zero, 0x8000'0000, Wscratch2);
        code.xor_(Wscratch0, Wscratch0, Wscratch1);
        code.slt(Wscratch2, Wscratch0, code.zero);
        code.maskeqz(Wscratch1, Wscratch0, Wscratch2);
        code.masknez(Wresult, Wresult, Wscratch2);
        code.add_d(Wresult, Wresult, Wscratch1);
    }

    template<>
    void EmitIR<IR::Opcode::SignedSaturatedSubWithFlag32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const auto overflow_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetOverflowFromOp);
        ASSERT(overflow_inst);

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Wa = ctx.reg_alloc.ReadW(args[0]);
        auto Wb = ctx.reg_alloc.ReadW(args[1]);
        auto Woverflow = ctx.reg_alloc.WriteW(overflow_inst);
        RegAlloc::Realize(Wresult, Wa, Wb, Woverflow);
        ctx.reg_alloc.SpillFlags();

        code.sub_d(Wresult, *Wa, Wb);
        code.xor_(Wscratch1, Wresult, Wa);
        code.xor_(Wscratch2, Wresult, Wb);
        code.and_(Wscratch0, Wscratch1, Wscratch2);

        code.slt(Wscratch2, Wa, Wb);
        code.maskeqz(Woverflow, Wscratch2, Wscratch2);

        code.srai_w(Wscratch0, Wresult, 31);
        code.add_imm(Wscratch1, code.zero, 0x8000'0000, Wscratch2);
        code.xor_(Wscratch0, Wscratch0, Wscratch1);
        code.slt(Wscratch2, Wscratch0, code.zero);
        code.maskeqz(Wscratch1, Wscratch0, Wscratch2);
        code.masknez(Wresult, Wresult, Wscratch2);
        code.add_d(Wresult, Wresult, Wscratch1);
    }

    template<>
    void EmitIR<IR::Opcode::SignedSaturation>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const auto overflow_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetOverflowFromOp);

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        const size_t N = args[1].GetImmediateU8();
        ASSERT(N >= 1 && N <= 32);

        if (N == 32) {
            ctx.reg_alloc.DefineAsExisting(inst, args[0]);
            if (overflow_inst) {
                auto Woverflow = ctx.reg_alloc.WriteW(overflow_inst);
                RegAlloc::Realize(Woverflow);
                code.add_d(*Woverflow, code.zero, code.zero);
            }
            return;
        }

        const u32 positive_saturated_value = (1u << (N - 1)) - 1;
        const u32 negative_saturated_value = ~u32{0} << (N - 1);

        auto Woperand = ctx.reg_alloc.ReadW(args[0]);
        auto Wresult = ctx.reg_alloc.WriteW(inst);
        RegAlloc::Realize(Woperand, Wresult);
        ctx.reg_alloc.SpillFlags();

        code.add_imm(Wscratch0, code.zero, negative_saturated_value, Wscratch2);
        code.slt(Wscratch2, Wscratch0, Woperand);
        code.maskeqz(Wscratch0, Woperand, Wscratch2);
        code.masknez(Wscratch1, Wscratch0, Wscratch2);
        code.add_d(Wresult, Wscratch0, Wscratch1);
//    code.CMP(*Woperand, Wscratch0);
//    code.CSEL(Wresult, Woperand, Wscratch0, GT);

        code.add_imm(Wscratch1, code.zero, positive_saturated_value, Wscratch2);
        code.slt(Wscratch2, *Woperand, Wscratch1);
        code.maskeqz(Wscratch0, Wresult, Wscratch2);
        code.masknez(Wscratch1, Wscratch0, Wscratch2);
        code.add_d(Wresult, Wscratch0, Wscratch1);
//    code.CMP(*Woperand, Wscratch1);
//    code.CSEL(Wresult, Wresult, Wscratch1, LT);

        if (overflow_inst) {
            auto Woverflow = ctx.reg_alloc.WriteW(overflow_inst);
            RegAlloc::Realize(Woverflow);
            code.sub_d(Wscratch1, Wresult, Woperand);
            code.sltu(Woverflow, code.zero, Wscratch1);
//        code.CMP(*Wresult, Woperand);
//        code.CSET(Woverflow, NE);
        }
    }

    template<>
    void EmitIR<IR::Opcode::UnsignedSaturation>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const auto overflow_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetOverflowFromOp);

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Wresult = ctx.reg_alloc.WriteW(inst);
        auto Woperand = ctx.reg_alloc.ReadW(args[0]);
        RegAlloc::Realize(Wresult, Woperand);
        ctx.reg_alloc.SpillFlags();

        const size_t N = args[1].GetImmediateU8();
        ASSERT(N <= 31);
        const u32 saturated_value = (1u << N) - 1;

        code.slt(Wscratch2, code.zero, Woperand);
        code.maskeqz(Wresult, Woperand, Wscratch2);
//    code.CMP(*Woperand, 0);
//    code.CSEL(Wresult, Woperand, code.zero, GT);
        code.add_imm(Wscratch0, code.zero, saturated_value, Wscratch2);
        code.slt(Wresult, Woperand, Wscratch0);
        if (overflow_inst) {
            auto Woverflow = ctx.reg_alloc.WriteW(overflow_inst);
            RegAlloc::Realize(Woverflow);
            code.sltu(Woverflow, Wscratch0, Woperand);
//        code.CSET(Woverflow, HI);
        }
        code.maskeqz(Wscratch1, Wresult, Wresult);
        code.masknez(Wscratch2, Wscratch0, Wresult);
        code.add_d(Wresult, Wscratch1, Wscratch2);
//    code.CMP(*Woperand, Wscratch0);
//    code.CSEL(Wresult, Wresult, Wscratch0, LT);


    }

    template<>
    void EmitIR<IR::Opcode::SignedSaturatedAdd8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitUnsignedSaturatedOp<NZCV::Op::Add, 8>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::SignedSaturatedAdd16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitUnsignedSaturatedOp<NZCV::Op::Add, 16>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::SignedSaturatedAdd32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitUnsignedSaturatedOp<NZCV::Op::Add, 32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::SignedSaturatedAdd64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitUnsignedSaturatedOp<NZCV::Op::Add, 64>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::SignedSaturatedDoublingMultiplyReturnHigh16>(BlockOfCode &code, EmitContext &ctx,
                                                                         IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::SignedSaturatedDoublingMultiplyReturnHigh32>(BlockOfCode &code, EmitContext &ctx,
                                                                         IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::SignedSaturatedSub8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitUnsignedSaturatedOp<NZCV::Op::Sub, 8>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::SignedSaturatedSub16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitUnsignedSaturatedOp<NZCV::Op::Sub, 16>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::SignedSaturatedSub32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitUnsignedSaturatedOp<NZCV::Op::Sub, 32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::SignedSaturatedSub64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitUnsignedSaturatedOp<NZCV::Op::Sub, 64>(code, ctx, inst);
        // TODO Signed ?
    }

    template<>
    void EmitIR<IR::Opcode::UnsignedSaturatedAdd8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitUnsignedSaturatedOp<NZCV::Op::Add, 8>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::UnsignedSaturatedAdd16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitUnsignedSaturatedOp<NZCV::Op::Add, 16>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::UnsignedSaturatedAdd32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitUnsignedSaturatedOp<NZCV::Op::Add, 32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::UnsignedSaturatedAdd64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitUnsignedSaturatedOp<NZCV::Op::Add, 64>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::UnsignedSaturatedSub8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitUnsignedSaturatedOp<NZCV::Op::Sub, 8>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::UnsignedSaturatedSub16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitUnsignedSaturatedOp<NZCV::Op::Sub, 16>(code, ctx, inst);
        // TODO fix in CODE_WD
    }

    template<>
    void EmitIR<IR::Opcode::UnsignedSaturatedSub32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitUnsignedSaturatedOp<NZCV::Op::Sub, 32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::UnsignedSaturatedSub64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitUnsignedSaturatedOp<NZCV::Op::Sub, 64>(code, ctx, inst);
    }

}  // namespace Dynarmic::Backend::LoongArch64
