/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/loongarch64/a32_jitstate.h"
#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/emit_arm64.h"
#include "dynarmic/backend/loongarch64/emit_context.h"
#include "dynarmic/backend/loongarch64/reg_alloc.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"
#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"

namespace Dynarmic::Backend::LoongArch64 {

using namespace Xbyak_loongarch64::util;

template<>
void EmitIR<IR::Opcode::SignedSaturatedAddWithFlag32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto overflow_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetOverflowFromOp);
    ASSERT(overflow_inst);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Wresult = ctx.reg_alloc.WriteW(inst);
    auto Wa = ctx.reg_alloc.ReadW(args[0]);
    auto Wb = ctx.reg_alloc.ReadW(args[1]);
    auto Woverflow = ctx.reg_alloc.WriteW(overflow_inst);
    RegAlloc::Realize(Wresult, Wa, Wb, Woverflow);
    ctx.reg_alloc.SpillFlags();

    code.ADDS(Wresult, *Wa, Wb);
    code.srai_w(Wscratch0, Wresult, 31);
    code.EOR(Wscratch0, Wscratch0, 0x8000'0000);
    code.CSEL(Wresult, Wresult, Wscratch0, VC);
    code.CSET(Woverflow, VS);
}

template<>
void EmitIR<IR::Opcode::SignedSaturatedSubWithFlag32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto overflow_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetOverflowFromOp);
    ASSERT(overflow_inst);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Wresult = ctx.reg_alloc.WriteW(inst);
    auto Wa = ctx.reg_alloc.ReadW(args[0]);
    auto Wb = ctx.reg_alloc.ReadW(args[1]);
    auto Woverflow = ctx.reg_alloc.WriteW(overflow_inst);
    RegAlloc::Realize(Wresult, Wa, Wb, Woverflow);
    ctx.reg_alloc.SpillFlags();

    code.SUBS(Wresult, *Wa, Wb);
    code.srai_w(Wscratch0, Wresult, 31);
    code.EOR(Wscratch0, Wscratch0, 0x8000'0000);
    code.CSEL(Wresult, Wresult, Wscratch0, VC);
    code.CSET(Woverflow, VS);
}

template<>
void EmitIR<IR::Opcode::SignedSaturation>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
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

    code.add_d(Wscratch0, negative_saturated_value, code.zero);
    code.add_d(Wscratch1, positive_saturated_value, code.zero);
    code.CMP(*Woperand, Wscratch0);
    code.CSEL(Wresult, Woperand, Wscratch0, GT);
    code.CMP(*Woperand, Wscratch1);
    code.CSEL(Wresult, Wresult, Wscratch1, LT);

    if (overflow_inst) {
        auto Woverflow = ctx.reg_alloc.WriteW(overflow_inst);
        RegAlloc::Realize(Woverflow);
        code.CMP(*Wresult, Woperand);
        code.CSET(Woverflow, NE);
    }
}

template<>
void EmitIR<IR::Opcode::UnsignedSaturation>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto overflow_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetOverflowFromOp);

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Wresult = ctx.reg_alloc.WriteW(inst);
    auto Woperand = ctx.reg_alloc.ReadW(args[0]);
    RegAlloc::Realize(Wresult, Woperand);
    ctx.reg_alloc.SpillFlags();

    const size_t N = args[1].GetImmediateU8();
    ASSERT(N <= 31);
    const u32 saturated_value = (1u << N) - 1;

    code.add_d(Wscratch0, saturated_value, code.zero);
    code.CMP(*Woperand, 0);
    code.CSEL(Wresult, Woperand, code.zero, GT);
    code.CMP(*Woperand, Wscratch0);
    code.CSEL(Wresult, Wresult, Wscratch0, LT);

    if (overflow_inst) {
        auto Woverflow = ctx.reg_alloc.WriteW(overflow_inst);
        RegAlloc::Realize(Woverflow);
        code.CSET(Woverflow, HI);
    }
}

template<>
void EmitIR<IR::Opcode::SignedSaturatedAdd8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::SignedSaturatedAdd16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::SignedSaturatedAdd32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::SignedSaturatedAdd64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::SignedSaturatedDoublingMultiplyReturnHigh16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::SignedSaturatedDoublingMultiplyReturnHigh32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::SignedSaturatedSub8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::SignedSaturatedSub16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::SignedSaturatedSub32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::SignedSaturatedSub64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::UnsignedSaturatedAdd8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::UnsignedSaturatedAdd16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::UnsignedSaturatedAdd32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::UnsignedSaturatedAdd64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::UnsignedSaturatedSub8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::UnsignedSaturatedSub16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::UnsignedSaturatedSub32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

template<>
void EmitIR<IR::Opcode::UnsignedSaturatedSub64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    (void)code;
    (void)ctx;
    (void)inst;
    ASSERT_FALSE("Unimplemented");
}

}  // namespace Dynarmic::Backend::LoongArch64
