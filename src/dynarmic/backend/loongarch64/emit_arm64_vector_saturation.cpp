/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <mcl/mp/metavalue/lift_value.hpp>

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
#include "xbyak_loongarch64_util.h"

namespace Dynarmic::Backend::LoongArch64 {

using namespace Xbyak_loongarch64::util;

template<size_t size, typename EmitFn>
static void Emit(Xbyak_loongarch64::CodeGenerator&, EmitContext& ctx, IR::Inst* inst, EmitFn emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Qresult = ctx.reg_alloc.WriteQ(inst);
    auto Qa = ctx.reg_alloc.ReadQ(args[0]);
    auto Qb = ctx.reg_alloc.ReadQ(args[1]);
    RegAlloc::Realize(Qresult, Qa, Qb);
    ctx.fpsr.Load();

    if constexpr (size == 8) {
        emit(Qresult->B16(), Qa->B16(), Qb->B16());
    } else if constexpr (size == 16) {
        emit(Qresult->H8(), Qa->H8(), Qb->H8());
    } else if constexpr (size == 32) {
        emit(Qresult->S4(), Qa->S4(), Qb->S4());
    } else if constexpr (size == 64) {
        emit(Qresult->D2(), Qa->D2(), Qb->D2());
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<size>>);
    }
}

template<>
void EmitIR<IR::Opcode::VectorSignedSaturatedAdd8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SQADD(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorSignedSaturatedAdd16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SQADD(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorSignedSaturatedAdd32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SQADD(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorSignedSaturatedAdd64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SQADD(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorSignedSaturatedSub8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SQSUB(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorSignedSaturatedSub16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SQSUB(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorSignedSaturatedSub32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SQSUB(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorSignedSaturatedSub64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SQSUB(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorUnsignedSaturatedAdd8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UQADD(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorUnsignedSaturatedAdd16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UQADD(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorUnsignedSaturatedAdd32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UQADD(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorUnsignedSaturatedAdd64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UQADD(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorUnsignedSaturatedSub8>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UQSUB(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorUnsignedSaturatedSub16>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UQSUB(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorUnsignedSaturatedSub32>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UQSUB(Vresult, Va, Vb); });
}

template<>
void EmitIR<IR::Opcode::VectorUnsignedSaturatedSub64>(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    Emit<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UQSUB(Vresult, Va, Vb); });
}

}  // namespace Dynarmic::Backend::LoongArch64
