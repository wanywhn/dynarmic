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
#include "mcl/type_traits/function_info.hpp"

namespace Dynarmic::Backend::LoongArch64 {

    using namespace Xbyak_loongarch64::util;

    template<typename EmitFn>
    static void EmitTwoOp(Xbyak_loongarch64::CodeGenerator &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        auto Qoperand = ctx.reg_alloc.ReadQ(args[0]);
        RegAlloc::Realize(Qresult, Qoperand);

        emit(Qresult, Qoperand);
    }

    template<size_t size, typename EmitFn>
    static void
    EmitTwoOpArranged(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        EmitTwoOp(code, ctx, inst, [&](auto &Qresult, auto &Qoperand) {
            if constexpr (size == 8) {
                emit(Qresult->B16(), Qoperand->B16());
            } else if constexpr (size == 16) {
                emit(Qresult->H8(), Qoperand->H8());
            } else if constexpr (size == 32) {
                emit(Qresult->S4(), Qoperand->S4());
            } else if constexpr (size == 64) {
                emit(Qresult->D2(), Qoperand->D2());
            } else {
                static_assert(Common::always_false_v<mcl::mp::lift_value<size>>);
            }
        });
    }

    template<size_t size, typename EmitFn>
    static void
    EmitTwoOpArrangedSaturated(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        EmitTwoOpArranged<size>(code, ctx, inst, [&](auto Vresult, auto Voperand) {
            ctx.fpsr.Load();
            emit(Vresult, Voperand);
        });
    }

    template<size_t size, typename EmitFn>
    static void
    EmitTwoOpArrangedWiden(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        EmitTwoOp(code, ctx, inst, [&](auto &Qresult, auto &Qoperand) {
            if constexpr (size == 8) {
                emit(Qresult->H8(), Qoperand->toD().B8());
            } else if constexpr (size == 16) {
                emit(Qresult->S4(), Qoperand->toD().H4());
            } else if constexpr (size == 32) {
                emit(Qresult->D2(), Qoperand->toD().S2());
            } else {
                static_assert(Common::always_false_v<mcl::mp::lift_value<size>>);
            }
        });
    }

    template<size_t size, typename EmitFn>
    static void
    EmitTwoOpArrangedNarrow(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        EmitTwoOp(code, ctx, inst, [&](auto &Qresult, auto &Qoperand) {
            if constexpr (size == 16) {
                emit(Qresult->toD().B8(), Qoperand->H8());
            } else if constexpr (size == 32) {
                emit(Qresult->toD().H4(), Qoperand->S4());
            } else if constexpr (size == 64) {
                emit(Qresult->toD().S2(), Qoperand->D2());
            } else {
                static_assert(Common::always_false_v<mcl::mp::lift_value<size>>);
            }
        });
    }

    template<size_t size, typename EmitFn>
    static void
    EmitTwoOpArrangedSaturatedNarrow(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst,
                                     EmitFn emit) {
        EmitTwoOpArrangedNarrow<size>(code, ctx, inst, [&](auto Vresult, auto Voperand) {
            ctx.fpsr.Load();
            emit(Vresult, Voperand);
        });
    }

    template<size_t size, typename EmitFn>
    static void
    EmitTwoOpArrangedPairWiden(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        EmitTwoOp(code, ctx, inst, [&](auto &Qresult, auto &Qoperand) {
            if constexpr (size == 8) {
                emit(Qresult->H8(), Qoperand->B16());
            } else if constexpr (size == 16) {
                emit(Qresult->S4(), Qoperand->H8());
            } else if constexpr (size == 32) {
                emit(Qresult->D2(), Qoperand->S4());
            } else {
                static_assert(Common::always_false_v<mcl::mp::lift_value<size>>);
            }
        });
    }

    template<size_t size, typename EmitFn>
    static void
    EmitTwoOpArrangedLower(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        EmitTwoOp(code, ctx, inst, [&](auto &Qresult, auto &Qoperand) {
            if constexpr (size == 8) {
                emit(Qresult->toD().B8(), Qoperand->toD().B8());
            } else if constexpr (size == 16) {
                emit(Qresult->toD().H4(), Qoperand->toD().H4());
            } else if constexpr (size == 32) {
                emit(Qresult->toD().S2(), Qoperand->toD().S2());
            } else {
                static_assert(Common::always_false_v<mcl::mp::lift_value<size>>);
            }
        });
    }

    template<typename EmitFn>
    static void EmitThreeOp(Xbyak_loongarch64::CodeGenerator &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        auto Qa = ctx.reg_alloc.ReadQ(args[0]);
        auto Qb = ctx.reg_alloc.ReadQ(args[1]);
        RegAlloc::Realize(Qresult, Qa, Qb);

        emit(Qresult, Qa, Qb);
    }

    template<size_t size, typename EmitFn>
    static void
    EmitThreeOpArranged(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        EmitThreeOp(code, ctx, inst, [&](auto &Qresult, auto &Qa, auto &Qb) {
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
        });
    }

    template<size_t size, typename EmitFn>
    static void EmitThreeOpArrangedSaturated(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst,
                                             EmitFn emit) {
        EmitThreeOpArranged<size>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            ctx.fpsr.Load();
            emit(Vresult, Va, Vb);
        });
    }

    template<size_t size, typename EmitFn>
    static void
    EmitThreeOpArrangedWiden(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        EmitThreeOp(code, ctx, inst, [&](auto &Qresult, auto &Qa, auto &Qb) {
            if constexpr (size == 8) {
                emit(Qresult->H8(), Qa->toD().B8(), Qb->toD().B8());
            } else if constexpr (size == 16) {
                emit(Qresult->S4(), Qa->toD().H4(), Qb->toD().H4());
            } else if constexpr (size == 32) {
                emit(Qresult->D2(), Qa->toD().S2(), Qb->toD().S2());
            } else if constexpr (size == 64) {
                emit(Qresult->Q1(), Qa->toD().D1(), Qb->toD().D1());
            } else {
                static_assert(Common::always_false_v<mcl::mp::lift_value<size>>);
            }
        });
    }

    template<size_t size, typename EmitFn>
    static void
    EmitThreeOpArrangedSaturatedWiden(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst,
                                      EmitFn emit) {
        EmitThreeOpArrangedWiden<size>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            ctx.fpsr.Load();
            emit(Vresult, Va, Vb);
        });
    }

    template<size_t size, typename EmitFn>
    static void
    EmitThreeOpArrangedLower(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        EmitThreeOp(code, ctx, inst, [&](auto &Qresult, auto &Qa, auto &Qb) {
            if constexpr (size == 8) {
                emit(Qresult->toD().B8(), Qa->toD().B8(), Qb->toD().B8());
            } else if constexpr (size == 16) {
                emit(Qresult->toD().H4(), Qa->toD().H4(), Qb->toD().H4());
            } else if constexpr (size == 32) {
                emit(Qresult->toD().S2(), Qa->toD().S2(), Qb->toD().S2());
            } else {
                static_assert(Common::always_false_v<mcl::mp::lift_value<size>>);
            }
        });
    }

    template<size_t size, typename EmitFn>
    static void
    EmitSaturatedAccumulate(Xbyak_loongarch64::CodeGenerator &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qaccumulator = ctx.reg_alloc.ReadWriteQ(args[1], inst);  // NB: Swapped
        auto Qoperand = ctx.reg_alloc.ReadQ(args[0]);                 // NB: Swapped
        RegAlloc::Realize(Qaccumulator, Qoperand);
        ctx.fpsr.Load();

        if constexpr (size == 8) {
            emit(Qaccumulator->B16(), Qoperand->B16());
        } else if constexpr (size == 16) {
            emit(Qaccumulator->H8(), Qoperand->H8());
        } else if constexpr (size == 32) {
            emit(Qaccumulator->S4(), Qoperand->S4());
        } else if constexpr (size == 64) {
            emit(Qaccumulator->D2(), Qoperand->D2());
        } else {
            static_assert(Common::always_false_v<mcl::mp::lift_value<size>>);
        }
    }

    template<size_t size, typename EmitFn>
    static void EmitImmShift(Xbyak_loongarch64::CodeGenerator &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        auto Qoperand = ctx.reg_alloc.ReadQ(args[0]);
        const u8 shift_amount = args[1].GetImmediateU8();
        RegAlloc::Realize(Qresult, Qoperand);

        if constexpr (size == 8) {
            emit(Qresult->B16(), Qoperand->B16(), shift_amount);
        } else if constexpr (size == 16) {
            emit(Qresult->H8(), Qoperand->H8(), shift_amount);
        } else if constexpr (size == 32) {
            emit(Qresult->S4(), Qoperand->S4(), shift_amount);
        } else if constexpr (size == 64) {
            emit(Qresult->D2(), Qoperand->D2(), shift_amount);
        } else {
            static_assert(Common::always_false_v<mcl::mp::lift_value<size>>);
        }
    }

    template<size_t size, typename EmitFn>
    static void
    EmitImmShiftSaturated(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        EmitImmShift<size>(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            ctx.fpsr.Load();
            emit(Vresult, Voperand, shift_amount);
        });
    }

    template<size_t size, typename EmitFn>
    static void EmitReduce(Xbyak_loongarch64::CodeGenerator &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Vresult = ctx.reg_alloc.WriteVec<size>(inst);
        auto Qoperand = ctx.reg_alloc.ReadQ(args[0]);
        RegAlloc::Realize(Vresult, Qoperand);

        if constexpr (size == 8) {
            emit(Vresult, Qoperand->B16());
        } else if constexpr (size == 16) {
            emit(Vresult, Qoperand->H8());
        } else if constexpr (size == 32) {
            emit(Vresult, Qoperand->S4());
        } else if constexpr (size == 64) {
            emit(Vresult, Qoperand->D2());
        } else {
            static_assert(Common::always_false_v<mcl::mp::lift_value<size>>);
        }
    }

    template<size_t size, typename EmitFn>
    static void EmitGetElement(Xbyak_loongarch64::CodeGenerator &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
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
    EmitIR<IR::Opcode::VectorGetElement8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitGetElement<8>(code, ctx, inst,
                          [&](auto &Wresult, auto &Qvalue, u8 index) { code.UMOV(Wresult, Qvalue->Belem()[index]); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorGetElement16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitGetElement<16>(code, ctx, inst,
                           [&](auto &Wresult, auto &Qvalue, u8 index) { code.UMOV(Wresult, Qvalue->Helem()[index]); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorGetElement32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitGetElement<32>(code, ctx, inst,
                           [&](auto &Wresult, auto &Qvalue, u8 index) { code.UMOV(Wresult, Qvalue->Selem()[index]); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorGetElement64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitGetElement<64>(code, ctx, inst,
                           [&](auto &Xresult, auto &Qvalue, u8 index) { code.UMOV(Xresult, Qvalue->Delem()[index]); });
    }

    template<size_t size, typename EmitFn>
    static void EmitSetElement(Xbyak_loongarch64::CodeGenerator &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
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
    EmitIR<IR::Opcode::VectorSetElement8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitSetElement<8>(
                code, ctx, inst,
                [&](auto &Qvector, auto &Wvalue, u8 index) { code.add_d(Qvector->Belem()[index], Wvalue); }, code.zero);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSetElement16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitSetElement<16>(
                code, ctx, inst,
                [&](auto &Qvector, auto &Wvalue, u8 index) { code.add_d(Qvector->Helem()[index], Wvalue); }, code.zero);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSetElement32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitSetElement<32>(
                code, ctx, inst,
                [&](auto &Qvector, auto &Wvalue, u8 index) { code.add_d(Qvector->Selem()[index], Wvalue); }, code.zero);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSetElement64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitSetElement<64>(
                code, ctx, inst,
                [&](auto &Qvector, auto &Xvalue, u8 index) { code.add_d(Qvector->Delem()[index], Xvalue); }, code.zero);
    }

    template<>
    void EmitIR<IR::Opcode::VectorAbs8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.ABS(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAbs16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.ABS(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAbs32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.ABS(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAbs64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.ABS(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAdd8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.ADD(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAdd16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.ADD(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAdd32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.ADD(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAdd64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.ADD(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAnd>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.andi(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorAndNot>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.BIC(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorArithmeticShiftRight8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitImmShift<8>(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.SSHR(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorArithmeticShiftRight16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitImmShift<16>(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.SSHR(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorArithmeticShiftRight32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitImmShift<32>(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.SSHR(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorArithmeticShiftRight64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitImmShift<64>(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.SSHR(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorArithmeticVShift8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SSHL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorArithmeticVShift16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SSHL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorArithmeticVShift32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SSHL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorArithmeticVShift64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SSHL(Vresult, Va, Vb); });
    }

    template<size_t size, typename EmitFn>
    static void EmitBroadcast(Xbyak_loongarch64::CodeGenerator &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qvector = ctx.reg_alloc.WriteQ(inst);
        auto Rvalue = ctx.reg_alloc.ReadReg<std::max<size_t>(32, size)>(args[0]);
        RegAlloc::Realize(Qvector, Rvalue);

        // TODO: fpr source

        emit(Qvector, Rvalue);
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastLower8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                   IR::Inst *inst) {
        EmitBroadcast<8>(code, ctx, inst, [&](auto &Qvector, auto &Wvalue) { code.DUP(Qvector->toD().B8(), Wvalue); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastLower16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitBroadcast<16>(code, ctx, inst, [&](auto &Qvector, auto &Wvalue) { code.DUP(Qvector->toD().H4(), Wvalue); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastLower32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitBroadcast<32>(code, ctx, inst, [&](auto &Qvector, auto &Wvalue) { code.DUP(Qvector->toD().S2(), Wvalue); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorBroadcast8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitBroadcast<8>(code, ctx, inst, [&](auto &Qvector, auto &Wvalue) { code.DUP(Qvector->B16(), Wvalue); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorBroadcast16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitBroadcast<16>(code, ctx, inst, [&](auto &Qvector, auto &Wvalue) { code.DUP(Qvector->H8(), Wvalue); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorBroadcast32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitBroadcast<32>(code, ctx, inst, [&](auto &Qvector, auto &Wvalue) { code.DUP(Qvector->S4(), Wvalue); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorBroadcast64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitBroadcast<64>(code, ctx, inst, [&](auto &Qvector, auto &Xvalue) { code.DUP(Qvector->D2(), Xvalue); });
    }

    template<size_t size, typename EmitFn>
    static void
    EmitBroadcastElement(Xbyak_loongarch64::CodeGenerator &, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qvector = ctx.reg_alloc.WriteQ(inst);
        auto Qvalue = ctx.reg_alloc.ReadQ(args[0]);
        const u8 index = args[1].GetImmediateU8();
        RegAlloc::Realize(Qvector, Qvalue);

        emit(Qvector, Qvalue, index);
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastElementLower8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitBroadcastElement<8>(code, ctx, inst, [&](auto &Qvector, auto &Qvalue, u8 index) {
            code.DUP(Qvector->toD().B8(), Qvalue->Belem()[index]);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastElementLower16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitBroadcastElement<16>(code, ctx, inst, [&](auto &Qvector, auto &Qvalue, u8 index) {
            code.DUP(Qvector->toD().H4(), Qvalue->Helem()[index]);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastElementLower32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitBroadcastElement<32>(code, ctx, inst, [&](auto &Qvector, auto &Qvalue, u8 index) {
            code.DUP(Qvector->toD().S2(), Qvalue->Selem()[index]);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastElement8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitBroadcastElement<8>(code, ctx, inst, [&](auto &Qvector, auto &Qvalue, u8 index) {
            code.DUP(Qvector->B16(), Qvalue->Belem()[index]);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastElement16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitBroadcastElement<16>(code, ctx, inst, [&](auto &Qvector, auto &Qvalue, u8 index) {
            code.DUP(Qvector->H8(), Qvalue->Helem()[index]);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastElement32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitBroadcastElement<32>(code, ctx, inst, [&](auto &Qvector, auto &Qvalue, u8 index) {
            code.DUP(Qvector->S4(), Qvalue->Selem()[index]);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorBroadcastElement64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitBroadcastElement<64>(code, ctx, inst, [&](auto &Qvector, auto &Qvalue, u8 index) {
            code.DUP(Qvector->D2(), Qvalue->Delem()[index]);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorCountLeadingZeros8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitTwoOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.CLZ(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorCountLeadingZeros16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitTwoOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.CLZ(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorCountLeadingZeros32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitTwoOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.CLZ(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveEven8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UZP1(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveEven16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UZP1(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveEven32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UZP1(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveEven64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UZP1(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveEvenLower8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitThreeOpArrangedLower<8>(code, ctx, inst,
                                    [&](auto Vresult, auto Va, auto Vb) { code.UZP1(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveEvenLower16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitThreeOpArrangedLower<16>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.UZP1(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveEvenLower32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitThreeOpArrangedLower<32>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.UZP1(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveOdd8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UZP2(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveOdd16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UZP2(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveOdd32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UZP2(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveOdd64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UZP2(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveOddLower8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitThreeOpArrangedLower<8>(code, ctx, inst,
                                    [&](auto Vresult, auto Va, auto Vb) { code.UZP2(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveOddLower16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitThreeOpArrangedLower<16>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.UZP2(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorDeinterleaveOddLower32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitThreeOpArrangedLower<32>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.UZP2(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorEor>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.EOR(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorEqual8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.CMEQ(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorEqual16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.CMEQ(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorEqual32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.CMEQ(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorEqual64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.CMEQ(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorEqual128>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::VectorExtract>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        auto Qa = ctx.reg_alloc.ReadQ(args[0]);
        auto Qb = ctx.reg_alloc.ReadQ(args[1]);
        const u8 position = args[2].GetImmediateU8();
        ASSERT(position % 8 == 0);
        RegAlloc::Realize(Qresult, Qa, Qb);

        code.EXT(Qresult->B16(), Qa->B16(), Qb->B16(), position / 8);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorExtractLower>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Dresult = ctx.reg_alloc.WriteD(inst);
        auto Da = ctx.reg_alloc.ReadD(args[0]);
        auto Db = ctx.reg_alloc.ReadD(args[1]);
        const u8 position = args[2].GetImmediateU8();
        ASSERT(position % 8 == 0);
        RegAlloc::Realize(Dresult, Da, Db);

        code.EXT(Dresult->B8(), Da->B8(), Db->B8(), position / 8);
    }

    template<>
    void EmitIR<IR::Opcode::VectorGreaterS8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.CMGT(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorGreaterS16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.CMGT(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorGreaterS32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.CMGT(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorGreaterS64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.CMGT(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingAddS8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SHADD(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingAddS16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SHADD(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingAddS32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SHADD(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingAddU8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UHADD(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingAddU16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UHADD(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingAddU32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UHADD(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingSubS8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SHSUB(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingSubS16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SHSUB(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingSubS32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SHSUB(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingSubU8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UHSUB(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingSubU16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UHSUB(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorHalvingSubU32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UHSUB(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorInterleaveLower8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.ZIP1(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorInterleaveLower16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.ZIP1(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorInterleaveLower32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.ZIP1(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorInterleaveLower64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.ZIP1(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorInterleaveUpper8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.ZIP2(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorInterleaveUpper16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.ZIP2(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorInterleaveUpper32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.ZIP2(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorInterleaveUpper64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.ZIP2(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalShiftLeft8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitImmShift<8>(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.SHL(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalShiftLeft16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitImmShift<16>(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.SHL(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalShiftLeft32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitImmShift<32>(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.SHL(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalShiftLeft64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitImmShift<64>(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.SHL(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalShiftRight8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitImmShift<8>(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.USHR(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalShiftRight16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitImmShift<16>(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.USHR(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalShiftRight32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitImmShift<32>(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.USHR(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalShiftRight64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitImmShift<64>(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            code.USHR(Vresult, Voperand, shift_amount);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorLogicalVShift8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.USHL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalVShift16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                   IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.USHL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalVShift32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                   IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.USHL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorLogicalVShift64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                   IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.USHL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMaxS8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SMAX(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMaxS16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SMAX(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMaxS32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SMAX(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMaxS64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::VectorMaxU8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UMAX(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMaxU16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UMAX(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMaxU32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UMAX(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMaxU64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::VectorMinS8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SMIN(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMinS16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SMIN(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMinS32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SMIN(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMinS64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::VectorMinU8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UMIN(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMinU16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UMIN(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMinU32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UMIN(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMinU64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::VectorMultiply8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.MUL(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorMultiply16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.MUL(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorMultiply32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.MUL(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorMultiply64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        ASSERT_MSG(ctx.conf.very_verbose_debugging_output, "VectorMultiply64 is for debugging only");
        EmitThreeOp(code, ctx, inst, [&](auto &Qresult, auto &Qa, auto &Qb) {
            code.FMOV(Xscratch0, Qa->toD());
            code.FMOV(Xscratch1, Qb->toD());
            code.mul_d(Xscratch0, Xscratch0, Xscratch1);
            code.FMOV(Qresult->toD(), Xscratch0);
            code.FMOV(Xscratch0, Qa->Delem()[1]);
            code.FMOV(Xscratch1, Qb->Delem()[1]);
            code.mul_d(Xscratch0, Xscratch0, Xscratch1);
            code.FMOV(Qresult->Delem()[1], Xscratch0);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMultiplySignedWiden8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitThreeOpArrangedWiden<8>(code, ctx, inst,
                                    [&](auto Vresult, auto Va, auto Vb) { code.SMULL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMultiplySignedWiden16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitThreeOpArrangedWiden<16>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.SMULL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMultiplySignedWiden32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitThreeOpArrangedWiden<32>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.SMULL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMultiplyUnsignedWiden8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitThreeOpArrangedWiden<8>(code, ctx, inst,
                                    [&](auto Vresult, auto Va, auto Vb) { code.UMULL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMultiplyUnsignedWiden16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitThreeOpArrangedWiden<16>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.UMULL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorMultiplyUnsignedWiden32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitThreeOpArrangedWiden<32>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.UMULL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorNarrow16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArrangedNarrow<16>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.XTN(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorNarrow32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArrangedNarrow<32>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.XTN(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorNarrow64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArrangedNarrow<64>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.XTN(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorNot>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.NOT(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorOr>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.ORR(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddLower8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                   IR::Inst *inst) {
        EmitThreeOpArrangedLower<8>(code, ctx, inst,
                                    [&](auto Vresult, auto Va, auto Vb) { code.ADDP(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddLower16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOpArrangedLower<16>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.ADDP(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddLower32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOpArrangedLower<32>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.ADDP(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddSignedWiden8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitTwoOpArrangedPairWiden<8>(code, ctx, inst,
                                      [&](auto Vresult, auto Voperand) { code.SADDLP(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddSignedWiden16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitTwoOpArrangedPairWiden<16>(code, ctx, inst,
                                       [&](auto Vresult, auto Voperand) { code.SADDLP(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddSignedWiden32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitTwoOpArrangedPairWiden<32>(code, ctx, inst,
                                       [&](auto Vresult, auto Voperand) { code.SADDLP(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddUnsignedWiden8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitTwoOpArrangedPairWiden<8>(code, ctx, inst,
                                      [&](auto Vresult, auto Voperand) { code.UADDLP(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddUnsignedWiden16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                            IR::Inst *inst) {
        EmitTwoOpArrangedPairWiden<16>(code, ctx, inst,
                                       [&](auto Vresult, auto Voperand) { code.UADDLP(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedAddUnsignedWiden32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                            IR::Inst *inst) {
        EmitTwoOpArrangedPairWiden<32>(code, ctx, inst,
                                       [&](auto Vresult, auto Voperand) { code.UADDLP(Vresult, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedAdd8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.ADDP(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedAdd16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.ADDP(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedAdd32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.ADDP(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedAdd64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.ADDP(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMaxS8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SMAXP(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMaxS16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SMAXP(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMaxS32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SMAXP(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMaxU8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UMAXP(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMaxU16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UMAXP(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMaxU32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UMAXP(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMinS8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SMINP(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMinS16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SMINP(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMinS32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SMINP(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMinU8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UMINP(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMinU16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UMINP(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorPairedMinU32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UMINP(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMaxLowerS8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOpArrangedLower<8>(code, ctx, inst,
                                    [&](auto Vresult, auto Va, auto Vb) { code.SMAXP(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMaxLowerS16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArrangedLower<16>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.SMAXP(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMaxLowerS32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArrangedLower<32>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.SMAXP(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMaxLowerU8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOpArrangedLower<8>(code, ctx, inst,
                                    [&](auto Vresult, auto Va, auto Vb) { code.UMAXP(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMaxLowerU16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArrangedLower<16>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.UMAXP(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMaxLowerU32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArrangedLower<32>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.UMAXP(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMinLowerS8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOpArrangedLower<8>(code, ctx, inst,
                                    [&](auto Vresult, auto Va, auto Vb) { code.SMINP(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMinLowerS16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArrangedLower<16>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.SMINP(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMinLowerS32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArrangedLower<32>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.SMINP(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMinLowerU8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOpArrangedLower<8>(code, ctx, inst,
                                    [&](auto Vresult, auto Va, auto Vb) { code.UMINP(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMinLowerU16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArrangedLower<16>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.UMINP(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPairedMinLowerU32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitThreeOpArrangedLower<32>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.UMINP(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPolynomialMultiply8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.PMUL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPolynomialMultiplyLong8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitThreeOpArrangedWiden<8>(code, ctx, inst,
                                    [&](auto Vresult, auto Va, auto Vb) { code.PMULL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPolynomialMultiplyLong64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                            IR::Inst *inst) {
        EmitThreeOpArrangedWiden<64>(code, ctx, inst,
                                     [&](auto Vresult, auto Va, auto Vb) { code.PMULL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorPopulationCount>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                   IR::Inst *inst) {
        EmitTwoOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.vpcnt_b(Vresult, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReverseBits>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Voperand) {
            code.addi_d(Xscratch0, code.zero, 0xFF);
            code.vreplgr2vr_b(code.vr0, Xscratch0);
            code.vbitrevi_b(Vresult, Voperand, code.vr0);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReverseElementsInHalfGroups8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitTwoOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.REV16(Vresult, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReverseElementsInWordGroups8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitTwoOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.REV32(Vresult, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReverseElementsInWordGroups16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                            IR::Inst *inst) {
        EmitTwoOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.REV32(Vresult, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReverseElementsInLongGroups8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitTwoOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.REV64(Vresult, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReverseElementsInLongGroups16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                            IR::Inst *inst) {
        EmitTwoOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.REV64(Vresult, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReverseElementsInLongGroups32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                            IR::Inst *inst) {
        EmitTwoOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.REV64(Vresult, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReduceAdd8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitReduce<8>(code, ctx, inst,
                      [&](auto &Bresult, auto Voperand) { code.vhaddw_h_b(Bresult, Voperand, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReduceAdd16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitReduce<16>(code, ctx, inst,
                       [&](auto &Hresult, auto Voperand) { code.vhaddw_w_h(Hresult, Voperand, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReduceAdd32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitReduce<32>(code, ctx, inst,
                       [&](auto &Sresult, auto Voperand) { code.vhaddw_d_w(Sresult, Voperand, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorReduceAdd64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitReduce<64>(code, ctx, inst,
                       [&](auto &Dresult, auto Voperand) { code.vhaddw_q_d(Dresult, Voperand, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRotateWholeVectorRight>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                          IR::Inst *inst) {
        EmitImmShift<8>(code, ctx, inst, [&](auto Vresult, auto Voperand, u8 shift_amount) {
            ASSERT(shift_amount % 8 == 0);
            const u8 ext_imm = (shift_amount % 128) / 8;
            code.vrotri_b(Vresult, Voperand, ext_imm);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingHalvingAddS8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vavgr_b(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingHalvingAddS16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vavgr_h(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingHalvingAddS32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vavgr_w(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingHalvingAddU8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst,
                               [&](auto Vresult, auto Va, auto Vb) { code.vavgr_bu(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingHalvingAddU16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vavgr_hu(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingHalvingAddU32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vavgr_wu(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingShiftLeftS8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SRSHL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingShiftLeftS16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SRSHL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingShiftLeftS32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SRSHL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingShiftLeftS64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.SRSHL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingShiftLeftU8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.URSHL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingShiftLeftU16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.URSHL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingShiftLeftU32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.URSHL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorRoundingShiftLeftU64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitThreeOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.URSHL(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignExtend8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArrangedWiden<8>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.SXTL(Vresult, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignExtend16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArrangedWiden<16>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.SXTL(Vresult, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignExtend32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArrangedWiden<32>(code, ctx, inst, [&](auto Vresult, auto Voperand) { code.SXTL(Vresult, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignExtend64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedAbsoluteDifference8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                             IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vabsd_b(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedAbsoluteDifference16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                              IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vabsd_h(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedAbsoluteDifference32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                              IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vabsd_w(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedMultiply16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedMultiply32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAbs8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitTwoOpArrangedSaturated<8>(code, ctx, inst,
                                      [&](auto Vresult, auto Voperand) { code.SQABS(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAbs16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitTwoOpArrangedSaturated<16>(code, ctx, inst,
                                       [&](auto Vresult, auto Voperand) { code.SQABS(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAbs32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitTwoOpArrangedSaturated<32>(code, ctx, inst,
                                       [&](auto Vresult, auto Voperand) { code.SQABS(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAbs64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitTwoOpArrangedSaturated<64>(code, ctx, inst,
                                       [&](auto Vresult, auto Voperand) { code.SQABS(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAccumulateUnsigned8>(Xbyak_loongarch64::CodeGenerator &code,
                                                                      EmitContext &ctx, IR::Inst *inst) {
        EmitSaturatedAccumulate<8>(code, ctx, inst,
                                   [&](auto Vaccumulator, auto Voperand) { code.SUQADD(Vaccumulator, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAccumulateUnsigned16>(Xbyak_loongarch64::CodeGenerator &code,
                                                                       EmitContext &ctx, IR::Inst *inst) {
        EmitSaturatedAccumulate<16>(code, ctx, inst,
                                    [&](auto Vaccumulator, auto Voperand) { code.SUQADD(Vaccumulator, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAccumulateUnsigned32>(Xbyak_loongarch64::CodeGenerator &code,
                                                                       EmitContext &ctx, IR::Inst *inst) {
        EmitSaturatedAccumulate<32>(code, ctx, inst,
                                    [&](auto Vaccumulator, auto Voperand) { code.SUQADD(Vaccumulator, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedAccumulateUnsigned64>(Xbyak_loongarch64::CodeGenerator &code,
                                                                       EmitContext &ctx, IR::Inst *inst) {
        EmitSaturatedAccumulate<64>(code, ctx, inst,
                                    [&](auto Vaccumulator, auto Voperand) { code.SUQADD(Vaccumulator, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedDoublingMultiplyHigh16>(Xbyak_loongarch64::CodeGenerator &code,
                                                                         EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArrangedSaturated<16>(code, ctx, inst,
                                         [&](auto Vresult, auto Va, auto Vb) { code.SQDMULH(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedDoublingMultiplyHigh32>(Xbyak_loongarch64::CodeGenerator &code,
                                                                         EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArrangedSaturated<32>(code, ctx, inst,
                                         [&](auto Vresult, auto Va, auto Vb) { code.SQDMULH(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedDoublingMultiplyHighRounding16>(Xbyak_loongarch64::CodeGenerator &code,
                                                                                 EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArrangedSaturated<16>(code, ctx, inst,
                                         [&](auto Vresult, auto Va, auto Vb) { code.SQRDMULH(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedDoublingMultiplyHighRounding32>(Xbyak_loongarch64::CodeGenerator &code,
                                                                                 EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArrangedSaturated<32>(code, ctx, inst,
                                         [&](auto Vresult, auto Va, auto Vb) { code.SQRDMULH(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedDoublingMultiplyLong16>(Xbyak_loongarch64::CodeGenerator &code,
                                                                         EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArrangedSaturatedWiden<16>(code, ctx, inst,
                                              [&](auto Vresult, auto Va, auto Vb) { code.SQDMULL(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedDoublingMultiplyLong32>(Xbyak_loongarch64::CodeGenerator &code,
                                                                         EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArrangedSaturatedWiden<32>(code, ctx, inst,
                                              [&](auto Vresult, auto Va, auto Vb) { code.SQDMULL(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignedSaturatedNarrowToSigned16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                              IR::Inst *inst) {
        EmitTwoOpArrangedSaturatedNarrow<16>(code, ctx, inst,
                                             [&](auto Vresult, auto Voperand) { code.SQXTN(Vresult, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignedSaturatedNarrowToSigned32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                              IR::Inst *inst) {
        EmitTwoOpArrangedSaturatedNarrow<32>(code, ctx, inst,
                                             [&](auto Vresult, auto Voperand) { code.SQXTN(Vresult, Voperand); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorSignedSaturatedNarrowToSigned64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                              IR::Inst *inst) {
        EmitTwoOpArrangedSaturatedNarrow<64>(code, ctx, inst,
                                             [&](auto Vresult, auto Voperand) { code.SQXTN(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedNarrowToUnsigned16>(Xbyak_loongarch64::CodeGenerator &code,
                                                                     EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArrangedSaturatedNarrow<16>(code, ctx, inst,
                                             [&](auto Vresult, auto Voperand) { code.SQXTUN(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedNarrowToUnsigned32>(Xbyak_loongarch64::CodeGenerator &code,
                                                                     EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArrangedSaturatedNarrow<32>(code, ctx, inst,
                                             [&](auto Vresult, auto Voperand) { code.SQXTUN(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedNarrowToUnsigned64>(Xbyak_loongarch64::CodeGenerator &code,
                                                                     EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArrangedSaturatedNarrow<64>(code, ctx, inst,
                                             [&](auto Vresult, auto Voperand) { code.SQXTUN(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedNeg8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitTwoOpArrangedSaturated<8>(code, ctx, inst,
                                      [&](auto Vresult, auto Voperand) { code.SQNEG(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedNeg16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitTwoOpArrangedSaturated<16>(code, ctx, inst,
                                       [&](auto Vresult, auto Voperand) { code.SQNEG(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedNeg32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitTwoOpArrangedSaturated<32>(code, ctx, inst,
                                       [&](auto Vresult, auto Voperand) { code.SQNEG(Vresult, Voperand); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedNeg64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                        IR::Inst *inst) {
        EmitTwoOpArrangedSaturated<64>(code, ctx, inst,
                                       [&](auto Vresult, auto Voperand) { code.SQNEG(Vresult, Voperand); });
    }

    template<typename Lambda>
    static void
    EmitTwoArgumentFallbackWithSaturation(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst,
                                          Lambda lambda) {
        const auto fn = static_cast<mcl::equivalent_function_type<Lambda> *>(lambda);
        constexpr u32 stack_space = 3 * 16;
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        const Xbyak::Xmm arg1 = ctx.reg_alloc.UseXmm(args[0]);
        const Xbyak::Xmm arg2 = ctx.reg_alloc.UseXmm(args[1]);
        const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
        ctx.reg_alloc.EndOfAllocScope();

        ctx.reg_alloc.HostCall(nullptr);
        ctx.reg_alloc.AllocStackSpace(stack_space + ABI_SHADOW_SPACE);
        code.lea(code.ABI_PARAM1, ptr[rsp + ABI_SHADOW_SPACE + 0 * 16]);
        code.lea(code.ABI_PARAM2, ptr[rsp + ABI_SHADOW_SPACE + 1 * 16]);
        code.lea(code.ABI_PARAM3, ptr[rsp + ABI_SHADOW_SPACE + 2 * 16]);

        code.movaps(xword[code.ABI_PARAM2], arg1);
        code.movaps(xword[code.ABI_PARAM3], arg2);
        code.CallFunction(fn);
        code.movaps(result, xword[rsp + ABI_SHADOW_SPACE + 0 * 16]);

        ctx.reg_alloc.ReleaseStackSpace(stack_space + ABI_SHADOW_SPACE);

        code.or_(code.byte[code.r15 + code.GetJitStateInfo().offsetof_fpsr_qc], code.ABI_RETURN.cvt8());

        ctx.reg_alloc.DefineValue(inst, result);
    }

    template<typename T, typename U = std::make_unsigned_t<T>>
    static bool VectorSignedSaturatedShiftLeft(VectorArray <T> &dst, const VectorArray <T> &data,
                                               const VectorArray <T> &shift_values) {
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
    void EmitIR<IR::Opcode::VectorSignedSaturatedShiftLeft8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                             IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, VectorSignedSaturatedShiftLeft<s8>);
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedShiftLeft16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                              IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, VectorSignedSaturatedShiftLeft<s16>);
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedShiftLeft32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                              IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, VectorSignedSaturatedShiftLeft<s32>);
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedShiftLeft64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                              IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, VectorSignedSaturatedShiftLeft<s64>);
    }

    template<typename T, typename U = std::make_unsigned_t<T>>
    static bool
    VectorSignedSaturatedShiftLeftUnsigned(VectorArray <T> &dst, const VectorArray <T> &data, u8 shift_amount) {
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

    template<typename Lambda>
    static void
    EmitTwoArgumentFallbackWithSaturationAndImmediate(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst, Lambda lambda) {
        const auto fn = static_cast<mcl::equivalent_function_type<Lambda> *>(lambda);
        constexpr u32 stack_space = 2 * 16;
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        const Xbyak::Xmm arg1 = ctx.reg_alloc.UseXmm(args[0]);
        const u8 arg2 = args[1].GetImmediateU8();
        const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
        ctx.reg_alloc.EndOfAllocScope();

        ctx.reg_alloc.HostCall(nullptr);
        ctx.reg_alloc.AllocStackSpace(stack_space + ABI_SHADOW_SPACE);
        code.lea(code.ABI_PARAM1, ptr[rsp + ABI_SHADOW_SPACE + 0 * 16]);
        code.lea(code.ABI_PARAM2, ptr[rsp + ABI_SHADOW_SPACE + 1 * 16]);

        code.movaps(xword[code.ABI_PARAM2], arg1);
        code.mov(code.ABI_PARAM3, arg2);
        code.CallFunction(fn);
        code.movaps(result, xword[rsp + ABI_SHADOW_SPACE + 0 * 16]);

        ctx.reg_alloc.ReleaseStackSpace(stack_space + ABI_SHADOW_SPACE);

        code.or_(code.byte[code.r15 + code.GetJitStateInfo().offsetof_fpsr_qc], code.ABI_RETURN.cvt8());

        ctx.reg_alloc.DefineValue(inst, result);
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedShiftLeftUnsigned8>(Xbyak_loongarch64::CodeGenerator &code,
                                                                     EmitContext &ctx, IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturationAndImmediate(code, ctx, inst, VectorSignedSaturatedShiftLeftUnsigned<s8>);
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedShiftLeftUnsigned16>(Xbyak_loongarch64::CodeGenerator &code,
                                                                      EmitContext &ctx, IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturationAndImmediate(code, ctx, inst, VectorSignedSaturatedShiftLeftUnsigned<s16>);
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedShiftLeftUnsigned32>(Xbyak_loongarch64::CodeGenerator &code,
                                                                      EmitContext &ctx, IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturationAndImmediate(code, ctx, inst, VectorSignedSaturatedShiftLeftUnsigned<s32>);
    }

    template<>
    void EmitIR<IR::Opcode::VectorSignedSaturatedShiftLeftUnsigned64>(Xbyak_loongarch64::CodeGenerator &code,
                                                                      EmitContext &ctx, IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturationAndImmediate(code, ctx, inst, VectorSignedSaturatedShiftLeftUnsigned<s64>);
    }

    template<>
    void EmitIR<IR::Opcode::VectorSub8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<8>(
                code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsub_b(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSub16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<16>(
                code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsub_h(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSub32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<32>(
                code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsub_w(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorSub64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpArranged<64>(
                code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vsub_d(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::VectorTable>(Xbyak_loongarch64::CodeGenerator &, EmitContext &, IR::Inst *inst) {
        // Do nothing. We *want* to hold on to the refcount for our arguments, so VectorTableLookup can use our arguments.
        ASSERT_MSG(inst->UseCount() == 1, "Table cannot be used multiple times");
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorTableLookup64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        ASSERT(inst->GetArg(1).GetInst()->GetOpcode() == IR::Opcode::VectorTable);

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto table = ctx.reg_alloc.GetArgumentInfo(inst->GetArg(1).GetInst());

        const size_t table_size = std::count_if(table.begin(), table.end(),
                                                [](const auto &elem) { return !elem.IsVoid(); });
        const bool is_defaults_zero = inst->GetArg(0).IsZero();

        auto Dresult = is_defaults_zero ? ctx.reg_alloc.WriteD(inst) : ctx.reg_alloc.ReadWriteD(args[0], inst);
        auto Dindices = ctx.reg_alloc.ReadD(args[2]);
        std::vector<RAReg<Xbyak_loongarch64::DReg>> Dtable;
        for (size_t i = 0; i < table_size; i++) {
            Dtable.emplace_back(ctx.reg_alloc.ReadD(table[i]));
        }
        RegAlloc::Realize(Dresult, Dindices);
        for (size_t i = 0; i < table_size; i++) {
            RegAlloc::Realize(Dtable[i]);
        }

        switch (table_size) {
            case 1:
                code.MOVI(V2.B16(), 0x08);
                code.CMGE(V2.B8(), Dindices->B8(), V2.B8());
                code.ORR(V2.B8(), Dindices->B8(), V2.B8());
                code.FMOV(D0, Dtable[0]);
                if (is_defaults_zero) {
                    code.TBL(Dresult->B8(), Xbyak_loongarch64::List{V0.B16()}, D2.B8());
                } else {
                    code.TBX(Dresult->B8(), Xbyak_loongarch64::List{V0.B16()}, D2.B8());
                }
                break;
            case 2:
                code.ZIP1(V0.D2(), Dtable[0]->toQ().D2(), Dtable[1]->toQ().D2());
                if (is_defaults_zero) {
                    code.TBL(Dresult->B8(), Xbyak_loongarch64::List{V0.B16()}, Dindices->B8());
                } else {
                    code.TBX(Dresult->B8(), Xbyak_loongarch64::List{V0.B16()}, Dindices->B8());
                }
                break;
            case 3:
                code.MOVI(V2.B16(), 0x18);
                code.CMGE(V2.B8(), Dindices->B8(), V2.B8());
                code.ORR(V2.B8(), Dindices->B8(), V2.B8());
                code.ZIP1(V0.D2(), Dtable[0]->toQ().D2(), Dtable[1]->toQ().D2());
                code.FMOV(D1, Dtable[2]);
                if (is_defaults_zero) {
                    code.TBL(Dresult->B8(), Xbyak_loongarch64::List{V0.B16(), V1.B16()}, D2.B8());
                } else {
                    code.TBX(Dresult->B8(), Xbyak_loongarch64::List{V0.B16(), V1.B16()}, D2.B8());
                }
                break;
            case 4:
                code.ZIP1(V0.D2(), Dtable[0]->toQ().D2(), Dtable[1]->toQ().D2());
                code.ZIP1(V1.D2(), Dtable[2]->toQ().D2(), Dtable[3]->toQ().D2());
                if (is_defaults_zero) {
                    code.TBL(Dresult->B8(), Xbyak_loongarch64::List{V0.B16(), V1.B16()}, Dindices->B8());
                } else {
                    code.TBX(Dresult->B8(), Xbyak_loongarch64::List{V0.B16(), V1.B16()}, Dindices->B8());
                }
                break;
            default:
                ASSERT_FALSE("Unsupported table_size");
        }
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorTableLookup128>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        ASSERT(inst->GetArg(1).GetInst()->GetOpcode() == IR::Opcode::VectorTable);

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto table = ctx.reg_alloc.GetArgumentInfo(inst->GetArg(1).GetInst());

        const size_t table_size = std::count_if(table.begin(), table.end(),
                                                [](const auto &elem) { return !elem.IsVoid(); });
        const bool is_defaults_zero = inst->GetArg(0).IsZero();

        auto result = is_defaults_zero ? ctx.reg_alloc.WriteQ(inst) : ctx.reg_alloc.ReadWriteQ(args[0], inst);
        auto indicies = ctx.reg_alloc.ReadQ(args[2]);
        std::vector<RAReg<Xbyak_loongarch64::VReg>> Qtable;
        for (size_t i = 0; i < table_size; i++) {
            Qtable.emplace_back(ctx.reg_alloc.ReadQ(table[i]));
        }
        RegAlloc::Realize(result, indicies);
        for (size_t i = 0; i < table_size; i++) {
            RegAlloc::Realize(Qtable[i]);
        }

        if (table_size == 1) {
//        const Xbyak::Xmm indicies = ctx.reg_alloc.UseXmm(args[2]);
            const Xbyak::Xmm defaults = result;//ctx.reg_alloc.UseXmm(args[0]);
            const Xbyak::Xmm xmm_table0 = Qtable[0];

            code.addi_d(Xscratch0, code.zero, 0x70);
            code.vreplgr2vr_b(xmm1, Xscratch0);
            code.vsadd_b(xmm0, xmm1, indicies);

            code.vshuf_b(xmm_table0, xmm_table0, indicies);
            code.vbitsel_v(xmm_table0, xmm0, xmm_table0, defaults);
//        code.pshufb(xmm_table0, indicies);
//        code.pblendvb(xmm_table0, defaults);

//        ctx.reg_alloc.DefineValue(inst, xmm_table0);
            return;
        }

        if (is_defaults_zero && table_size == 2) {
//        const Xbyak::Xmm indicies = ctx.reg_alloc.UseScratchXmm(args[2]);
            const Xbyak::Xmm xmm_table0 = Qtable[0];
            const Xbyak::Xmm xmm_table1 = Qtable[1];


//            code.movaps(xmm0, indicies);
//            code.paddusb(xmm0, code.MConst(xword, 0x7070707070707070, 0x7070707070707070));
            code.addi_d(Xscratch0, code.zero, 0x70);
            code.vreplgr2vr_b(xmm1, Xscratch0);
            code.vsadd_b(xmm0, xmm1, indicies);

            code.addi_d(Xscratch0, code.zero, 0x60);
            code.vreplgr2vr_b(xmm1, Xscratch0);
            code.vsadd_b(indicies, xmm1, indicies);

//        code.paddusb(indicies, code.MConst(xword, 0x6060606060606060, 0x6060606060606060));
//        code.pshufb(xmm_table0, xmm0);
            code.vshuf_b(xmm_table0, xmm_table0, xmm0);
            code.vshuf_b(xmm_table1, xmm_table1, indicies);
//        code.pshufb(xmm_table1, indicies);
            code.vbitsel_v(xmm_table0, xmm0, xmm_table0, xmm_table1);
//        code.pblendvb(xmm_table0, xmm_table1);

//        ctx.reg_alloc.DefineValue(inst, xmm_table0);
            return;
        }
//    const Xbyak::Xmm indicies = ctx.reg_alloc.UseXmm(args[2]);
//    const Xbyak::Xmm result = ctx.reg_alloc.UseScratchXmm(args[0]);
        const Xbyak::Xmm masked = ctx.reg_alloc.ScratchXmm();

        code.addi_d(Xscratch0, code.zero, 0xF0);
        code.vreplgr2vr_b(masked, Xscratch0);
        code.vand_v(masked, masked, indicies);
//    code.movaps(masked, code.MConst(xword, 0xF0F0F0F0F0F0F0F0, 0xF0F0F0F0F0F0F0F0));
//    code.pand(masked, indicies);

        for (size_t i = 0; i < table_size; ++i) {
            const auto xmm_table = Qtable[i];

            const u64 table_index = mcl::bit::replicate_element<u8, u64>(i * 16);

            if (table_index == 0) {
                code.vxor_v(code.vr0, code.vr0, code.vr0);
                code.vseq_b(code.vr0, masked, code.vr0);
            } else {
                code.add_imm(Xscratch0, code.zero, i * 16, Xscratch1);
                code.vreplgr2vr_b(masked, Xscratch0);
                code.vseq_b(code.vr0, masked, code.vr0);
//            code.movaps(xmm0, code.MConst(xword, table_index, table_index));
//            code.pcmpeqb(xmm0, masked);
            }
            code.vshuf_b(xmm_table, xmm_table, indicies);
            code.vbitsel_v(result, code.vr0, result, xmm_table);
//        code.pblendvb(result, xmm_table);

//        ctx.reg_alloc.Release(xmm_table);
        }

//    ctx.reg_alloc.DefineValue(inst, result);
        return;

    }

    template<>
    void
    EmitIR<IR::Opcode::VectorTranspose8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        const bool part = inst->GetArg(2).GetU1();
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            !part ? code.vpackev_b(Vresult, Vb, Va) : code.vpackod_b(Vresult, Vb, Va);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorTranspose16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        const bool part = inst->GetArg(2).GetU1();
        EmitThreeOpArranged<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            !part ? code.vpackev_h(Vresult, Vb, Va) : code.vpackod_h(Vresult, Vb, Va);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorTranspose32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        const bool part = inst->GetArg(2).GetU1();
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            !part ? code.vpackev_w(Vresult, Vb, Va) : code.vpackod_w(Vresult, Vb, Va);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorTranspose64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        const bool part = inst->GetArg(2).GetU1();
        EmitThreeOpArranged<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            !part ? code.vpackev_d(Vresult, Vb, Va) : code.vpackod_d(Vresult, Vb, Va);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedAbsoluteDifference8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                               IR::Inst *inst) {
        EmitThreeOpArranged<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.vabsd_b(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorUnsignedAbsoluteDifference16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitThreeOpArranged<16>(code, ctx, inst,
                                [&](auto Vresult, auto Va, auto Vb) { code.vabsd_h(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorUnsignedAbsoluteDifference32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitThreeOpArranged<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) {
            code.vabsd_w(Vresult, Va, Vb);
        });
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedMultiply16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedMultiply32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
    }

    template<typename Lambda>
    static void
    EmitOneArgumentFallback(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst, Lambda lambda) {
        const auto fn = static_cast<mcl::equivalent_function_type<Lambda> *>(lambda);
        constexpr u32 stack_space = 2 * 16;
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        const Xbyak::Xmm arg1 = ctx.reg_alloc.UseXmm(args[0]);
        const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
        ctx.reg_alloc.EndOfAllocScope();

        ctx.reg_alloc.HostCall(nullptr);
        ctx.reg_alloc.AllocStackSpace(stack_space + ABI_SHADOW_SPACE);
        code.lea(code.ABI_PARAM1, ptr[rsp + ABI_SHADOW_SPACE + 0 * 16]);
        code.lea(code.ABI_PARAM2, ptr[rsp + ABI_SHADOW_SPACE + 1 * 16]);

        code.movaps(xword[code.ABI_PARAM2], arg1);
        code.CallFunction(fn);
        code.movaps(result, xword[rsp + ABI_SHADOW_SPACE + 0 * 16]);

        ctx.reg_alloc.ReleaseStackSpace(stack_space + ABI_SHADOW_SPACE);

        ctx.reg_alloc.DefineValue(inst, result);
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedRecipEstimate>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        EmitOneArgumentFallback(code, ctx, inst, [](VectorArray <u32> &result, const VectorArray <u32> &a) {
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
    void EmitIR<IR::Opcode::VectorUnsignedRecipSqrtEstimate>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                             IR::Inst *inst) {
        EmitOneArgumentFallback(code, ctx, inst, [](VectorArray <u32> &result, const VectorArray <u32> &a) {
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
    static bool EmitVectorUnsignedSaturatedAccumulateSigned(VectorArray <U> &result, const VectorArray <T> &lhs,
                                                            const VectorArray <T> &rhs) {
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
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedAccumulateSigned8>(Xbyak_loongarch64::CodeGenerator &code,
                                                                      EmitContext &ctx, IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, EmitVectorUnsignedSaturatedAccumulateSigned<s8>);
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedAccumulateSigned16>(Xbyak_loongarch64::CodeGenerator &code,
                                                                       EmitContext &ctx, IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, EmitVectorUnsignedSaturatedAccumulateSigned<s16>);
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedAccumulateSigned32>(Xbyak_loongarch64::CodeGenerator &code,
                                                                       EmitContext &ctx, IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, EmitVectorUnsignedSaturatedAccumulateSigned<s32>);
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedAccumulateSigned64>(Xbyak_loongarch64::CodeGenerator &code,
                                                                       EmitContext &ctx, IR::Inst *inst) {
        // TODO use vsadd?
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst,
                                              [](VectorArray <u64> &result, const VectorArray <u64> &lhs,
                                                 const VectorArray <u64> &rhs) {
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

    template<typename Lambda>
    static void
    EmitOneArgumentFallbackWithSaturation(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst,
                                          Lambda lambda) {
        const auto fn = static_cast<mcl::equivalent_function_type<Lambda> *>(lambda);
        constexpr u32 stack_space = 2 * 16;
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        const Xbyak::Xmm arg1 = ctx.reg_alloc.UseXmm(args[0]);
        const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
        ctx.reg_alloc.EndOfAllocScope();

        ctx.reg_alloc.HostCall(nullptr);
        ctx.reg_alloc.AllocStackSpace(stack_space + ABI_SHADOW_SPACE);
        code.lea(code.ABI_PARAM1, ptr[rsp + ABI_SHADOW_SPACE + 0 * 16]);
        code.lea(code.ABI_PARAM2, ptr[rsp + ABI_SHADOW_SPACE + 1 * 16]);

        code.movaps(xword[code.ABI_PARAM2], arg1);
        code.CallFunction(fn);
        code.movaps(result, xword[rsp + ABI_SHADOW_SPACE + 0 * 16]);

        ctx.reg_alloc.ReleaseStackSpace(stack_space + ABI_SHADOW_SPACE);

        code.or_(code.byte[code.r15 + code.GetJitStateInfo().offsetof_fpsr_qc], code.ABI_RETURN.cvt8());

        ctx.reg_alloc.DefineValue(inst, result);
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedNarrow16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                             IR::Inst *inst) {
        EmitOneArgumentFallbackWithSaturation(code, ctx, inst,
                                              [](VectorArray <u8> &result, const VectorArray <u16> &a) {
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
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedNarrow32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                             IR::Inst *inst) {
        EmitOneArgumentFallbackWithSaturation(code, ctx, inst,
                                              [](VectorArray <u16> &result, const VectorArray <u32> &a) {
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
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedNarrow64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                             IR::Inst *inst) {
        EmitOneArgumentFallbackWithSaturation(code, ctx, inst,
                                              [](VectorArray <u32> &result, const VectorArray <u64> &a) {
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

    using A64FullVectorWidth = std::integral_constant<size_t, 128>;
// Array alias that always sizes itself according to the given type T
// relative to the size of a vector register. e.g. T = u32 would result
// in a std::array<u32, 4>.
    template<typename T>
    using VectorArray = std::array<T, A64FullVectorWidth::value / mcl::bitsizeof<T>>;

    template<typename T, typename S = std::make_signed_t<T>>
    static bool VectorUnsignedSaturatedShiftLeft(VectorArray<T> &dst, const VectorArray<T> &data,
                                                 const VectorArray<T> &shift_values) {
        static_assert(std::is_unsigned_v<T>, "T must be an unsigned type.");

        bool qc_flag = false;

        constexpr size_t bit_size = mcl::bitsizeof<T>;
        constexpr S negative_bit_size = -static_cast<S>(bit_size);

        for (size_t i = 0; i < dst.size(); i++) {
            const T element = data[i];
            const S shift = std::clamp(static_cast<S>(mcl::bit::sign_extend<8>(static_cast<T>(shift_values[i] & 0xFF))),
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

    template<typename Lambda>
    static void
    EmitTwoArgumentFallbackWithSaturation(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst,
                                          Lambda lambda) {
        const auto fn = static_cast<mcl::equivalent_function_type<Lambda> *>(lambda);
        constexpr u32 stack_space = 3 * 16;
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        const Xbyak::Xmm arg1 = ctx.reg_alloc.UseXmm(args[0]);
        const Xbyak::Xmm arg2 = ctx.reg_alloc.UseXmm(args[1]);
        const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
        ctx.reg_alloc.EndOfAllocScope();

        ctx.reg_alloc.HostCall(nullptr);
        ctx.reg_alloc.AllocStackSpace(stack_space + ABI_SHADOW_SPACE);
        code.lea(code.ABI_PARAM1, ptr[rsp + ABI_SHADOW_SPACE + 0 * 16]);
        code.lea(code.ABI_PARAM2, ptr[rsp + ABI_SHADOW_SPACE + 1 * 16]);
        code.lea(code.ABI_PARAM3, ptr[rsp + ABI_SHADOW_SPACE + 2 * 16]);

        code.movaps(xword[code.ABI_PARAM2], arg1);
        code.movaps(xword[code.ABI_PARAM3], arg2);
        code.CallFunction(fn);
        code.movaps(result, xword[rsp + ABI_SHADOW_SPACE + 0 * 16]);

        ctx.reg_alloc.ReleaseStackSpace(stack_space + ABI_SHADOW_SPACE);

        code.or_(code.byte[code.r15 + code.GetJitStateInfo().offsetof_fpsr_qc], code.ABI_RETURN.cvt8());

        ctx.reg_alloc.DefineValue(inst, result);
    }

    template<>
    void EmitIR<IR::Opcode::VectorUnsignedSaturatedShiftLeft8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                               IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, VectorUnsignedSaturatedShiftLeft<u8>);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorUnsignedSaturatedShiftLeft16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, VectorUnsignedSaturatedShiftLeft<u16>);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorUnsignedSaturatedShiftLeft32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, VectorUnsignedSaturatedShiftLeft<u32>);
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorUnsignedSaturatedShiftLeft64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx,
                                                           IR::Inst *inst) {
//    EmitThreeOpArrangedSaturated<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.UQSHL(Vresult, Va, Vb); });
        EmitTwoArgumentFallbackWithSaturation(code, ctx, inst, VectorUnsignedSaturatedShiftLeft<u64>);

    }

    template<>
    void
    EmitIR<IR::Opcode::VectorZeroExtend8>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArrangedWiden<8>(code, ctx, inst, [&](auto Vresult, auto Voperand) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vilvl_b(Vresult, Vresult, Voperand);
//        code.UXTL(Vresult, Voperand);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorZeroExtend16>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArrangedWiden<16>(code, ctx, inst, [&](auto Vresult, auto Voperand) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vilvl_h(Vresult, Vresult, Voperand);
//        code.UXTL(Vresult, Voperand);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorZeroExtend32>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOpArrangedWiden<32>(code, ctx, inst, [&](auto Vresult, auto Voperand) {
            code.vxor_v(Vresult, Vresult, Vresult);
            code.vilvl_w(Vresult, Vresult, Voperand);
            // FIXME use which?
//        code.vsllwil_hu_bu(Vresult, Voperand, 0);?
//        code.UXTL(Vresult, Voperand);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::VectorZeroExtend64>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Qresult, auto &Qoperand) {
            code.vor_v(Qresult, Qoperand, Qoperand);
//        code.FMOV(Qresult->toD(), Qoperand->toD()); });
        }

        template<>
        void
        EmitIR<IR::Opcode::VectorZeroUpper>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
            EmitTwoOp(code, ctx, inst, [&](auto &Qresult, auto &Qoperand) { code.vsubi_du(Qresult, Qoperand, 0); });
        }

        template<>
        void EmitIR<IR::Opcode::ZeroVector>(Xbyak_loongarch64::CodeGenerator &code, EmitContext &ctx, IR::Inst *inst) {
            auto Qresult = ctx.reg_alloc.WriteQ(inst);
            RegAlloc::Realize(Qresult);
            code.vreplgr2vr_d(Qresult, code.zero);
        }

    }  // namespace Dynarmic::Backend::LoongArch64
