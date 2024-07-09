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
#include "dynarmic/common/fp/util.h"
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
            code.NAME##w(args...);   \
        } else {                     \
            code.NAME##d(args...);   \
        }                            \
    }

    enum class CheckInputNaN {
        Yes,
        No,
    };

    using A64FullVectorWidth = std::integral_constant<size_t, 128>;

// Array alias that always sizes itself according to the given type T
// relative to the size of a vector register. e.g. T = u32 would result
// in a std::array<u32, 4>.
    template<typename T>
    using VectorArray = std::array<T, A64FullVectorWidth::value / mcl::bitsizeof<T>>;

    template<size_t fsize>
    Xbyak_loongarch64::VReg GetVectorOf(BlockOfCode& code, u64 value) {
        if constexpr (fsize == 16) {
            code.add_imm(Xscratch0, code.zero, (value << 48) | (value << 32) | (value << 16) | value, Xscratch1);
            code.vinsgr2vr_d(Vscratch2, Xscratch0, 0);
            code.vinsgr2vr_d(Vscratch2, Xscratch0, 1);
        } else if constexpr (fsize == 32) {
            code.add_imm(Xscratch0, code.zero, (value << 32) | value, Xscratch1);
            code.vinsgr2vr_d(Vscratch2, Xscratch0, 0);
            code.vinsgr2vr_d(Vscratch2, Xscratch0, 1);
        } else {
            static_assert(fsize == 64);
            code.add_imm(Xscratch0, code.zero, value, Xscratch1);
            code.vinsgr2vr_d(Vscratch2, Xscratch0, 0);
            code.vinsgr2vr_d(Vscratch2, Xscratch0, 1);
        }
        return Vscratch2;
    }

    template<size_t fsize, u64 value>
    Xbyak_loongarch64::VReg GetVectorOf(BlockOfCode& code) {
        if constexpr (fsize == 16) {
            code.add_imm(Xscratch0, code.zero, (value << 48) | (value << 32) | (value << 16) | value, Xscratch1);
            code.vinsgr2vr_d(Vscratch2, Xscratch0, 0);
            code.vinsgr2vr_d(Vscratch2, Xscratch0, 1);
        } else if constexpr (fsize == 32) {
            code.add_imm(Xscratch0, code.zero, (value << 32) | value, Xscratch1);
            code.vinsgr2vr_d(Vscratch2, Xscratch0, 0);
            code.vinsgr2vr_d(Vscratch2, Xscratch0, 1);
        } else {
            static_assert(fsize == 64);
            code.add_imm(Xscratch0, code.zero, value, Xscratch1);
            code.vinsgr2vr_d(Vscratch2, Xscratch0, 0);
            code.vinsgr2vr_d(Vscratch2, Xscratch0, 1);
        }
        return Vscratch2;
    }

    template<size_t fsize>
    Xbyak_loongarch64::VReg GetNaNVector(BlockOfCode& code) {
        using FPT = mcl::unsigned_integer_of_size<fsize>;
        return GetVectorOf<fsize, FP::FPInfo<FPT>::DefaultNaN()>(code);
    }

    template<size_t fsize>
    Xbyak_loongarch64::VReg GetNegativeZeroVector(BlockOfCode& code) {
        using FPT = mcl::unsigned_integer_of_size<fsize>;
        return GetVectorOf<fsize, FP::FPInfo<FPT>::Zero(true)>(code);
    }

    template<size_t fsize>
    Xbyak_loongarch64::VReg GetNonSignMaskVector(BlockOfCode& code) {
        using FPT = mcl::unsigned_integer_of_size<fsize>;
        constexpr FPT non_sign_mask = FP::FPInfo<FPT>::exponent_mask | FP::FPInfo<FPT>::mantissa_mask;
        return GetVectorOf<fsize, non_sign_mask>(code);
    }

    template<size_t fsize>
    Xbyak_loongarch64::VReg GetSmallestNormalVector(BlockOfCode& code) {
        using FPT = mcl::unsigned_integer_of_size<fsize>;
        constexpr FPT smallest_normal_number = FP::FPValue<FPT, false, FP::FPInfo<FPT>::exponent_min, 1>();
        return GetVectorOf<fsize, smallest_normal_number>(code);
    }

    template<size_t fsize, bool sign, int exponent, mcl::unsigned_integer_of_size<fsize> value>
    Xbyak_loongarch64::VReg GetVectorOf(BlockOfCode& code) {
        using FPT = mcl::unsigned_integer_of_size<fsize>;
        return GetVectorOf<fsize, FP::FPValue<FPT, sign, exponent, value>()>(code);
    }

    template<size_t fsize, size_t nargs, typename NaNHandler>
    void HandleNaNs(BlockOfCode& code, EmitContext& ctx, bool fpcr_controlled,
                    std::array<Xbyak_loongarch64::VReg, nargs + 1> xmms,
                    const Xbyak_loongarch64::VReg& nan_mask, NaNHandler nan_handler) {
        static_assert(fsize == 32 || fsize == 64, "fsize must be either 32 or 64");

        code.vand_v(Vscratch0, nan_mask, nan_mask);
        code.vsetnez_v(0, Vscratch0);

        SharedLabel end = GenSharedLabel(), nan = GenSharedLabel();

        code.bcnez(0, *nan);
        code.L(*end);

        ctx.deferred_emits.emplace_back([=, &code, &ctx] {
            code.L(*nan);

            const auto result = xmms[0];
            const size_t stack_space = xmms.size() * 16;

            ABI_PushRegisters(code, ABI_CALLEE_SAVE & ~ToRegList(result), sizeof(StackLayout) + stack_space);

            for (size_t i = 0; i < xmms.size(); ++i) {
                code.vst(xmms[i], code.sp, i* 16);
            }
            code.addi_d(code.a0, code.sp, 0 * 16);
            code.add_imm(code.a1, code.zero, ctx.FPCR(fpcr_controlled).Value(), Xscratch0);

            code.CallFunction(nan_handler);

            code.vld(result, code.sp, 0 * 16);

            ABI_PopRegisters(code, ABI_CALLEE_SAVE & ~ToRegList(result), sizeof(StackLayout) + stack_space);
            code.b(*end);
        });
    }

    template<size_t fsize>
    void ForceToDefaultNaN(BlockOfCode& code, FP::FPCR fpcr, Xbyak_loongarch64::VReg result) {
        if (fpcr.DN()) {
//            if (code.HasHostFeature(HostFeature::AVX)) {
                auto nan_mask = Vscratch0;
                FCODE(vfcmp_cun_)(nan_mask, result, result);
                // TODO check this want qNaN or sNaN
                code.vbitsel_v(result, result, GetNaNVector<fsize>(code), nan_mask);
        }
    }

    template<size_t fsize, template<typename> class Indexer, size_t narg>
    struct NaNHandler {
    public:
        using FPT = mcl::unsigned_integer_of_size<fsize>;

        using function_type = void (*)(std::array<VectorArray<FPT>, narg>&, FP::FPCR);

        static function_type GetDefault() {
            return GetDefaultImpl(std::make_index_sequence<narg - 1>{});
        }

    private:
        template<size_t... argi>
        static function_type GetDefaultImpl(std::index_sequence<argi...>) {
            const auto result = [](std::array<VectorArray<FPT>, narg>& values, FP::FPCR) {
                VectorArray<FPT>& result = values[0];
                for (size_t elementi = 0; elementi < result.size(); ++elementi) {
                    const auto current_values = Indexer<FPT>{}(elementi, values[argi + 1]...);
                    if (auto r = FP::ProcessNaNs(std::get<argi>(current_values)...)) {
                        result[elementi] = *r;
                    } else if (FP::IsNaN(result[elementi])) {
                        result[elementi] = FP::FPInfo<FPT>::DefaultNaN();
                    }
                }
            };

            return static_cast<function_type>(result);
        }
    };

    template<typename T>
    struct DefaultIndexer {
        std::tuple<T> operator()(size_t i, const VectorArray<T>& a) {
            return std::make_tuple(a[i]);
        }

        std::tuple<T, T> operator()(size_t i, const VectorArray<T>& a, const VectorArray<T>& b) {
            return std::make_tuple(a[i], b[i]);
        }

        std::tuple<T, T, T> operator()(size_t i, const VectorArray<T>& a, const VectorArray<T>& b, const VectorArray<T>& c) {
            return std::make_tuple(a[i], b[i], c[i]);
        }
    };
    static constexpr u32 subnormal_mask = 0x110;

    template<size_t fsize>
    void DenormalsAreZero(BlockOfCode& code, FP::FPCR fpcr, std::initializer_list<Xbyak_loongarch64::VReg> to_daz, Xbyak_loongarch64::VReg tmp) {

        if (fpcr.FZ()) {
            code.vxor_v(tmp, tmp, tmp);

            if (fpcr.RMode() != FP::RoundingMode::TowardsMinusInfinity) {
                code.vor_v(tmp, tmp, GetNegativeZeroVector<fsize>(code));
            }
            for (const Xbyak_loongarch64::VReg& xmm : to_daz) {
                Xbyak_loongarch64::Label not_normals;
                FCODE(vfclass_)(Vscratch1, xmm);
                code.vand_v(Vscratch1, Vscratch1, GetVectorOf<fsize, subnormal_mask>(code));
                code.vseteqz_v(0, Vscratch1);
                code.bcnez(0, not_normals);
                code.vxor_v(xmm, tmp, tmp);
                code.L(not_normals);
            }
        }
    }

    template<typename EmitFn>
    static void MaybeStandardFPSCRValue(BlockOfCode &code, EmitContext &ctx, bool fpcr_controlled,
                                        EmitFn emit) {

        const bool switch_mxcsr = ctx.FPCR(fpcr_controlled) != ctx.FPCR();
        // TODO Unsafe_IgnoreStandardFPCRValue
        if (switch_mxcsr && !ctx.conf.HasOptimization(OptimizationFlag::Unsafe_IgnoreStandardFPCRValue)) {
            code.EnterStandardASIMD();
            emit();
            code.LeaveStandardASIMD();
        } else {
            emit();
        }
    }

    template<size_t fsize, template<typename> class Indexer, typename Function>
    void EmitThreeOpVectorOperation(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst, Function fn,
                                    CheckInputNaN check_input_nan = CheckInputNaN::No,
                                    typename NaNHandler<fsize, Indexer, 3>::function_type nan_handler = NaNHandler<fsize, Indexer, 3>::GetDefault()) {
        static_assert(fsize == 32 || fsize == 64, "fsize must be either 32 or 64");

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        const bool fpcr_controlled = args[2].GetImmediateU1();

        if (ctx.FPCR(fpcr_controlled).DN() || ctx.conf.HasOptimization(OptimizationFlag::Unsafe_InaccurateNaN)) {
            auto xmm_a = ctx.reg_alloc.ReadQ(args[0]);
            auto xmm_b = ctx.reg_alloc.ReadQ(args[1]);
            auto result = ctx.reg_alloc.WriteQ(inst);
            RegAlloc::Realize(xmm_a, xmm_b, result);

            if constexpr (std::is_member_function_pointer_v<Function>) {
                MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] {
                    (code.*fn)(xmm_a, xmm_b);
                });
            } else {
                MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] {
                    fn(xmm_a, xmm_b);
                });
            }

            if (!ctx.conf.HasOptimization(OptimizationFlag::Unsafe_InaccurateNaN)) {
                ForceToDefaultNaN<fsize>(code, ctx.FPCR(fpcr_controlled), xmm_a);
            }
            code.vxor_v(result, result, result);
            code.vor_v(result, result, xmm_a);
            return;
        }
        auto result = ctx.reg_alloc.WriteQ(inst);
        auto xmm_a = ctx.reg_alloc.ReadQ(args[0]);
        auto xmm_b = ctx.reg_alloc.ReadQ(args[1]);
        RegAlloc::Realize(result, xmm_b, xmm_a);

        auto nan_mask = Vscratch0;

//        code.vxor_v(result, result, result);
        code.vor_v(result, xmm_a, xmm_a);

        if (check_input_nan == CheckInputNaN::Yes) {
            FCODE(vfcmp_cun_)(nan_mask, *xmm_a, *xmm_b);
        }

        if constexpr (std::is_member_function_pointer_v<Function>) {
            (code.*fn)(result, xmm_b);
        } else {
            fn(result, xmm_b);
        }

        if (check_input_nan == CheckInputNaN::Yes) {
            FCODE(vfcmp_cun_)(nan_mask, nan_mask, *result);
        } else {
            FCODE(vfcmp_cun_)(nan_mask, *result, *result);
        }

        HandleNaNs<fsize, 2>(code, ctx, fpcr_controlled, {result, xmm_a, xmm_b}, nan_mask, nan_handler);
    }


    template<size_t fsize, bool is_max>
    static void EmitFPVectorMinMax(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
        const bool fpcr_controlled = inst->GetArg(2).GetU1();

        if (ctx.FPCR(fpcr_controlled).DN()) {
            auto args = ctx.reg_alloc.GetArgumentInfo(inst);
            auto result = ctx.reg_alloc.WriteQ(inst);
            auto xmm_a = ctx.reg_alloc.ReadQ(args[0]);
            auto xmm_b = ctx.reg_alloc.ReadQ(args[1]);
            RegAlloc::Realize(result, xmm_a, xmm_b);

            // TODO is this safe? how to get extra scratch reg?
            auto mask = code.vr31;
            auto eq = code.vr30;
            auto nan_mask = code.vr29;

            MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] {
                DenormalsAreZero<fsize>(code, ctx.FPCR(fpcr_controlled), {xmm_a, xmm_b}, mask);

//                if (code.HasHostFeature(HostFeature::AVX)) {
                    FCODE(vfcmp_ceq_)(mask, *xmm_a, *xmm_b);
                    FCODE(vfcmp_cun_)(nan_mask, *xmm_a, *xmm_b);
                    if constexpr (is_max) {
                        code.vand_v(eq, xmm_a, xmm_b);
                        FCODE(vfmax_)(*result, *xmm_a, *xmm_b);
                    } else {
                        code.vor_v(eq, xmm_a, xmm_b);
                        FCODE(vfmin_)(*result, *xmm_a, *xmm_b);
                    }
                    code.vbitsel_v(result, result, eq, mask);
                    code.vbitsel_v(result, result, GetNaNVector<fsize>(code), nan_mask);

            });

            return;
        }

        EmitThreeOpVectorOperation<fsize, DefaultIndexer>(
                code, ctx, inst, [&](Xbyak_loongarch64::VReg result, Xbyak_loongarch64::VReg xmm_b) {
                    Xbyak_loongarch64::VReg mask = code.vr31;
                    Xbyak_loongarch64::VReg eq = code.vr30;

                    if (ctx.FPCR(fpcr_controlled).FZ()) {
                        const Xbyak_loongarch64::VReg prev_xmm_b = xmm_b;
                        xmm_b = code.vr29;
                        code.vxor_v(xmm_b, xmm_b, xmm_b);
                        code.vor_v(xmm_b, xmm_b , prev_xmm_b);
                        DenormalsAreZero<fsize>(code, ctx.FPCR(fpcr_controlled), {result, xmm_b}, mask);
                    }

                    // What we are doing here is handling the case when the inputs are differently signed zeros.
                    // x86-64 treats differently signed zeros as equal while ARM does not.
                    // Thus if we AND together things that x86-64 thinks are equal we'll get the positive zero.

                    // vrangep{s,d} here ends up not being significantly shorter than the AVX implementation

//                    if (code.HasHostFeature(HostFeature::AVX)) {
                        FCODE(vfcmp_ceq_)(mask, result, xmm_b);
                        if constexpr (is_max) {
                            code.vand_v(eq, result, xmm_b);
                            FCODE(vfmax_)(result, result, xmm_b);
                        } else {
                            code.vor_v(eq, result, xmm_b);
                            FCODE(vfmin_)(result, result, xmm_b);
                        }
                        code.vbitsel_v(result, result, eq, mask);

                },
                CheckInputNaN::Yes);
    }

    template<size_t fsize, bool is_max>
    static void EmitFPVectorMinMaxNumeric(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
        const bool fpcr_controlled = inst->GetArg(2).GetU1();

        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto xmm_a = ctx.reg_alloc.ReadQ(args[0]);
        auto xmm_b = ctx.reg_alloc.ReadQ(args[1]);
        auto result = ctx.reg_alloc.WriteQ(inst);

        RegAlloc::Realize(xmm_a, xmm_b, result);
        auto intermediate_result = code.vr31;
        auto tmp1 = code.vr30;
        auto tmp2 = code.vr29;

//        const Xbyak::Xmm tmp1 = xmm0;

        // NaN requirements:
        // op1     op2      result
        // SNaN    anything op1
        // !SNaN   SNaN     op2
        // QNaN    !NaN     op2
        // !NaN    QNaN     op1
        // QNaN    QNaN     op1

//        if (code.HasHostFeature(HostFeature::AVX)) {
            MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] {
                using FPT = mcl::unsigned_integer_of_size<fsize>;

                // result = xmm_a == SNaN || xmm_b == QNaN
                {
                    // evaluate xmm_b == QNaN
                    FCODE(vfcmp_cun_)(tmp1, *xmm_b, *xmm_b);
                    ICODE(vslli_)(tmp2, *xmm_b, static_cast<u8>(fsize - FP::FPInfo<FPT>::explicit_mantissa_width));
                    {
                        code.vsrai_w(tmp2, tmp2, 31);
                        if constexpr (fsize == 64) {
                            code.vshuf4i_w(tmp2, tmp2, 0b11110101);
                        }
                    }
                    code.vand_v(result, tmp1, tmp2);

                    // evaluate xmm_a == SNaN
                    FCODE(vfcmp_cun_)(tmp1, *xmm_a, *xmm_a);
                    //TODO fix all occur explicit_mantissa_width
                    ICODE(vslli_)(tmp2, *xmm_a, static_cast<u8>(fsize - FP::FPInfo<FPT>::explicit_mantissa_width));
                    {
                        // upper could be true even if use vfcmp_sun_ if the operate is qNan
                        // so here just test if is a sNaN
                        code.vbitrevi_w(tmp2, tmp2, 31);
                        code.vsrai_w(tmp2, tmp2, 31);
                        if constexpr (fsize == 64) {
                            code.vshuf4i_w(tmp2, tmp2, 0b11110101);
                        }
                    }
                    code.vand_v(tmp2, tmp2, tmp1);

                    code.vor_v(result, result, tmp2);
                }

                // Denormalization quiets SNaNs, therefore should happen after SNaN detection!
                DenormalsAreZero<fsize>(code, ctx.FPCR(fpcr_controlled), {xmm_a, xmm_b}, tmp1);

                // intermediate result = max/min(xmm_a, xmm_b)
                {
                    const auto eq_mask = tmp1;
                    const auto eq = tmp2;

                    FCODE(vfcmp_ceq_)(eq_mask, *xmm_a, *xmm_b);

                    if constexpr (is_max) {
                        code.vand_v(eq, xmm_a, xmm_b);
                        // FIXME don't know if it is a document err or hardware err:
                        // this seems to don't work like 3.1.1.3指令产生的非数结果. if there is  a qNaN, the
                        // result is other reguler number. if there is a sNaN, the resule is a qNaN.
                        // thus we xor mantissa_msb to
                        FCODE(vfmax_)(intermediate_result, *xmm_a, *xmm_b);
                    } else {
                        code.vor_v(eq, xmm_a, xmm_b);
                        FCODE(vfmin_)(intermediate_result, *xmm_a, *xmm_b);
                    }
                    code.vbitsel_v(intermediate_result, intermediate_result, eq, eq_mask);
                }

                {
                    code.vbitsel_v(result, intermediate_result, xmm_a, result);
                }

                if (ctx.FPCR(fpcr_controlled).DN()) {
                    const auto ord_mask = tmp1;

                    FCODE(vfcmp_cun_)(ord_mask, *result, *result);
                    code.vbitsel_v(result, result, GetNaNVector<fsize>(code), ord_mask);

                } else {
                    const auto nan_mask = tmp1;

                    FCODE(vfcmp_cun_)(nan_mask, *result, *result);
                    code.vand_v(nan_mask, nan_mask, GetVectorOf<fsize, FP::FPInfo<FPT>::mantissa_msb>(code));
                    code.vor_v(result, result, nan_mask);
                }
            });
            //        }
    }

    template<typename EmitFn>
    static void EmitTwoOp(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        auto Qa = ctx.reg_alloc.ReadQ(args[0]);
        const bool fpcr_controlled = args[1].IsVoid() || args[1].GetImmediateU1();
        RegAlloc::Realize(Qresult, Qa);
        ctx.fpsr.Load();

        MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] { emit(Qresult, Qa); });
    }


    template<typename EmitFn>
    static void EmitThreeOp(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        auto Qa = ctx.reg_alloc.ReadQ(args[0]);
        auto Qb = ctx.reg_alloc.ReadQ(args[1]);
        const bool fpcr_controlled = args[2].GetImmediateU1();
        RegAlloc::Realize(Qresult, Qa, Qb);
        ctx.fpsr.Load();

        MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] { emit(Qresult, Qa, Qb); });
    }

    template<typename Lambda>
    void EmitThreeOpFallback(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst, Lambda lambda) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        auto Qa = ctx.reg_alloc.ReadQ(args[0]);
        auto Qb = ctx.reg_alloc.ReadQ(args[1]);
        const bool fpcr_controlled = args[2].GetImmediateU1();
        RegAlloc::Realize(Qresult, Qa, Qb);

        EmitThreeOpFallbackWithoutRegAlloc(code, ctx, Qresult, Qa, Qb, lambda, fpcr_controlled);

    }


    template<size_t size, typename EmitFn>
    static void EmitFMA(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qresult = ctx.reg_alloc.ReadWriteQ(args[0], inst);
        auto Qm = ctx.reg_alloc.ReadQ(args[1]);
        auto Qn = ctx.reg_alloc.ReadQ(args[2]);
        const bool fpcr_controlled = args[3].GetImmediateU1();
        RegAlloc::Realize(Qresult, Qm, Qn);
        ctx.fpsr.Load();

        MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] {

            emit(Qresult, Qm, Qn);
        });
    }

    template<size_t size, typename EmitFn>
    static void EmitFromFixed(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst, EmitFn emit) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qto = ctx.reg_alloc.WriteQ(inst);
        auto Qfrom = ctx.reg_alloc.ReadQ(args[0]);
        const u8 fbits = args[1].GetImmediateU8();
        const FP::RoundingMode rounding_mode = static_cast<FP::RoundingMode>(args[2].GetImmediateU8());
        const bool fpcr_controlled = args[3].GetImmediateU1();
        ASSERT(rounding_mode == ctx.FPCR(fpcr_controlled).RMode());
        RegAlloc::Realize(Qto, Qfrom);

        MaybeStandardFPSCRValue(code, ctx, fpcr_controlled, [&] {
                emit(Qto, Qfrom, fbits);
        });
    }


    template<typename Lambda>
    static void EmitTwoOpFallbackWithoutRegAlloc(BlockOfCode &code, EmitContext &ctx,
                                                 Xbyak_loongarch64::VReg Qresult, Xbyak_loongarch64::VReg Qarg1,
                                                 Lambda lambda, bool fpcr_controlled) {
        const auto fn = static_cast<mcl::equivalent_function_type<Lambda> *>(lambda);

        const u32 fpcr = ctx.FPCR(fpcr_controlled).Value();
        constexpr u64 stack_size = sizeof(u64) * 4;

        ABI_PushRegisters(code, ABI_CALLER_SAVE & ~ToRegList(Qresult), stack_size);

        code.add_imm(code.a0, code.sp, 0 * 16, Xscratch1);
        code.add_imm(code.a1, code.sp, 1 * 16, Xscratch1);
        code.add_imm(code.a2, code.zero, fpcr, Xscratch1);
        code.add_imm(code.a3, Xstate, ctx.conf.state_fpsr_offset, Xscratch1);
        code.vst(Qarg1, code.a1, 0);

        code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(fn), Xscratch1);
        code.jirl(code.ra, Xscratch0, 0);
        code.vld(Qresult, code.sp, 0);

        ABI_PopRegisters(code, ABI_CALLER_SAVE & ~ToRegList(Qresult), stack_size);
    }

    template<size_t fpcr_controlled_arg_index = 1, typename Lambda>
    static void
    EmitTwoOpFallback(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst, Lambda lambda) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qarg1 = ctx.reg_alloc.ReadQ(args[0]);
        auto Qresult = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(Qarg1, Qresult);
        ctx.reg_alloc.SpillFlags();
        ctx.fpsr.Spill();

        const bool fpcr_controlled = args[fpcr_controlled_arg_index].GetImmediateU1();
        EmitTwoOpFallbackWithoutRegAlloc(code, ctx, Qresult, Qarg1, lambda, fpcr_controlled);
    }

    template<typename Lambda>
    void EmitThreeOpFallbackWithoutRegAlloc(BlockOfCode& code, EmitContext& ctx, Xbyak_loongarch64::VReg result,
                                            Xbyak_loongarch64::VReg arg1, Xbyak_loongarch64::VReg arg2,
                                            Lambda lambda, bool fpcr_controlled) {
        const auto fn = static_cast<mcl::equivalent_function_type<Lambda>*>(lambda);

        const u32 fpcr = ctx.FPCR(fpcr_controlled).Value();

        constexpr u64 stack_size = sizeof(u64) * 4;

        ABI_PushRegisters(code, ABI_CALLER_SAVE & ~ToRegList(result), stack_size);

        code.add_imm(code.a0, code.sp, 0 * 16, Xscratch1);
        code.add_imm(code.a1, code.sp, 1 * 16, Xscratch1);
        code.add_imm(code.a2, code.sp, 2 * 16, Xscratch1);
        code.add_imm(code.a3, code.zero, fpcr, Xscratch1);
        code.add_imm(code.a4, Xstate, ctx.conf.state_fpsr_offset, Xscratch1);
        code.vst(arg1, code.a1, 0);
        code.vst(arg2, code.a2, 0);

        code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(fn), Xscratch1);
        code.jirl(code.ra, Xscratch0, 0);
        code.vld(result, code.sp, 0);

        ABI_PopRegisters(code, ABI_CALLER_SAVE & ~ToRegList(result), stack_size);

    }
    enum class LoadPreviousResult {
        Yes,
        No,
    };
    
    template<LoadPreviousResult load_previous_result = LoadPreviousResult::No, typename Lambda>
    void EmitFourOpFallbackWithoutRegAlloc(BlockOfCode& code, EmitContext& ctx, Xbyak_loongarch64::VReg result, 
                                           Xbyak_loongarch64::VReg arg1, Xbyak_loongarch64::VReg arg2, Xbyak_loongarch64::VReg arg3,
                                           Lambda lambda, bool fpcr_controlled) {
        const auto fn = static_cast<mcl::equivalent_function_type<Lambda>*>(lambda);


        constexpr u32 stack_size = 4 * 16;

        ABI_PushRegisters(code, ABI_CALLER_SAVE & ~ToRegList(result), stack_size);

        code.add_imm(code.a0, code.sp, 0 * 16, Xscratch1);
        code.add_imm(code.a1, code.sp, 1 * 16, Xscratch1);
        code.add_imm(code.a2, code.sp, 2 * 16, Xscratch1);
        code.add_imm(code.a3, code.sp, 3 * 16, Xscratch1);
        code.add_imm(code.a4, code.zero, ctx.FPCR(fpcr_controlled).Value(), Xscratch1);
        code.add_imm(code.a5, Xstate, ctx.conf.state_fpsr_offset, Xscratch1);
        
        if constexpr (load_previous_result == LoadPreviousResult::Yes) {
            code.vst(result, code.a0, 0);
        }
        code.vst(arg1, code.a1, 0);
        code.vst(arg2, code.a2, 0);
        code.vst(arg3, code.a3, 0);

        code.add_imm(Xscratch0, code.zero, mcl::bit_cast<u64>(fn), Xscratch1);
        code.jirl(code.ra, Xscratch0, 0);
        code.vld(result, code.sp, 0);

        ABI_PopRegisters(code, ABI_CALLER_SAVE & ~ToRegList(result), stack_size);
        
    }

    template<typename Lambda>
    void EmitFourOpFallback(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst, Lambda lambda) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        const bool fpcr_controlled = args[3].GetImmediateU1();
        auto arg1 = ctx.reg_alloc.ReadQ(args[0]);
        auto arg2 = ctx.reg_alloc.ReadQ(args[1]);
        auto arg3 = ctx.reg_alloc.ReadQ(args[2]);
        auto result = ctx.reg_alloc.WriteQ(inst);
        RegAlloc::Realize(arg1, arg2, arg3, result);

        EmitFourOpFallbackWithoutRegAlloc(code, ctx, result, arg1, arg2, arg3, lambda, fpcr_controlled);

    }
    
    template<size_t fsize>
    void FPVectorAbs(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto result = ctx.reg_alloc.WriteQ(inst);
        auto a = ctx.reg_alloc.ReadQ(args[0]);
        RegAlloc::Realize(a, result);
        code.vand_v(result, a, GetNonSignMaskVector<fsize>(code));
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorAbs16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        FPVectorAbs<16>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorAbs32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        FPVectorAbs<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorAbs64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        FPVectorAbs<64>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorAdd32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto &Vresult, auto &Va, auto &Vb) { code.vfadd_s(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorAdd64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto &Vresult, auto &Va, auto &Vb) { code.vfadd_d(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorDiv32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto &Vresult, auto &Va, auto &Vb) { code.vfdiv_s(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorDiv64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto &Vresult, auto &Va, auto &Vb) { code.vfdiv_d(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorEqual16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOpFallback(code, ctx, inst, [](VectorArray<u16>& result, const VectorArray<u16>& op1, const VectorArray<u16>& op2, FP::FPCR fpcr, FP::FPSR& fpsr) {
            for (size_t i = 0; i < result.size(); i++) {
                result[i] = FP::FPCompareEQ(op1[i], op2[i], fpcr, fpsr) ? 0xFFFF : 0;
            }
        });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorEqual32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto &Vresult, auto &Va, auto &Vb) { code.vfcmp_ceq_s(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorEqual64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        // FIXME MaybeStandardFPSCRValue
        EmitThreeOp(code, ctx, inst,
                                [&](auto &Vresult, auto &Va, auto &Vb) { code.vfcmp_ceq_d(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorFromHalf32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        const auto rounding_mode = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());

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
                                    [](VectorArray<u32>& output, const VectorArray<u16>& input, FP::FPCR fpcr, FP::FPSR& fpsr) {
                                        constexpr FP::RoundingMode rounding_mode = mp::get<0, I>::value;

                                        for (size_t i = 0; i < output.size(); ++i) {
                                            output[i] = FP::FPConvert<u32, u16>(input[i], fpcr, rounding_mode, fpsr);
                                        }
                                    })};
                },
                mp::cartesian_product<rounding_list>{});

        EmitTwoOpFallback<2>(code, ctx, inst, lut.at(std::make_tuple(rounding_mode)));
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorFromSignedFixed32>(BlockOfCode &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
//        EmitFromFixed<32>(code, ctx, inst, [&](auto Vto, auto Vfrom, u8 fbits) {
//            fbits ? code.vffint_s_w(Vto, Vfrom, fbits) : code.SCVTF(Vto, Vfrom);
//        });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorFromSignedFixed64>(BlockOfCode &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
//        EmitFromFixed<64>(code, ctx, inst, [&](auto Vto, auto Vfrom, u8 fbits) {
//            fbits ? code.SCVTF(Vto, Vfrom, fbits) : code.SCVTF(Vto, Vfrom);
//        });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorFromUnsignedFixed32>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
//        EmitFromFixed<32>(code, ctx, inst, [&](auto Vto, auto Vfrom, u8 fbits) {
//            fbits ? code.UCVTF(Vto, Vfrom, fbits) : code.UCVTF(Vto, Vfrom);
//        });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorFromUnsignedFixed64>(BlockOfCode &code, EmitContext &ctx,
                                                         IR::Inst *inst) {
        (void) code;
        (void) ctx;
        (void) inst;
        ASSERT_FALSE("Unimplemented");
        // FIXME EmitFPVectorFromUnsignedFixed64
//        EmitFromFixed<64>(code, ctx, inst, [&](auto Vto, auto Vfrom, u8 fbits) {

//            fbits ? code.UCVTF(Vto, Vfrom, fbits) : code.UCVTF(Vto, Vfrom);
//        });
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorGreater32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto &Vresult, auto &Va, auto &Vb) { code.vfcmp_clt_s(Vresult, Vb, Va); });
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorGreater64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto &Vresult, auto &Va, auto &Vb) { code.vfcmp_clt_d(Vresult, Vb, Va); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorGreaterEqual32>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto &Vresult, auto &Va, auto &Vb) { code.vfcmp_cle_s(Vresult, Vb, Va); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorGreaterEqual64>(BlockOfCode &code, EmitContext &ctx,
                                                    IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto &Vresult, auto &Va, auto &Vb) {code.vfcmp_cle_d(Vresult, Vb, Va);     });
    }



    template<>
    void EmitIR<IR::Opcode::FPVectorMax32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFPVectorMinMax<32, true>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorMax64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFPVectorMinMax<64, true>(code, ctx, inst);
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorMaxNumeric32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFPVectorMinMaxNumeric<32, true>(code, ctx, inst);
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorMaxNumeric64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFPVectorMinMaxNumeric<64, true>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorMin32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFPVectorMinMax<32, false>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorMin64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFPVectorMinMax<64, false>(code, ctx, inst);
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorMinNumeric32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFPVectorMinMaxNumeric<32, false>(code, ctx, inst);
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorMinNumeric64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFPVectorMinMaxNumeric<64, false>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorMul32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto &Vresult, auto &Va, auto &Vb) { code.vfmul_s(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorMul64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst,
                                [&](auto &Vresult, auto &Va, auto &Vb) { code.vfmul_d(Vresult, Va, Vb); });
    }

    template<size_t fsize>
    void EmitFPVectorMulAdd(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
        using FPT = mcl::unsigned_integer_of_size<fsize>;

        const auto fallback_fn = [](VectorArray<FPT>& result, const VectorArray<FPT>& addend, const VectorArray<FPT>& op1, const VectorArray<FPT>& op2, FP::FPCR fpcr, FP::FPSR& fpsr) {
            for (size_t i = 0; i < result.size(); i++) {
                result[i] = FP::FPMulAdd<FPT>(addend[i], op1[i], op2[i], fpcr, fpsr);
            }
        };
        
        EmitFourOpFallback(code, ctx, inst, fallback_fn);
    }
    
    template<>
    void
    EmitIR<IR::Opcode::FPVectorMulAdd16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFPVectorMulAdd<16>(code, ctx, inst);

    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorMulAdd32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFPVectorMulAdd<32>(code, ctx, inst);

//        EmitFMA<32>(code, ctx, inst, [&](auto &Va, auto Vn, auto Vm) { code.vfmadd_s(Va, Vn, Vm); });
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorMulAdd64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFPVectorMulAdd<64>(code, ctx, inst);

//        EmitFMA<64>(code, ctx, inst, [&](auto &Va, auto Vn, auto Vm) { code.vfmadd_d(Va, Vn, Vm); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorMulX32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        // FIXME , HandleNaNs
        EmitThreeOp(code, ctx, inst, [&](auto &Vresult, auto &Va, auto &Vb) { code.vfmul_s(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorMulX64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto &Vresult, auto &Va, auto &Vb) { code.vfmul_d(Vresult, Va, Vb); });
    }

    template<size_t fsize>
    void FPVectorNeg(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Qa = ctx.reg_alloc.ReadWriteQ(args[0], inst);
        using FPT = mcl::unsigned_integer_of_size<fsize>;
        constexpr FPT sign_mask = FP::FPInfo<FPT>::sign_mask;
        constexpr u64 sign_mask64 = mcl::bit::replicate_element<fsize, u64>(sign_mask);

        if constexpr (fsize == 64) {
            code.add_imm(Xscratch0, code.zero, sign_mask64, Xscratch1);
            code.vinsgr2vr_d(Vscratch2, Xscratch0 ,0);
            code.vxor_v(Qa, Qa, Vscratch2);
        } else if constexpr (fsize == 32) {
            code.add_imm(Xscratch0, code.zero, sign_mask64, Xscratch1);
            code.vinsgr2vr_w(Vscratch2, Xscratch0 ,0);
            code.vxor_v(Qa, Qa, Vscratch2);
        } else if constexpr (fsize == 16) {
            code.add_imm(Xscratch0, code.zero, sign_mask64, Xscratch1);
            code.vinsgr2vr_h(Vscratch2, Xscratch0 ,0);
            code.vxor_v(Qa, Qa, Vscratch2);
        }


//        ctx.reg_alloc.DefineValue(inst, a);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorNeg16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        FPVectorNeg<16>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorNeg32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        FPVectorNeg<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorNeg64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        FPVectorNeg<64>(code, ctx, inst);
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorPairedAdd32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto &Vresult, auto &Va, auto &Vb) {
            code.vpickev_w(Vresult, Vb, Va);
            code.vpickod_w(Vscratch2, Vb, Va);
            code.vfadd_s(Vresult, Vresult, Vscratch2);
//            code.vinsgr2vr_d(Vresult, code.zero , 1);
        });
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorPairedAdd64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto &Vresult, auto &Va, auto &Vb) {
            code.vpickev_d(Vresult, Vb, Va);
            code.vpickod_d(Vscratch2, Vb, Va);
            code.vfadd_d(Vresult, Vresult, Vscratch2);
//            code.vinsgr2vr_d(Vresult, code.zero , 1);
//            code.FADDP(Vresult, Va, Vb);
        });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorPairedAddLower32>(BlockOfCode &code, EmitContext &ctx,
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
    void EmitIR<IR::Opcode::FPVectorPairedAddLower64>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto &Qresult, auto &Qa, auto &Qb) {
            code.vfadd_d(Qresult, Qa, Qb);
            code.vinsgr2vr_d(Qresult, code.zero ,1);
//            code.ZIP1(V0.D2(), Qa->D2(), Qb->D2());
//            code.FADDP(Qresult->toD(), V0.D2());
        });
    }

    template<size_t fsize>
    static void EmitRecipEstimate(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
        using FPT = mcl::unsigned_integer_of_size<fsize>;

        EmitTwoOpFallback(code, ctx, inst, [](VectorArray<FPT>& result, const VectorArray<FPT>& operand, FP::FPCR fpcr, FP::FPSR& fpsr) {
            for (size_t i = 0; i < result.size(); i++) {
                result[i] = FP::FPRecipEstimate<FPT>(operand[i], fpcr, fpsr);
            }
        });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRecipEstimate16>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitRecipEstimate<16>(code, ctx, inst);

    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRecipEstimate32>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitRecipEstimate<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRecipEstimate64>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitRecipEstimate<64>(code, ctx, inst);
    }


    template<size_t fsize>
    static void EmitRecipStepFused(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
        using FPT = mcl::unsigned_integer_of_size<fsize>;

        const auto fallback_fn = [](VectorArray<FPT>& result, const VectorArray<FPT>& op1, const VectorArray<FPT>& op2, FP::FPCR fpcr, FP::FPSR& fpsr) {
            for (size_t i = 0; i < result.size(); i++) {
                result[i] = FP::FPRecipStepFused<FPT>(op1[i], op2[i], fpcr, fpsr);
            }
        };

        EmitThreeOpFallback(code, ctx, inst, fallback_fn);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRecipStepFused16>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitRecipStepFused<16>(code, ctx, inst);

    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRecipStepFused32>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitRecipStepFused<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRecipStepFused64>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitRecipStepFused<64>(code, ctx, inst);
    }

    template<size_t fsize>
    void EmitFPVectorRoundInt(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
        const auto rounding = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());
        const bool exact = inst->GetArg(2).GetU1();


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
    EmitIR<IR::Opcode::FPVectorRoundInt16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFPVectorRoundInt<16>(code, ctx, inst);

    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorRoundInt32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFPVectorRoundInt<32>(code, ctx, inst);

    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorRoundInt64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitFPVectorRoundInt<64>(code, ctx, inst);

    }

    template<size_t fsize>
    static void EmitRSqrtEstimate(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
        using FPT = mcl::unsigned_integer_of_size<fsize>;

        const auto fallback_fn = [](VectorArray<FPT>& result, const VectorArray<FPT>& operand, FP::FPCR fpcr, FP::FPSR& fpsr) {
            for (size_t i = 0; i < result.size(); i++) {
                result[i] = FP::FPRSqrtEstimate<FPT>(operand[i], fpcr, fpsr);
            }
        };

        EmitTwoOpFallback(code, ctx, inst, fallback_fn);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRSqrtEstimate16>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitRSqrtEstimate<16>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRSqrtEstimate32>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitRSqrtEstimate<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRSqrtEstimate64>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitRSqrtEstimate<64>(code, ctx, inst);
    }

    template<size_t fsize>
    static void EmitRSqrtStepFused(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
        using FPT = mcl::unsigned_integer_of_size<fsize>;

        const auto fallback_fn = [](VectorArray<FPT>& result, const VectorArray<FPT>& op1, const VectorArray<FPT>& op2, FP::FPCR fpcr, FP::FPSR& fpsr) {
            for (size_t i = 0; i < result.size(); i++) {
                result[i] = FP::FPRSqrtStepFused<FPT>(op1[i], op2[i], fpcr, fpsr);
            }
        };

        EmitThreeOpFallback(code, ctx, inst, fallback_fn);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRSqrtStepFused16>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitRSqrtStepFused<16>(code, ctx, inst);

    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRSqrtStepFused32>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitRSqrtStepFused<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorRSqrtStepFused64>(BlockOfCode &code, EmitContext &ctx,
                                                      IR::Inst *inst) {
        EmitRSqrtStepFused<64>(code, ctx, inst);

    }

    template<>
    void EmitIR<IR::Opcode::FPVectorSqrt32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Va) { code.vfsqrt_s(Vresult, Va); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorSqrt64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitTwoOp(code, ctx, inst, [&](auto &Vresult, auto &Va) { code.vfsqrt_d(Vresult, Va); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorSub32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto &Vresult, auto &Va, auto &Vb) { code.vfsub_s(Vresult, Va, Vb); });
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorSub64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitThreeOp(code, ctx, inst, [&](auto &Vresult, auto &Va, auto &Vb) { code.vfsub_d(Vresult, Va, Vb); });
    }

    template<>
    void
    EmitIR<IR::Opcode::FPVectorToHalf32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        const auto rounding_mode = static_cast<FP::RoundingMode>(args[1].GetImmediateU8());
        ASSERT(rounding_mode == FP::RoundingMode::ToNearest_TieEven);
//        const bool fpcr_controlled = args[2].GetImmediateU1();

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
    void EmitFPVectorToFixed(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
        const size_t fbits = inst->GetArg(1).GetU8();
        const auto rounding = static_cast<FP::RoundingMode>(inst->GetArg(2).GetU8());
        [[maybe_unused]] const bool fpcr_controlled = inst->GetArg(3).GetU1();

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
    void EmitIR<IR::Opcode::FPVectorToSignedFixed16>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitFPVectorToFixed<16, false>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorToSignedFixed32>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitFPVectorToFixed<32, false>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorToSignedFixed64>(BlockOfCode &code, EmitContext &ctx,
                                                     IR::Inst *inst) {
        EmitFPVectorToFixed<64, false>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorToUnsignedFixed16>(BlockOfCode &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitFPVectorToFixed<16, true>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorToUnsignedFixed32>(BlockOfCode &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitFPVectorToFixed<32, true>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::FPVectorToUnsignedFixed64>(BlockOfCode &code, EmitContext &ctx,
                                                       IR::Inst *inst) {
        EmitFPVectorToFixed<64, true>(code, ctx, inst);

    }

}  // namespace Dynarmic::Backend::LoongArch64
