/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <functional>
#include <memory>
#include <type_traits>

#include <mcl/stdint.hpp>

#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/callback.h"
#include "dynarmic/backend/loongarch64/jitstate_info.h"
#include "dynarmic/common/cast_util.h"
#include "dynarmic/interface/halt_reason.h"
#include "dynarmic/ir/cond.h"

#include "xbyak_loongarch64.h"

namespace Dynarmic::Backend::LoongArch64 {

using CodePtr = std::byte*;

    class BlockOfCode final : public Xbyak_loongarch64::CodeGenerator {
public:
    BlockOfCode(size_t total_code_size, JitStateInfo jsi);
    BlockOfCode(const BlockOfCode&) = delete;

    /// Call when external emitters have finished emitting their preludes.
    void PreludeComplete();

    /// Change permissions to RW. This is required to support systems with W^X enforced.
    void EnableWriting();
    /// Change permissions to RX. This is required to support systems with W^X enforced.
    void DisableWriting();

    /// Clears this block of code and resets code pointer to beginning.
    void ClearCache();
    /// Calculates how much space is remaining to use.
    size_t SpaceRemaining() const;
    /// Ensure at least codesize bytes of code cache memory are committed at the current code_ptr.
    void EnsureMemoryCommitted(size_t codesize);

    /// Runs emulated code from code_ptr.
    HaltReason RunCode(void* jit_state, CodePtr code_ptr) const;
    /// Runs emulated code from code_ptr for a single cycle.
    HaltReason StepCode(void* jit_state, CodePtr code_ptr) const;

        /// Code emitter: Makes guest MXCSR the current MXCSR
    void SwitchMxcsrOnEntry();
    /// Code emitter: Makes saved host MXCSR the current MXCSR
    void SwitchMxcsrOnExit();
    /// Code emitter: Enter standard ASIMD MXCSR region
    void EnterStandardASIMD();
    void LeaveStandardASIMD() ;

    /// Code emitter: Load required flags for conditional cond from rax into host rflags
    void LoadRequiredFlagsForCondFromRax(IR::Cond cond);

    /// Code emitter: Calls the function
    template<typename FunctionPointer>
    void CallFunction(FunctionPointer fn) {
        static_assert(std::is_pointer_v<FunctionPointer> && std::is_function_v<std::remove_pointer_t<FunctionPointer>>,
                      "Supplied type must be a pointer to a function");

        const u64 address = reinterpret_cast<u64>(fn);
        // Far call FIXME?
        const u64 distance = address - (getCurr<u64>());
        if (distance >= 0x2000000ULL && distance < 0xFFFFFFFFFC000000ULL) {
            // Far call
            add_imm(Xscratch0, zero, address, Xscratch1);
            jirl(ra, Xscratch0, 0);
        } else {
            bl((int64_t)distance);
        }

    }
    /// Code emitter: Calls the function
    template<typename FunctionPointer>
    void JumpFunction(FunctionPointer fn) {
        static_assert(std::is_pointer_v<FunctionPointer> && std::is_function_v<std::remove_pointer_t<FunctionPointer>>,
                      "Supplied type must be a pointer to a function");

        const u64 address = reinterpret_cast<u64>(fn);
        // Far call FIXME?
        const u64 distance = address - (getCurr<u64>());
        if (distance >= 0x2000000ULL && distance < 0xFFFFFFFFFC000000ULL) {
            // Far call
            add_imm(Xscratch0, zero, address, Xscratch1);
            jirl(zero, Xscratch0, 0);
        } else {
            b((int64_t)distance);
        }

    }
    /// Code emitter: Calls the lambda. Lambda must not have any captures.
    template<typename Lambda>
    void CallLambda(Lambda l) {
        CallFunction(Common::FptrCast(l));
    }

    CodePtr GetCodeBegin() const;
    size_t GetTotalCodeSize() const;

    const void* GetReturnFromRunCodeAddress() const {
        return return_from_run_code[0];
    }

    const void* GetForceReturnFromRunCodeAddress() const {
        return return_from_run_code[FORCE_RETURN];
    }


    /// Allocate memory of `size` bytes from the same block of memory the code is in.
    /// This is useful for objects that need to be placed close to or within code.
    /// The lifetime of this memory is the same as the code around it.
    void* AllocateFromCodeSpace(size_t size);

    void SetCodePtr(CodePtr code_ptr);
    void B(const void *a);
    void BL(const void *a);
    void LDLableData_d(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::Label &label);
    void LDLableData_w(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::Label &label);

        // ABI registers

    static const Xbyak_loongarch64::XReg ABI_RETURN;
    static const Xbyak_loongarch64::XReg ABI_RETURN2;
    static const Xbyak_loongarch64::XReg ABI_PARAM1;
    static const Xbyak_loongarch64::XReg ABI_PARAM2;
    static const Xbyak_loongarch64::XReg ABI_PARAM3;
    static const Xbyak_loongarch64::XReg ABI_PARAM4;
    static const Xbyak_loongarch64::XReg ABI_PARAM5;
    static const Xbyak_loongarch64::XReg ABI_PARAM6;
    static const std::array<Xbyak_loongarch64::XReg, ABI_PARAM_COUNT> ABI_PARAMS;

    JitStateInfo GetJitStateInfo() const { return jsi; }

    private:
        JitStateInfo jsi;

    bool prelude_complete = false;
    CodePtr code_begin = nullptr;

    using RunCodeFuncType = HaltReason (*)(void*, CodePtr);
    RunCodeFuncType run_code = nullptr;
    RunCodeFuncType step_code = nullptr;
    static constexpr size_t MXCSR_ALREADY_EXITED = 1 << 0;
    static constexpr size_t FORCE_RETURN = 1 << 1;
    std::array<const void*, 4> return_from_run_code;

    };

}  // namespace Dynarmic::Backend::LoongArch64
