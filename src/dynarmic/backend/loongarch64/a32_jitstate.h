/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>

#include <mcl/stdint.hpp>

#include "dynarmic/frontend/A32/a32_location_descriptor.h"

namespace Dynarmic::Backend::LoongArch64 {

    class BlockOfCode;

#ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable : 4324)  // Structure was padded due to alignment specifier
#endif

    struct A32JitState {
        using ProgramCounterType = u32;

        A32JitState() { ResetRSB(); }

        std::array<u32, 16> regs{};  // Current register file.
        // TODO: Mode-specific register sets unimplemented.

        u32 upper_location_descriptor = 0;
        u32 cpsr_jaifm = 0;
        u32 cpsr_ge = 0;
        u32 cpsr_nzcv = 0;
        u32 cpsr_q = 0;

        u32 Cpsr() const;

        void SetCpsr(u32 cpsr);

        alignas(16) std::array<u32, 64> ext_regs{};  // Extension registers.

        // For internal use (See: BlockOfCode::RunCode)
        u32 guest_FCSR = 0x00001f80;
        u32 asimd_MXCSR = 0x00009fc0;
        volatile u32 halt_reason = 0;

        // Exclusive state
        u32 exclusive_state = 0;

        static constexpr size_t RSBSize = 8;  // MUST be a power of 2.
        static constexpr size_t RSBPtrMask = RSBSize - 1;
        u32 rsb_ptr = 0;
        std::array<u64, RSBSize> rsb_location_descriptors;
        std::array<u64, RSBSize> rsb_codeptrs;

        void ResetRSB();

        u32 fpsr_exc = 0;
        u32 fpsr_qc = 0;
        u32 fpsr = 0;
        u32 fpsr_nzcv = 0;

        u32 Fpscr() const;

        void SetFpscr(u32 FPSCR);

        IR::LocationDescriptor GetLocationDescriptor() const {
            return IR::LocationDescriptor{regs[15] | (static_cast<u64>(upper_location_descriptor) << 32)};
        }
//    u64 GetUniqueHash() const noexcept {
//        return (static_cast<u64>(upper_location_descriptor) << 32) | (static_cast<u64>(Reg[15]));
//    }

    };

#ifdef _MSC_VER
#    pragma warning(pop)
#endif

//using CodePtr = const void*;

}  // namespace Dynarmic::Backend::X64
