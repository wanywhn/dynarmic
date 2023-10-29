/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/loongarch64/a64_jitstate.h"

#include <mcl/bit/bit_field.hpp>

#include "dynarmic/frontend/A64/a64_location_descriptor.h"

namespace Dynarmic::Backend::LoongArch64 {

/**
 * Comparing LoongArch FCSR and Arm FPCR
 * ========================
 *
 * LoongArch FCSR exception masks
 * -------------------------
 * Enables  bit 4  Invalid Operation exception trap enable
 * Enables  bit 3  Division by Zero exception trap enable
 * Enables  bit 2  Overflow exception trap enable
 * Enables  bit 1  Underflow exception trap enable
 * Enables  bit 0  Inexact exception trap enable
 *
 * A64 FPCR exception trap enables
 * -------------------------------
 * IDE  bit 15  Input Denormal exception trap enable
 * IXE  bit 12  Inexact exception trap enable
 * UFE  bit 11  Underflow exception trap enable
 * OFE  bit 10  Overflow exception trap enable
 * DZE  bit 9   Division by Zero exception trap enable
 * IOE  bit 8   Invalid Operation exception trap enable
 *
 * LoongArch FCSR mode bits
 * -------------------
 * RM   bits 8-9  Round to {0 = TiesToEven, 1 = Zero, 2 = Positive, 3 = Negative}
 *
 * A64 FPCR mode bits
 * ------------------
 * AHP  bit 26  Alternative half-precision
 * DN   bit 25  Default NaN
 * FZ   bit 24  Flush to Zero
 * RMode    bits 22-23  Round to {0 = Nearest, 1 = Positive, 2 = Negative, 3 = Zero}
 * FZ16 bit 19  Flush to Zero for half-precision
 */

    constexpr u32 FPCR_MASK = 0x07C89F00;

    u32 A64JitState::GetFpcr() const {
        return fpcr;
    }

    void A64JitState::SetFpcr(u32 value) {
        fpcr = value & FPCR_MASK;

        asimd_MXCSR &= 0x1F1F0000UL;
        guest_FCSR &= 0x1F1F0000UL;
        asimd_MXCSR &= ~0x1FUL;
        guest_FCSR &= ~0x1FUL;  // Mask all exceptions

        // RMode
        const std::array<u32, 4> MXCSR_RMode{0x0, 0x200, 0x300, 0x100};
        guest_FCSR |= MXCSR_RMode[(value >> 22) & 0x3];

        if (mcl::bit::get_bit<24>(value)) {
//        guest_FCSR |= (1 << 15);  // SSE Flush to Zero
//        guest_FCSR |= (1 << 6);   // SSE Denormals are Zero
        }
    }

/**
 * Comparing LoongArch FCSR and FPSR
 * ========================
 *
 * LoongArch FCSR exception flags
 * -------------------------
 * UE   bit 20   Invalid Operation cumulative exception bit
 * OE   bit 19   Division by Zero cumulative exception bit
 * ZE   bit 18   Overflow cumulative exception bit
 * DE   bit 17   Underflow cumulative exception bit
 * IE   bit 16   Inexact cumulative exception bit
 *
 * A64 FPSR cumulative exception bits
 * ----------------------------------
 * QC   bit 27  Cumulative saturation bit
 * IDC  bit 7   Input Denormal cumulative exception bit       // Only ever set when FPCR.FTZ = 1
 * IXC  bit 4   Inexact cumulative exception bit
 * UFC  bit 3   Underflow cumulative exception bit
 * OFC  bit 2   Overflow cumulative exception bit
 * DZC  bit 1   Division by Zero cumulative exception bit
 * IOC  bit 0   Invalid Operation cumulative exception bit
 */

    u32 A64JitState::GetFpsr() const {
        const u32 mxcsr = guest_FCSR | asimd_MXCSR;
        u32 fpsr = 0;
        fpsr |= (mxcsr & 0x100000) >> 20;
        fpsr |= ((mxcsr & 0x80000) >> 19) << 1;
        fpsr |= ((mxcsr & 0x40000) >> 18) << 2;
        fpsr |= ((mxcsr & 0x20000) >> 17) << 3;
        fpsr |= ((mxcsr & 0x10000) >> 16) << 4;
        fpsr |= fpsr_exc;
        fpsr |= (fpsr_qc == 0 ? 0 : 1) << 27;
        return fpsr;
    }

    void A64JitState::SetFpsr(u32 value) {
        guest_FCSR &= ~0x1F000000;
        asimd_MXCSR &= ~0x0000003D;
        fpsr_qc = (value >> 27) & 1;
        fpsr_exc = value & 0x9F;
    }

}  // namespace Dynarmic::Backend::X64
