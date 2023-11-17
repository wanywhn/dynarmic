/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/loongarch64/abi.h"

#include <vector>

#include <mcl/bit/bit_field.hpp>
#include <mcl/stdint.hpp>

#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"

namespace Dynarmic::Backend::LoongArch64 {

using namespace Xbyak_loongarch64::util;

static constexpr size_t gpr_size = 8;
static constexpr size_t fpr_size = 16;

struct FrameInfo {
    std::vector<unsigned int> gprs;
    std::vector<unsigned int> fprs;
    size_t frame_size;
    size_t gprs_size;
    size_t fprs_size;
};

static std::vector<unsigned int> ListToIndexes(u32 list) {
    std::vector<unsigned int> indexes;
    for (int i = 0; i < 32; i++) {
        if (mcl::bit::get_bit(i, list)) {
            indexes.emplace_back(i);
        }
    }
    return indexes;
}

static FrameInfo CalculateFrameInfo(RegisterList rl, size_t frame_size) {
    const auto gprs = ListToIndexes(static_cast<u32>(rl));
    const auto fprs = ListToIndexes(static_cast<u32>(rl >> 32));

    const size_t num_gprs = gprs.size();
    const size_t num_fprs = fprs.size();

    const size_t gprs_size = (num_gprs + 1) / 2 * 16;
    const size_t fprs_size = num_fprs * 16;

    return {
        gprs,
        fprs,
        frame_size,
        gprs_size,
        fprs_size,
    };
}

#define DO_IT(TYPE, REG_TYPE, SINGLE_OP, OFFSET)                                                                     \
    if (frame_info.TYPE##s.size() > 0) {                                                                             \
        for (size_t i = 0; i < frame_info.TYPE##s.size(); i += 1) {                                                  \
            code.SINGLE_OP(Xbyak_loongarch64::REG_TYPE{frame_info.TYPE##s[i]}, code.sp, (OFFSET) + i * TYPE##_size); \
        }                                                                                                            \
    }

void ABI_PushRegisters(Xbyak_loongarch64::CodeGenerator& code, RegisterList rl, size_t frame_size) {
    const FrameInfo frame_info = CalculateFrameInfo(rl, frame_size);

    code.sub_imm(code.sp, code.sp, frame_info.gprs_size + frame_info.fprs_size, code.t0);

    DO_IT(gpr, XReg, stptr_d, 0)
    DO_IT(fpr, XReg, fst_d, frame_info.gprs_size)

    code.sub_imm(code.sp, code.sp, frame_info.frame_size, code.t0);
}

void ABI_PopRegisters(Xbyak_loongarch64::CodeGenerator& code, RegisterList rl, size_t frame_size) {
    const FrameInfo frame_info = CalculateFrameInfo(rl, frame_size);

    code.add_imm(code.sp, code.sp, frame_info.frame_size, code.t0);

    DO_IT(gpr, XReg, ldptr_d, 0)
    DO_IT(fpr, XReg, fld_d, frame_info.gprs_size)

    code.add_imm(code.sp, code.sp, frame_info.gprs_size + frame_info.fprs_size, code.t0);
}

}  // namespace Dynarmic::Backend::LoongArch64
