/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <mcl/bit_cast.hpp>
#include <mcl/stdint.hpp>
#include <mcl/type_traits/function_info.hpp>
#include <boost/predef.h>

namespace Dynarmic::Backend::LoongArch64 {

    struct DevirtualizedCall {
        u64 fn_ptr;
        u64 this_ptr;
    };

// https://github.com/ARM-software/abi-aa/blob/main/cppabi64/cppabi64.rst#representation-of-pointer-to-member-function
    template<auto mfp>
    DevirtualizedCall DevirtualizeDefault(mcl::class_type<decltype(mfp)> *this_) {
        struct MemberFunctionPointer {
            // Address of non-virtual function or index into vtable.
            u64 ptr;
            // LSB is discriminator for if function is virtual. Other bits are this adjustment.
            u64 adj;
        } mfp_struct = mcl::bit_cast<MemberFunctionPointer>(mfp);

        static_assert(sizeof(MemberFunctionPointer) == 16);
        static_assert(sizeof(MemberFunctionPointer) == sizeof(mfp));

        u64 fn_ptr = mfp_struct.ptr;
        u64 this_ptr = mcl::bit_cast<u64>(this_) + (mfp_struct.adj >> 1);
        if (mfp_struct.adj & 1) {
            u64 vtable = mcl::bit_cast_pointee<u64>(this_ptr);
            fn_ptr = mcl::bit_cast_pointee<u64>(vtable + fn_ptr);
        }
        return DevirtualizedCall{fn_ptr, this_ptr};
    }
// http://itanium-cxx-abi.github.io/cxx-abi/abi.html
    template<auto mfp>
    DevirtualizedCall DevirtualizeItanium(mcl::class_type<decltype(mfp)>* this_) {
        struct MemberFunctionPointer {
            /// For a non-virtual function, this is a simple function pointer.
            /// For a virtual function, it is (1 + virtual table offset in bytes).
            u64 ptr;
            /// The required adjustment to `this`, prior to the call.
            u64 adj;
        } mfp_struct = mcl::bit_cast<MemberFunctionPointer>(mfp);

        static_assert(sizeof(MemberFunctionPointer) == 16);
        static_assert(sizeof(MemberFunctionPointer) == sizeof(mfp));

        u64 fn_ptr = mfp_struct.ptr;
        u64 this_ptr = reinterpret_cast<u64>(this_) + mfp_struct.adj;
        if (mfp_struct.ptr & 1) {
            u64 vtable = mcl::bit_cast_pointee<u64>(this_ptr);
            fn_ptr = mcl::bit_cast_pointee<u64>(vtable + fn_ptr - 1);
        }
        return DevirtualizedCall{fn_ptr, this_ptr};
    }

    template<auto mfp>
    DevirtualizedCall Devirtualize(mcl::class_type<decltype(mfp)> *this_) {
#if BOOST_COMP_GNUC
        return DevirtualizeDefault<mfp>(this_);
#elif BOOST_COMP_CLANG
        return DevirtualizeItanium<mfp>(this_);
#endif
    }

}  // namespace Dynarmic::Backend::LoongArch64
