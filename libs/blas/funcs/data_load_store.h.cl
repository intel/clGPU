/* Copyright (c) 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// =====================================================================================================================
// =====================================================================================================================
// =====================================================================================================================
// File: data_load_store.h.cl

#ifndef DATA_LOAD_STORE_H
#define DATA_LOAD_STORE_H

// Environment:
#ifndef SG_SUPPORTED
    #define SG_SUPPORTED 1
    #define DATA_LOAD_STORE_SG_SUPPORTED_UNDEF_MACRO_
#endif

// Includes:
#include <pp_load_store_helpers.h>


// =====================================================================================================================
// Loading data:
// =====================================================================================================================
//
// LOAD_DATA            - Loads continous block of data of specified type and number of elements. Does not check for
//                        limits. Can use different algorithms to load data depending on load size, type, etc.
// LOAD_DATA_WITH_LIMIT - Loads continous block of data of specified type and number of elements. Provides limit which
//                        restricts reading up to specified offset value. The buffer is filled with
//                        DATA_LS_LIMIT_FILL_VALUE for any data outside allowed range: [ptr, ptr + ptr_byte_limit).
//                        Can use different algorithms to load data depending on load size, type, etc.
//
// LOAD_DATA(type, sg_size, size, buffer, index, ptr, ptr_byte_offset)
// LOAD_DATA_WITH_LIMIT(type, sg_size, size, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)
//
// Customization:
//  DATA_LS_RO_AS            - Name of address space used as read-only address space (load address space).
//                             Default: __global.
//  DATA_LS_LIMIT_FILL_VALUE - Value used as filling value for any data outside the load range / limit.
//                             Default: 0.
//
// Environment:
//  SG_SUPPORTED - Indicates that sub-groups are supported (default: 1). Allowed values: 0, 1.
//
// Parameters:
//  - type            - [PP] Type of element to load into target "buffer".
//  - sg_size         - [PP] Selected sub-group size or SIMD size. Allowed values: 8, 16, 32.
//  - size            - [PP] Number of elements to load. Must be multiple of "sg_size" and currently limited to 256.
//  - buffer          - [UNIFORM] An lvalue expression of array of type elements. Please use expression without
//                      side-effects (it will be expanded possibly more than once). Destination buffer.
//  - index           - [UNIFORM] (0-based) Start position where data should be stored/written into "buffer".
//  - ptr             - [UNIFORM] Pointer convertible to "DATA_LS_RO_AS const void*" that represents
//                      input data to load. 
//  - ptr_byte_offset - [UNIFORM] A byte offset (unsigned integer) from start of input data pointed by ptr.
//                      It will be scattered if necessary. The data is loaded from "ptr" offset by "ptr_byte_offset".
//  - ptr_byte_limit  - [UNIFORM] Exclusive limit in bytes to which maximum read from "ptr" can take place. Any element
//                      which offset is outside [ptr, ptr + ptr_byte_limit) range will not be loaded.
//                      If element is not loaded because it is outside allowed range, the corresponding "buffer"
//                      location is filled with DATA_LS_LIMIT_FILL_VALUE.
//
// LUT mangled params:
//  - T<type>
//  - SG<sg_size>
//  - GC<grp_cnt> - [PP] Number of scalar read operations needed to read "size" elements.
//  - LC<0|1>     - [PP] Indicates that limit check is enabled. Allowed values: 0, 1.
//
// =====================================================================================================================

#ifndef DATA_LS_RO_AS
    #define DATA_LS_RO_AS              __global
#endif

#ifndef DATA_LS_LIMIT_FILL_VALUE
    #define DATA_LS_LIMIT_FILL_VALUE   0
#endif


#define LOAD_DATA_Txx_SGxx_GCxx_LC0_H1_(type, sg_size, grp_cnt, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)   \
do {                                                                                                                   \
    const uint LOAD_DATA_tmp_idx_   = (index);                                                                         \
    uint LOAD_DATA_tmp_byte_offset_ = SCATTER_TYPE_OFFSET_BY_LID(S, type, ptr_byte_offset);                            \
                                                                                                                       \
    __attribute__((opencl_unroll_hint))                                                                                \
    for (uint LOAD_DATA_tmp_li_ = 0; LOAD_DATA_tmp_li_ < (grp_cnt); ++LOAD_DATA_tmp_li_)                               \
    {                                                                                                                  \
        (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_] =                                                             \
            *(DATA_LS_RO_AS const type*)((DATA_LS_RO_AS const char*)(ptr) + LOAD_DATA_tmp_byte_offset_);               \
        LOAD_DATA_tmp_byte_offset_ = SHIFT_TYPE_OFFSET(type, LOAD_DATA_tmp_byte_offset_, sg_size);                     \
    }                                                                                                                  \
} while (false)

#define LOAD_DATA_Tfloat_SGxx_GCxx_LC0_H2_(sg_size, grp_cnt, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)   \
do {                                                                                                                \
    const uint LOAD_DATA_tmp_idx_   = (index);                                                                      \
    uint LOAD_DATA_tmp_byte_offset_ = SCATTER_TYPE_OFFSET_BY_LID(B, float, ptr_byte_offset);                        \
    const uint LOAD_DATA_tmp_lc_    = (grp_cnt);                                                                    \
    uint LOAD_DATA_tmp_li_          = 0;                                                                            \
                                                                                                                    \
    __attribute__((opencl_unroll_hint))                                                                             \
    for (; LOAD_DATA_tmp_li_ + 7 < LOAD_DATA_tmp_lc_; LOAD_DATA_tmp_li_ += 8)                                       \
    {                                                                                                               \
        float8 LOAD_DATA_tmps_ = as_float8(intel_sub_group_block_read8(                                             \
            (DATA_LS_RO_AS const uint*)((DATA_LS_RO_AS const char*)(ptr) + LOAD_DATA_tmp_byte_offset_)));           \
        (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_]     = LOAD_DATA_tmps_.s0;                                  \
        (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_ + 1] = LOAD_DATA_tmps_.s1;                                  \
        (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_ + 2] = LOAD_DATA_tmps_.s2;                                  \
        (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_ + 3] = LOAD_DATA_tmps_.s3;                                  \
        (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_ + 4] = LOAD_DATA_tmps_.s4;                                  \
        (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_ + 5] = LOAD_DATA_tmps_.s5;                                  \
        (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_ + 6] = LOAD_DATA_tmps_.s6;                                  \
        (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_ + 7] = LOAD_DATA_tmps_.s7;                                  \
        LOAD_DATA_tmp_byte_offset_ = SHIFT_TYPE_OFFSET(float, LOAD_DATA_tmp_byte_offset_, 8 * sg_size);             \
    }                                                                                                               \
                                                                                                                    \
    __attribute__((opencl_unroll_hint))                                                                             \
    for (; LOAD_DATA_tmp_li_ + 3 < LOAD_DATA_tmp_lc_; LOAD_DATA_tmp_li_ += 4)                                       \
    {                                                                                                               \
        float4 LOAD_DATA_tmps_ = as_float4(intel_sub_group_block_read4(                                             \
            (DATA_LS_RO_AS const uint*)((DATA_LS_RO_AS const char*)(ptr) + LOAD_DATA_tmp_byte_offset_)));           \
        (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_]     = LOAD_DATA_tmps_.s0;                                  \
        (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_ + 1] = LOAD_DATA_tmps_.s1;                                  \
        (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_ + 2] = LOAD_DATA_tmps_.s2;                                  \
        (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_ + 3] = LOAD_DATA_tmps_.s3;                                  \
        LOAD_DATA_tmp_byte_offset_ = SHIFT_TYPE_OFFSET(float, LOAD_DATA_tmp_byte_offset_, 4 * sg_size);             \
    }                                                                                                               \
                                                                                                                    \
    __attribute__((opencl_unroll_hint))                                                                             \
    for (; LOAD_DATA_tmp_li_ + 1 < LOAD_DATA_tmp_lc_; LOAD_DATA_tmp_li_ += 2)                                       \
    {                                                                                                               \
        float2 LOAD_DATA_tmps_ = as_float2(intel_sub_group_block_read2(                                             \
            (DATA_LS_RO_AS const uint*)((DATA_LS_RO_AS const char*)(ptr) + LOAD_DATA_tmp_byte_offset_)));           \
        (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_]     = LOAD_DATA_tmps_.s0;                                  \
        (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_ + 1] = LOAD_DATA_tmps_.s1;                                  \
        LOAD_DATA_tmp_byte_offset_ = SHIFT_TYPE_OFFSET(float, LOAD_DATA_tmp_byte_offset_, 2 * sg_size);             \
    }                                                                                                               \
                                                                                                                    \
    __attribute__((opencl_unroll_hint))                                                                             \
    for (; LOAD_DATA_tmp_li_ < LOAD_DATA_tmp_lc_; ++LOAD_DATA_tmp_li_)                                              \
    {                                                                                                               \
        (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_] =                                                          \
            as_float(intel_sub_group_block_read(                                                                    \
                (DATA_LS_RO_AS const uint*)((DATA_LS_RO_AS const char*)(ptr) + LOAD_DATA_tmp_byte_offset_)));       \
        LOAD_DATA_tmp_byte_offset_ = SHIFT_TYPE_OFFSET(float, LOAD_DATA_tmp_byte_offset_, sg_size);                 \
    }                                                                                                               \
} while (false)


#define LOAD_DATA_Txx_SGxx_GCxx_LC1_H1_(type, sg_size, grp_cnt, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit) \
do {                                                                                                                 \
    const uint LOAD_DATA_tmp_type_loff   = (uint) sizeof(type) - 1;                                                  \
    const uint LOAD_DATA_tmp_idx_        = (index);                                                                  \
    uint LOAD_DATA_tmp_byte_offset_      = SCATTER_TYPE_OFFSET_BY_LID(S, type, ptr_byte_offset);                     \
    const uint LOAD_DATA_tmp_byte_limit_ = (ptr_byte_limit);                                                         \
                                                                                                                     \
    __attribute__((opencl_unroll_hint))                                                                              \
    for (uint LOAD_DATA_tmp_li_ = 0; LOAD_DATA_tmp_li_ < (grp_cnt); ++LOAD_DATA_tmp_li_)                             \
    {                                                                                                                \
        if (LOAD_DATA_tmp_byte_offset_ + LOAD_DATA_tmp_type_loff >= LOAD_DATA_tmp_byte_limit_)                       \
        {                                                                                                            \
            (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_] = (DATA_LS_LIMIT_FILL_VALUE);                           \
        }                                                                                                            \
        else                                                                                                         \
        {                                                                                                            \
            (buffer)[LOAD_DATA_tmp_idx_ + LOAD_DATA_tmp_li_] =                                                       \
                *(DATA_LS_RO_AS const type*)((DATA_LS_RO_AS const char*)(ptr) + LOAD_DATA_tmp_byte_offset_);         \
        }                                                                                                            \
        LOAD_DATA_tmp_byte_offset_ = SHIFT_TYPE_OFFSET(type, LOAD_DATA_tmp_byte_offset_, sg_size);                   \
    }                                                                                                                \
} while (false)


#define LOAD_DATA_Tfloat_LC1_(sg_size, size, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)     \
    LOAD_DATA_Txx_SGxx_GCxx_LC1_H1_(float, sg_size, GRP_CNT(size, sg_size), buffer, index,            \
                                    ptr, ptr_byte_offset, ptr_byte_limit)
#define LOAD_DATA_Tdouble_LC1_(sg_size, size, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)    \
    LOAD_DATA_Txx_SGxx_GCxx_LC1_H1_(double, sg_size, GRP_CNT(size, sg_size), buffer, index,           \
                                    ptr, ptr_byte_offset, ptr_byte_limit)
#define LOAD_DATA_Tfloat2_LC1_(sg_size, size, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)    \
    LOAD_DATA_Txx_SGxx_GCxx_LC1_H1_(float2, sg_size, GRP_CNT(size, sg_size), buffer, index,           \
                                    ptr, ptr_byte_offset, ptr_byte_limit)
#define LOAD_DATA_Tdouble2_LC1_(sg_size, size, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)   \
    LOAD_DATA_Txx_SGxx_GCxx_LC1_H1_(double2, sg_size, GRP_CNT(size, sg_size), buffer, index,          \
                                    ptr, ptr_byte_offset, ptr_byte_limit)


#if SG_SUPPORTED
    #define LOAD_DATA_Tfloat_LC0_(sg_size, size, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)   \
        LOAD_DATA_Tfloat_SGxx_GCxx_LC0_H2_(sg_size, GRP_CNT(size, sg_size), buffer, index,              \
                                           ptr, ptr_byte_offset, ptr_byte_limit)
#else
    #define LOAD_DATA_Tfloat_LC0_(sg_size, size, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)   \
        LOAD_DATA_Txx_SGxx_GCxx_LC0_H1_(float, sg_size, GRP_CNT(size, sg_size), buffer, index,          \
                                        ptr, ptr_byte_offset, ptr_byte_limit)
#endif

#define LOAD_DATA_Tdouble_LC0_(sg_size, size, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)    \
    LOAD_DATA_Txx_SGxx_GCxx_LC0_H1_(double, sg_size, GRP_CNT(size, sg_size), buffer, index,           \
                                    ptr, ptr_byte_offset, ptr_byte_limit)                             
#define LOAD_DATA_Tfloat2_LC0_(sg_size, size, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)    \
    LOAD_DATA_Txx_SGxx_GCxx_LC0_H1_(float2, sg_size, GRP_CNT(size, sg_size), buffer, index,           \
                                    ptr, ptr_byte_offset, ptr_byte_limit)
#define LOAD_DATA_Tdouble2_LC0_(sg_size, size, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)   \
    LOAD_DATA_Txx_SGxx_GCxx_LC0_H1_(double2, sg_size, GRP_CNT(size, sg_size), buffer, index,          \
                                    ptr, ptr_byte_offset, ptr_byte_limit)


#define LOAD_DATA_H1_(type, sg_size, size, buffer, index, ptr, ptr_byte_offset)       \
    LOAD_DATA_T##type##_LC0_(sg_size, size, buffer, index, ptr, ptr_byte_offset, 0)
#define LOAD_DATA(type, sg_size, size, buffer, index, ptr, ptr_byte_offset)           \
    LOAD_DATA_H1_(type, sg_size, size, buffer, index, ptr, ptr_byte_offset)

#define LOAD_DATA_WITH_LIMIT_H1_(type, sg_size, size, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)   \
    LOAD_DATA_T##type##_LC1_(sg_size, size, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)
#define LOAD_DATA_WITH_LIMIT(type, sg_size, size, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)       \
    LOAD_DATA_WITH_LIMIT_H1_(type, sg_size, size, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)

// =====================================================================================================================
// =====================================================================================================================

// =====================================================================================================================
// Storing data:
// =====================================================================================================================
//
// STORE_DATA            - Stores continous block of data of specified type and number of elements. Does not check for
//                         limits. Can use different algorithms to store data depending on store size, type, etc.
// STORE_DATA_WITH_LIMIT - Stores continous block of data of specified type and number of elements. Provides limit which
//                         restricts writing up to specified offset value. Any writes outside allowed range:
//                         [ptr, ptr + ptr_byte_limit) will be discarded.
//                         Can use different algorithms to store data depending on store size, type, etc.
//
// STORE_DATA(type, sg_size, size, align, buffer, index, ptr, ptr_byte_offset)
// STORE_DATA_WITH_LIMIT(type, sg_size, size, align, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)
// 
//
// Customization:
//  DATA_LS_RW_AS - Name of address space used as read-write / write-only address space (store address space).
//                  Default: __global.
//
// Parameters:
//  - type            - [PP] Type of element in source "buffer" from which data will be stored into "ptr".
//  - sg_size         - [PP] Selected sub-group size or SIMD size. Allowed values: 8, 16, 32.
//  - size            - [PP] Number of elements to store. Must be multiple of "sg_size" and currently limited to 256.
//  - align           - [PP] Ensured / Provided alignment of "ptr". The value provided is treated as muliply of
//                      element size, e.g. 1 means sizeof(type), 2 means 2 * sizeof(type), etc.
//                      Must be positive and power of 2 and it is limited to 128.
//  - buffer          - [UNIFORM] An rvalue expression of array of type elements. Please use expression without
//                      side-effects (it will be expanded possibly more than once). Source buffer.
//  - index           - [UNIFORM] (0-based) Start position where data should be loaded/read from "buffer".
//  - ptr             - [UNIFORM] Pointer convertible to "DATA_LS_RW_AS void*" that represents
//                      output place to store data.
//  - ptr_byte_offset - [UNIFORM] A byte offset (unsigned integer) from start of output place pointed by "ptr".
//                      It will be scattered if necessary. The data is stored into "ptr" offset by "ptr_byte_offset".
//  - ptr_byte_limit  - [UNIFORM] Exclusive limit in bytes to which maximum write to "ptr" can take place. Any element
//                      which offset is outside [ptr, ptr + ptr_byte_limit) range will not be stored
//                      (will be discarded).
//
// LUT mangled params:
//  - T<type>
//  - SG<sg_size>
//  - GC<grp_cnt> - [PP] Number of scalar write operations needed to write "size" elements.
//  - EA<align>
//  - LC<0|1>     - [PP] Indicates that limit check is enabled. Allowed values: 0, 1.
//
// =====================================================================================================================

#ifndef DATA_LS_RW_AS
    #define DATA_LS_RW_AS   __global
#endif

#define STORE_DATA_Txx_SGxx_GCxx_EA1_LC0_H1_(type, sg_size, grp_cnt, buffer, index,                    \
                                             ptr, ptr_byte_offset, ptr_byte_limit)                     \
do {                                                                                                   \
    const uint STORE_DATA_tmp_idx_   = (index);                                                        \
    uint STORE_DATA_tmp_byte_offset_ = SCATTER_TYPE_OFFSET_BY_LID(S, type, ptr_byte_offset);           \
                                                                                                       \
    __attribute__((opencl_unroll_hint(grp_cnt)))                                                       \
    for (uint STORE_DATA_tmp_li_ = 0; STORE_DATA_tmp_li_ < (grp_cnt); ++STORE_DATA_tmp_li_)            \
    {                                                                                                  \
        *(DATA_LS_RW_AS type*)((DATA_LS_RW_AS char*)(ptr) + STORE_DATA_tmp_byte_offset_) =             \
            (buffer)[STORE_DATA_tmp_idx_ + STORE_DATA_tmp_li_];                                        \
        STORE_DATA_tmp_byte_offset_ = SHIFT_TYPE_OFFSET(type, STORE_DATA_tmp_byte_offset_, sg_size);   \
    }                                                                                                  \
} while (false)

#define STORE_DATA_Tfloat_SGxx_GCxx_EA4_LC0_H2_(sg_size, grp_cnt, buffer, index,                                       \
                                                ptr, ptr_byte_offset, ptr_byte_limit)                                  \
do {                                                                                                                   \
    const uint STORE_DATA_tmp_idx_   = (index);                                                                        \
    uint STORE_DATA_tmp_byte_offset_ = SCATTER_TYPE_OFFSET_BY_LID(B, float, ptr_byte_offset);                          \
    const uint STORE_DATA_tmp_lc_    = (grp_cnt);                                                                      \
    uint STORE_DATA_tmp_li_          = 0;                                                                              \
                                                                                                                       \
    __attribute__((opencl_unroll_hint))                                                                                \
    for (; STORE_DATA_tmp_li_ + 7 < STORE_DATA_tmp_lc_; STORE_DATA_tmp_li_ += 8)                                       \
    {                                                                                                                  \
        intel_sub_group_block_write8((DATA_LS_RW_AS uint*)((DATA_LS_RW_AS char*)(ptr) + STORE_DATA_tmp_byte_offset_),  \
            as_uint8((float8)(                                                                                         \
                (buffer)[STORE_DATA_tmp_idx_ + STORE_DATA_tmp_li_],                                                    \
                (buffer)[STORE_DATA_tmp_idx_ + STORE_DATA_tmp_li_ + 1],                                                \
                (buffer)[STORE_DATA_tmp_idx_ + STORE_DATA_tmp_li_ + 2],                                                \
                (buffer)[STORE_DATA_tmp_idx_ + STORE_DATA_tmp_li_ + 3],                                                \
                (buffer)[STORE_DATA_tmp_idx_ + STORE_DATA_tmp_li_ + 4],                                                \
                (buffer)[STORE_DATA_tmp_idx_ + STORE_DATA_tmp_li_ + 5],                                                \
                (buffer)[STORE_DATA_tmp_idx_ + STORE_DATA_tmp_li_ + 6],                                                \
                (buffer)[STORE_DATA_tmp_idx_ + STORE_DATA_tmp_li_ + 7]                                                 \
            )));                                                                                                       \
        STORE_DATA_tmp_byte_offset_ = SHIFT_TYPE_OFFSET(float, STORE_DATA_tmp_byte_offset_, 8 * sg_size);              \
    }                                                                                                                  \
                                                                                                                       \
    __attribute__((opencl_unroll_hint))                                                                                \
    for (; STORE_DATA_tmp_li_ + 3 < STORE_DATA_tmp_lc_; STORE_DATA_tmp_li_ += 4)                                       \
    {                                                                                                                  \
        intel_sub_group_block_write4((DATA_LS_RW_AS uint*)((DATA_LS_RW_AS char*)(ptr) + STORE_DATA_tmp_byte_offset_),  \
            as_uint4((float4)(                                                                                         \
                (buffer)[STORE_DATA_tmp_idx_ + STORE_DATA_tmp_li_],                                                    \
                (buffer)[STORE_DATA_tmp_idx_ + STORE_DATA_tmp_li_ + 1],                                                \
                (buffer)[STORE_DATA_tmp_idx_ + STORE_DATA_tmp_li_ + 2],                                                \
                (buffer)[STORE_DATA_tmp_idx_ + STORE_DATA_tmp_li_ + 3]                                                 \
            )));                                                                                                       \
        STORE_DATA_tmp_byte_offset_ = SHIFT_TYPE_OFFSET(float, STORE_DATA_tmp_byte_offset_, 4 * sg_size);              \
    }                                                                                                                  \
                                                                                                                       \
    __attribute__((opencl_unroll_hint))                                                                                \
    for (; STORE_DATA_tmp_li_ + 1 < STORE_DATA_tmp_lc_; STORE_DATA_tmp_li_ += 2)                                       \
    {                                                                                                                  \
        intel_sub_group_block_write2((DATA_LS_RW_AS uint*)((DATA_LS_RW_AS char*)(ptr) + STORE_DATA_tmp_byte_offset_),  \
            as_uint2((float2)(                                                                                         \
                (buffer)[STORE_DATA_tmp_idx_ + STORE_DATA_tmp_li_],                                                    \
                (buffer)[STORE_DATA_tmp_idx_ + STORE_DATA_tmp_li_ + 1]                                                 \
            )));                                                                                                       \
        STORE_DATA_tmp_byte_offset_ = SHIFT_TYPE_OFFSET(float, STORE_DATA_tmp_byte_offset_, 2 * sg_size);              \
    }                                                                                                                  \
                                                                                                                       \
    __attribute__((opencl_unroll_hint))                                                                                \
    for (; STORE_DATA_tmp_li_ < STORE_DATA_tmp_lc_; ++STORE_DATA_tmp_li_)                                              \
    {                                                                                                                  \
        intel_sub_group_block_write((DATA_LS_RW_AS uint*)((DATA_LS_RW_AS char*)(ptr) + STORE_DATA_tmp_byte_offset_),   \
            as_uint((buffer)[STORE_DATA_tmp_idx_ + STORE_DATA_tmp_li_]));                                              \
        STORE_DATA_tmp_byte_offset_ = SHIFT_TYPE_OFFSET(float, STORE_DATA_tmp_byte_offset_, sg_size);                  \
    }                                                                                                                  \
} while (false)
    
#define STORE_DATA_Tfloat_SGxx_GCxx_EA1_LC0_H2_(sg_size, grp_cnt, buffer, index,                                       \
                                                ptr, ptr_byte_offset, ptr_byte_limit)                                  \
    STORE_DATA_Txx_SGxx_GCxx_EA1_LC0_H1_(float, sg_size, grp_cnt, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)

#define STORE_DATA_Txx_SGxx_GCxx_EA1_LC1_H1_(type, sg_size, grp_cnt, buffer, index,                    \
                                             ptr, ptr_byte_offset, ptr_byte_limit)                     \
do {                                                                                                   \
    const uint STORE_DATA_tmp_type_loff   = (uint) sizeof(type) - 1;                                   \
    const uint STORE_DATA_tmp_idx_        = (index);                                                   \
    uint STORE_DATA_tmp_byte_offset_      = SCATTER_TYPE_OFFSET_BY_LID(S, type, ptr_byte_offset);      \
    const uint STORE_DATA_tmp_byte_limit_ = (ptr_byte_limit);                                          \
                                                                                                       \
    __attribute__((opencl_unroll_hint))                                                                \
    for (uint STORE_DATA_tmp_li_ = 0; STORE_DATA_tmp_li_ < (grp_cnt); ++STORE_DATA_tmp_li_)            \
    {                                                                                                  \
        if (STORE_DATA_tmp_byte_offset_ + STORE_DATA_tmp_type_loff >= STORE_DATA_tmp_byte_limit_)      \
            break;                                                                                     \
                                                                                                       \
        *(DATA_LS_RW_AS type*)((DATA_LS_RW_AS char*)(ptr) + STORE_DATA_tmp_byte_offset_) =             \
            (buffer)[STORE_DATA_tmp_idx_ + STORE_DATA_tmp_li_];                                        \
        STORE_DATA_tmp_byte_offset_ = SHIFT_TYPE_OFFSET(type, STORE_DATA_tmp_byte_offset_, sg_size);   \
    }                                                                                                  \
} while (false)


#define STORE_DATA_Txx_SGxx_GCxx_EAxx_LC0_H11_(type, sg_size, grp_cnt, align, buffer, index,   \
                                               ptr, ptr_byte_offset, ptr_byte_limit)           \
    STORE_DATA_Txx_SGxx_GCxx_EA##align##_LC0_H1_(type, sg_size, grp_cnt, buffer, index,        \
                                                 ptr, ptr_byte_offset, ptr_byte_limit)
#define STORE_DATA_Txx_SGxx_GCxx_EAxx_LC0_H1_(type, sg_size, grp_cnt, align, buffer, index,    \
                                             ptr, ptr_byte_offset, ptr_byte_limit)             \
    STORE_DATA_Txx_SGxx_GCxx_EAxx_LC0_H11_(type, sg_size, grp_cnt, align, buffer, index,       \
                                           ptr, ptr_byte_offset, ptr_byte_limit)

#define STORE_DATA_Tfloat_SGxx_GCxx_EAxx_LC0_H21_(sg_size, grp_cnt, align, buffer, index,      \
                                                  ptr, ptr_byte_offset, ptr_byte_limit)        \
    STORE_DATA_Tfloat_SGxx_GCxx_EA##align##_LC0_H2_(sg_size, grp_cnt, buffer, index,           \
                                                    ptr, ptr_byte_offset, ptr_byte_limit)
#define STORE_DATA_Tfloat_SGxx_GCxx_EAxx_LC0_H2_(sg_size, grp_cnt, align, buffer, index,       \
                                                 ptr, ptr_byte_offset, ptr_byte_limit)         \
    STORE_DATA_Tfloat_SGxx_GCxx_EAxx_LC0_H21_(sg_size, grp_cnt, align, buffer, index,          \
                                              ptr, ptr_byte_offset, ptr_byte_limit)

#define STORE_DATA_Txx_SGxx_GCxx_EAxx_LC1_H11_(type, sg_size, grp_cnt, align, buffer, index,   \
                                               ptr, ptr_byte_offset, ptr_byte_limit)           \
    STORE_DATA_Txx_SGxx_GCxx_EA##align##_LC1_H1_(type, sg_size, grp_cnt, buffer, index,        \
                                                 ptr, ptr_byte_offset, ptr_byte_limit)
#define STORE_DATA_Txx_SGxx_GCxx_EAxx_LC1_H1_(type, sg_size, grp_cnt, align, buffer, index,    \
                                              ptr, ptr_byte_offset, ptr_byte_limit)            \
    STORE_DATA_Txx_SGxx_GCxx_EAxx_LC1_H11_(type, sg_size, grp_cnt, align, buffer, index,       \
                                              ptr, ptr_byte_offset, ptr_byte_limit)


#define STORE_DATA_Tfloat_LC1_(sg_size, size, align, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)     \
    STORE_DATA_Txx_SGxx_GCxx_EAxx_LC1_H1_(float, sg_size, GRP_CNT(size, sg_size), SEL_ALIGN(align, 1, 1),     \
                                          buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)
#define STORE_DATA_Tdouble_LC1_(sg_size, size, align, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)    \
    STORE_DATA_Txx_SGxx_GCxx_EAxx_LC1_H1_(double, sg_size, GRP_CNT(size, sg_size), SEL_ALIGN(align, 1, 1),    \
                                          buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)
#define STORE_DATA_Tfloat2_LC1_(sg_size, size, align, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)    \
    STORE_DATA_Txx_SGxx_GCxx_EAxx_LC1_H1_(float2, sg_size, GRP_CNT(size, sg_size), SEL_ALIGN(align, 1, 1),    \
                                          buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)
#define STORE_DATA_Tdouble2_LC1_(sg_size, size, align, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)   \
    STORE_DATA_Txx_SGxx_GCxx_EAxx_LC1_H1_(double2, sg_size, GRP_CNT(size, sg_size), SEL_ALIGN(align, 1, 1),   \
                                          buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)


#if SG_SUPPORTED
    #define STORE_DATA_Tfloat_LC0_(sg_size, size, align, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)   \
        STORE_DATA_Tfloat_SGxx_GCxx_EAxx_LC0_H2_(sg_size, GRP_CNT(size, sg_size), SEL_ALIGN(align, 4, 1),       \
                                                 buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)
#else
    #define STORE_DATA_Tfloat_LC0_(sg_size, size, align, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)   \
        STORE_DATA_Txx_SGxx_GCxx_EAxx_LC0_H1_(float, sg_size, GRP_CNT(size, sg_size), SEL_ALIGN(align, 1, 1),   \
                                              buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)
#endif

#define STORE_DATA_Tdouble_LC0_(sg_size, size, align, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)    \
    STORE_DATA_Txx_SGxx_GCxx_EAxx_LC0_H1_(double, sg_size, GRP_CNT(size, sg_size), SEL_ALIGN(align, 1, 1),    \
                                          buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)
#define STORE_DATA_Tfloat2_LC0_(sg_size, size, align, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)    \
    STORE_DATA_Txx_SGxx_GCxx_EAxx_LC0_H1_(float2, sg_size, GRP_CNT(size, sg_size), SEL_ALIGN(align, 1, 1),    \
                                          buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)
#define STORE_DATA_Tdouble2_LC0_(sg_size, size, align, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)   \
    STORE_DATA_Txx_SGxx_GCxx_EAxx_LC0_H1_(double2, sg_size, GRP_CNT(size, sg_size), SEL_ALIGN(align, 1, 1),   \
                                          buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)


#define STORE_DATA_H1_(type, sg_size, size, align, buffer, index, ptr, ptr_byte_offset)                            \
    STORE_DATA_T##type##_LC0_(sg_size, size, align, buffer, index, ptr, ptr_byte_offset, 0)
#define STORE_DATA(type, sg_size, size, align, buffer, index, ptr, ptr_byte_offset)                                \
    STORE_DATA_H1_(type, sg_size, size, align, buffer, index, ptr, ptr_byte_offset)

#define STORE_DATA_WITH_LIMIT_H1_(type, sg_size, size, align, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit) \
    STORE_DATA_T##type##_LC1_(sg_size, size, align, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)
#define STORE_DATA_WITH_LIMIT(type, sg_size, size, align, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)     \
    STORE_DATA_WITH_LIMIT_H1_(type, sg_size, size, align, buffer, index, ptr, ptr_byte_offset, ptr_byte_limit)

// =====================================================================================================================
// =====================================================================================================================

#ifdef DATA_LOAD_STORE_SG_SUPPORTED_UNDEF_MACRO_
    #undef DATA_LOAD_STORE_SG_SUPPORTED_UNDEF_MACRO_
    #undef SG_SUPPORTED
#endif

#endif // DATA_LOAD_STORE_H
