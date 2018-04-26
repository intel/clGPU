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
// File: pp_load_store_helpers.h.cl
//
// Common pre-processor helpers for load/store.

#ifndef PP_LS_HELPERS_H
#define PP_LS_HELPERS_H

// Environment:
#ifndef SG_SUPPORTED
    #define SG_SUPPORTED   1
    #define PP_LS_HELPERS_SG_SUPPORTED_UNDEF_MACRO_
#endif

// =====================================================================================================================
// Pre-processor helpers:
// =====================================================================================================================
//
// GRP_CNT   - Returns number of sub-groups needed to process specified number of elements. Returned number is single
//             pre-processed value ([PP] value).
//
// GRP_CNT(size, sg_size)
//
// Parameters:
//  - size    - [PP] Number of elements to process. Must be multiple of sg_size and currently limited to 256.
//  - sg_size - [PP] Selected sub-group size or SIMD size. Allowed values: 8, 16, 32.
//
// LUT mangled params:
//  - S<size>
//  - SG<sg_size>
//
// =====================================================================================================================

#define GRP_CNT_S8_SG8_       1
#define GRP_CNT_S16_SG8_      2
#define GRP_CNT_S16_SG16_     1
#define GRP_CNT_S24_SG8_      3
#define GRP_CNT_S32_SG8_      4
#define GRP_CNT_S32_SG16_     2
#define GRP_CNT_S32_SG32_     1
#define GRP_CNT_S40_SG8_      5
#define GRP_CNT_S48_SG8_      6
#define GRP_CNT_S48_SG16_     3
#define GRP_CNT_S56_SG8_      7
#define GRP_CNT_S64_SG8_      8
#define GRP_CNT_S64_SG16_     4
#define GRP_CNT_S64_SG32_     2
#define GRP_CNT_S72_SG8_      9
#define GRP_CNT_S80_SG8_     10
#define GRP_CNT_S80_SG16_     5
#define GRP_CNT_S88_SG8_     11
#define GRP_CNT_S96_SG8_     12
#define GRP_CNT_S96_SG16_     6
#define GRP_CNT_S96_SG32_     3
#define GRP_CNT_S104_SG8_    13
#define GRP_CNT_S112_SG8_    14
#define GRP_CNT_S112_SG16_    7
#define GRP_CNT_S120_SG8_    15
#define GRP_CNT_S128_SG8_    16
#define GRP_CNT_S128_SG16_    8
#define GRP_CNT_S128_SG32_    4
#define GRP_CNT_S136_SG8_    17
#define GRP_CNT_S144_SG8_    18
#define GRP_CNT_S144_SG16_    9
#define GRP_CNT_S152_SG8_    19
#define GRP_CNT_S160_SG8_    20
#define GRP_CNT_S160_SG16_   10
#define GRP_CNT_S160_SG32_    5
#define GRP_CNT_S168_SG8_    21
#define GRP_CNT_S176_SG8_    22
#define GRP_CNT_S176_SG16_   11
#define GRP_CNT_S184_SG8_    23
#define GRP_CNT_S192_SG8_    24
#define GRP_CNT_S192_SG16_   12
#define GRP_CNT_S192_SG32_    6
#define GRP_CNT_S200_SG8_    25
#define GRP_CNT_S208_SG8_    26
#define GRP_CNT_S208_SG16_   13
#define GRP_CNT_S216_SG8_    27
#define GRP_CNT_S224_SG8_    28
#define GRP_CNT_S224_SG16_   14
#define GRP_CNT_S224_SG32_    7
#define GRP_CNT_S232_SG8_    29
#define GRP_CNT_S240_SG8_    30
#define GRP_CNT_S240_SG16_   15
#define GRP_CNT_S248_SG8_    31
#define GRP_CNT_S256_SG8_    32
#define GRP_CNT_S256_SG16_   16
#define GRP_CNT_S256_SG32_    8


#define GRP_CNT_H1_(size, sg_size)   GRP_CNT_S##size##_SG##sg_size##_
#define GRP_CNT(size, sg_size)       GRP_CNT_H1_(size, sg_size)

// =====================================================================================================================
//
// SEL_ALIGN - Selects compatible alignment. Selects candidate alignment if provided alignment "align" is greater-equal
//             than "candidate" ("candidate1", "candidate2") alignment. Otherwise, returns "fallback" alignment.
//             Returned number is [PP]-value.
//
// SEL_ALIGN(align, candidate, fallback)
// SEL_ALIGN2(align, candidate1, candidate2, fallback)
//
// Parameters:
//  - align      - [PP] Provided alignment (alignment of data accesses).
//  - candidate  - [PP] Candidate alignment for selection.
//  - candidate1 - [PP] First candidate alignment for selection. Must be greater than candidate2.
//  - candidate2 - [PP] Second candidate alignment for selection. Must be less than candidate1.
//  - fallback   - [PP] Alignment used if candidate alignment cannot be selected (too high). 
//
// LUT mangled params:
//  - A<align>
//  - CA<candidate>
//
// =====================================================================================================================

#define SEL_ALIGN_A1_CA1_(selection, fallback)     selection
#define SEL_ALIGN_A2_CA1_(selection, fallback)     selection
#define SEL_ALIGN_A4_CA1_(selection, fallback)     selection
#define SEL_ALIGN_A8_CA1_(selection, fallback)     selection
#define SEL_ALIGN_A16_CA1_(selection, fallback)    selection
#define SEL_ALIGN_A32_CA1_(selection, fallback)    selection
#define SEL_ALIGN_A64_CA1_(selection, fallback)    selection
#define SEL_ALIGN_A128_CA1_(selection, fallback)   selection

#define SEL_ALIGN_A1_CA2_(selection, fallback)     fallback
#define SEL_ALIGN_A2_CA2_(selection, fallback)     selection
#define SEL_ALIGN_A4_CA2_(selection, fallback)     selection
#define SEL_ALIGN_A8_CA2_(selection, fallback)     selection
#define SEL_ALIGN_A16_CA2_(selection, fallback)    selection
#define SEL_ALIGN_A32_CA2_(selection, fallback)    selection
#define SEL_ALIGN_A64_CA2_(selection, fallback)    selection
#define SEL_ALIGN_A128_CA2_(selection, fallback)   selection

#define SEL_ALIGN_A1_CA4_(selection, fallback)     fallback
#define SEL_ALIGN_A2_CA4_(selection, fallback)     fallback
#define SEL_ALIGN_A4_CA4_(selection, fallback)     selection
#define SEL_ALIGN_A8_CA4_(selection, fallback)     selection
#define SEL_ALIGN_A16_CA4_(selection, fallback)    selection
#define SEL_ALIGN_A32_CA4_(selection, fallback)    selection
#define SEL_ALIGN_A64_CA4_(selection, fallback)    selection
#define SEL_ALIGN_A128_CA4_(selection, fallback)   selection

#define SEL_ALIGN_A1_CA8_(selection, fallback)     fallback
#define SEL_ALIGN_A2_CA8_(selection, fallback)     fallback
#define SEL_ALIGN_A4_CA8_(selection, fallback)     fallback
#define SEL_ALIGN_A8_CA8_(selection, fallback)     selection
#define SEL_ALIGN_A16_CA8_(selection, fallback)    selection
#define SEL_ALIGN_A32_CA8_(selection, fallback)    selection
#define SEL_ALIGN_A64_CA8_(selection, fallback)    selection
#define SEL_ALIGN_A128_CA8_(selection, fallback)   selection

#define SEL_ALIGN_A1_CA16_(selection, fallback)     fallback
#define SEL_ALIGN_A2_CA16_(selection, fallback)     fallback
#define SEL_ALIGN_A4_CA16_(selection, fallback)     fallback
#define SEL_ALIGN_A8_CA16_(selection, fallback)     fallback
#define SEL_ALIGN_A16_CA16_(selection, fallback)    selection
#define SEL_ALIGN_A32_CA16_(selection, fallback)    selection
#define SEL_ALIGN_A64_CA16_(selection, fallback)    selection
#define SEL_ALIGN_A128_CA16_(selection, fallback)   selection

#define SEL_ALIGN_A1_CA32_(selection, fallback)     fallback
#define SEL_ALIGN_A2_CA32_(selection, fallback)     fallback
#define SEL_ALIGN_A4_CA32_(selection, fallback)     fallback
#define SEL_ALIGN_A8_CA32_(selection, fallback)     fallback
#define SEL_ALIGN_A16_CA32_(selection, fallback)    fallback
#define SEL_ALIGN_A32_CA32_(selection, fallback)    selection
#define SEL_ALIGN_A64_CA32_(selection, fallback)    selection
#define SEL_ALIGN_A128_CA32_(selection, fallback)   selection

#define SEL_ALIGN_A1_CA64_(selection, fallback)     fallback
#define SEL_ALIGN_A2_CA64_(selection, fallback)     fallback
#define SEL_ALIGN_A4_CA64_(selection, fallback)     fallback
#define SEL_ALIGN_A8_CA64_(selection, fallback)     fallback
#define SEL_ALIGN_A16_CA64_(selection, fallback)    fallback
#define SEL_ALIGN_A32_CA64_(selection, fallback)    fallback
#define SEL_ALIGN_A64_CA64_(selection, fallback)    selection
#define SEL_ALIGN_A128_CA64_(selection, fallback)   selection

#define SEL_ALIGN_A1_CA128_(selection, fallback)     fallback
#define SEL_ALIGN_A2_CA128_(selection, fallback)     fallback
#define SEL_ALIGN_A4_CA128_(selection, fallback)     fallback
#define SEL_ALIGN_A8_CA128_(selection, fallback)     fallback
#define SEL_ALIGN_A16_CA128_(selection, fallback)    fallback
#define SEL_ALIGN_A32_CA128_(selection, fallback)    fallback
#define SEL_ALIGN_A64_CA128_(selection, fallback)    fallback
#define SEL_ALIGN_A128_CA128_(selection, fallback)   selection


#define SEL_ALIGN_H1_(align, candidate, selection, fallback)    \
    SEL_ALIGN_A##align##_CA##candidate##_(selection, fallback)
#define SEL_ALIGN(align, candidate, fallback)                   \
    SEL_ALIGN_H1_(align, candidate, candidate, fallback)

#define SEL_ALIGN2(align, candidate1, candidate2, fallback)                                                \
    SEL_ALIGN_H1_(align, candidate2, SEL_ALIGN_H1_(align, candidate1, candidate1, candidate2), fallback)

// =====================================================================================================================
// =====================================================================================================================

// =====================================================================================================================
// Offset operations:
// =====================================================================================================================
//
// SHIFT_OFFSET      - Shifts byte base offset ("base_offset") by specified uniform value ("offset_shift") of bytes.
// SHIFT_TYPE_OFFSET - Shifts byte base offset ("base_offset") by specified uniform value ("offset_shift") of elements
//                     of specified "type" (assumed that size and alignment of element are equal).
//
// SHIFT_OFFSET(base_offset, offset_shift)
// SHIFT_TYPE_OFFSET(type, base_offset, offset_shift)
//
// Parameters:
//  - type             - [PP] Type of element. Its size (as signed integer) is used to scale applied shift value.
//  - base_offset      - [UNIFORM] A base byte offset.
//  - offset_shift     - [UNIFORM] A value (non-negative / unsigned recommended, but it can be negative) by which
//                       "base_offset" will be shifted (in bytes or in elements of type "type").
//
// =====================================================================================================================

#define SHIFT_OFFSET(base_offset, offset_shift)   ((base_offset) + (offset_shift))
#define SHIFT_TYPE_OFFSET(type, base_offset, offset_shift)             \
    SHIFT_OFFSET(base_offset, ((int) sizeof(type)) * (offset_shift))

// =====================================================================================================================
//
// SCATTER_OFFSET             - Scatters byte base offset ("base_offset") if necessary (shifts non-uniformly by
//                              specified "offset_scatter" value of bytes).
// SCATTER_TYPE_OFFSET        - Scatters byte base offset ("base_offset") if necessary (shifts non-uniformly by
//                              specified "offset_scatter" value of elements of specified "type"
//                              (assumed that size and alignment of element are equal)).
// SCATTER_TYPE_OFFSET_BY_LID - Scatters byte base offset ("base_offset") if necessary by get_sub_group_local_id()
//                              (if sub-groups are supported) or, otherwise, by get_local_id(0). Non-uniform local
//                              idenfier is casted to uint and scaled by the size of / alignment of "type"
//                              (non-uniform scatter is in elements of "type"). To assign different non-uniform local
//                              identifier, please redefine PP_LS_HELPERS_SCATTER_LID_EXP.
//
// SCATTER_OFFSET(mode, base_offset, offset_scatter)
// SCATTER_TYPE_OFFSET(mode, type, base_offset, offset_scatter)
// SCATTER_TYPE_OFFSET_BY_LID(mode, type, base_offset)
//
// Customization:
//  PP_LS_HELPERS_SCATTER_LID_EXP - Expression used as local identifier (for user customization).
//
// Environment:
//  SG_SUPPORTED - Indicates that sub-groups are supported (default: 1). Allowed values: 0, 1.
//
// Parameters:
//  - mode             - [PP] Mode of scatter offset operation (intended purpose of offset). Allowed values: B, S, U.
//                        * B - Block reads / writes (if sub-groups are supported). Falls back to scattered mode,
//                              if block mode is not available.
//                        * S - Scattered reads / writes. Default mode in OpenCL C.
//                        * U - Work-group uniform reads / writes (based on uniforms optimization).
//  - type             - [PP] Type of element. Its size (as signed integer) is used to scale applied scatter value.
//  - base_offset      - [UNIFORM] A base byte offset.
//  - offset_scatter   - [NON-UNIFORM] A value (non-negative / unsigned recommended, but can be negative) by which
//                       "base_offset" will be shifted non-uniformly (scattered).
//                       Scatter offset is not applied if sub-groups are supported and block mode is used (B mode),
//                       or work-group uniform mode is used (U mode).
//
// LUT mangled params:
//  - M<mode>
//
// =====================================================================================================================

#define SCATTER_OFFSET_MS_(base_offset, offset_scatter)   ((base_offset) + (offset_scatter))
#define SCATTER_OFFSET_MU_(base_offset, offset_scatter)   (base_offset)

#if SG_SUPPORTED
    #define SCATTER_OFFSET_MB_(base_offset, offset_scatter)   SCATTER_OFFSET_MU_(base_offset, offset_scatter)

    #ifndef PP_LS_HELPERS_SCATTER_LID_EXP
        #define PP_LS_HELPERS_SCATTER_LID_EXP                 get_sub_group_local_id()
    #endif
#else
    #define SCATTER_OFFSET_MB_(base_offset, offset_scatter)   SCATTER_OFFSET_MS_(base_offset, offset_scatter)

    #ifndef PP_LS_HELPERS_SCATTER_LID_EXP
        #define PP_LS_HELPERS_SCATTER_LID_EXP                 get_local_id(0)
    #endif
#endif


#define SCATTER_OFFSET_H1_(mode, base_offset, offset_scatter)   \
    SCATTER_OFFSET_M##mode##_(base_offset, offset_scatter)
#define SCATTER_OFFSET(mode, base_offset, offset_scatter)       \
    SCATTER_OFFSET_H1_(mode, base_offset, offset_scatter)

#define SCATTER_TYPE_OFFSET(mode, type, base_offset, offset_scatter)             \
    SCATTER_OFFSET(mode, base_offset, ((int) sizeof(type)) * (offset_scatter))

#define SCATTER_TYPE_OFFSET_BY_LID(mode, type, base_offset)                                \
    SCATTER_TYPE_OFFSET(mode, type, base_offset, (uint) (PP_LS_HELPERS_SCATTER_LID_EXP))

// =====================================================================================================================
// =====================================================================================================================

#ifdef PP_LS_HELPERS_SG_SUPPORTED_UNDEF_MACRO_
    #undef PP_LS_HELPERS_SG_SUPPORTED_UNDEF_MACRO_
    #undef SG_SUPPORTED
#endif

#endif // PP_LS_HELPERS_H
