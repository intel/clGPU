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
// File: gemm_n3_template.h.cl


// =====================================================================================================================
// =====================================================================================================================
//                                                    KERNEL TEMPLATE
// =====================================================================================================================
// =====================================================================================================================
//
// Kernel that performs BLAS-compatible GEMM (generic matrix multiply).
// The kernel does use parallelized version of standard algorithm (full dot product of row and column approach)
// of (non-parallel) complexity THETA(n*m*k) - in parallel version: THETA(n*m*k/p), assuming that p << n*m*k
// where p is number of processors in PRAM (<< means in this case a lot lower).
//
// Kernel cuts matrices into tiles in which it does fast parallel dot-product (each output tile is calculated by
// TARG_SG_SIZE work-items grouped in single sub-group). Calculation for mutliple output tiles are also done
// in parallel.
// Currenly tiles are as folows:
//  - Matrix A: TARG_TILE_M  rows x TARG_TILE_AK columns
//  - Matrix B: TARG_TILE_BK rows x TARG_TILE_N  columns
//  - Matrix C: TARG_TILE_M  rows x TARG_TILE_N  columns
//
// Template JIT Constants (PP Parameters):
//  - TARG_KERNEL_NAME          [id]   (default: sample_sgemm_AN_BN) Name of the kernel.
//  - TARG_SG_SIZE              [unit] (default: 16) Size of sub-group or SIMD used in kernel.
//                                     Allowed values: 8, 16, 32.
//  - TARG_DATA_TYPE            [type] (default: float) Type of element stored in matrices A, B and C.
//                                     For complex types please use OpenCL 2-component vectors (e.g. float2, double2).
//  - TARG_MATRIX_FMT_A         [enum] (default: C) Matrix A strorage format. Allowed values: C, R, HR.
//                                       * C  - column-major storage.
//                                       * R  - row-major storage (matrix transposed).
//                                       * HR - row-major storage, elements treated as conjugate
//                                              (matrix Hermitian-transposed).
//  - TARG_MATRIX_FMT_B         [enum] (default: C) Matrix B strorage format. Allowed values: C, R, HR.
//                                       * C  - column-major storage.
//                                       * R  - row-major storage (matrix transposed).
//                                       * HR - row-major storage, elements treated as conjugate
//                                              (matrix Hermitian-transposed).
//  - TARG_TILE_M               [uint] (default: 16) Tile size in m dimension. Must be positive.
//  - TARG_TILE_N               [uint] (default: 8)  Tile size in n dimension. Must be positive.
//  - TARG_TILE_AK              [uint] (default: 16) Tile size in k dimension (tile of matrix A). Must be positive.
//                                     TARG_TILE_AK must be dividable by TARG_TILE_BK,
//                                     or TARG_TILE_BK must be dividable
//                                     by TARG_TILE_AK.
//  - TARG_TILE_BK              [uint] (default: 16) Tile size in k dimension (tile of matrix B). Must be positive.
//                                     TARG_TILE_AK must be dividable by TARG_TILE_BK,
//                                     or TARG_TILE_BK must be dividable
//                                     by TARG_TILE_AK.
//  - TARG_TILE_IDX_GDIM_M      [uint] (default: 0) Number of dimension in get_group_id() which identifies index of
//                                     tile in m dimension.
//  - TARG_TILE_IDX_GDIM_N      [uint] (default: 1) Number of dimension in get_group_id() which identifies index of
//                                     tile in m dimension.
//  - TARG_DT_OP_NZT(x)         [expr] (default: ((x) != 0) ) Expression that tests TARG_DATA_TYPE to be non-zero.
//  - TARG_DT_OP_MUL(x, y)      [expr] (default: ((x) * (y)) ) Expression that multiplies two TARG_DATA_TYPE elements.
//  - TARG_DT_OP_ADD(x, y)      [expr] (default: ((x) + (y)) ) Expression that adds two TARG_DATA_TYPE elements.
//  - TARG_DT_OP_MAD(x, y, z)   [expr] (default: fma(x, y, z) ) Expression that muliply-add two TARG_DATA_TYPE elements.
//
// =====================================================================================================================

#ifndef GEMM_N3_TEMPLATE_H_DECLS
#define GEMM_N3_TEMPLATE_H_DECLS

// Environment:
#ifndef SG_SUPPORTED
    #define SG_SUPPORTED 1
    #define GEMM_N3_TEMPLATE_SG_SUPPORTED_UNDEF_MACRO_
#endif

// Includes:
#include <pp_load_store_helpers.h>
#include <tile_load_store.h>

// =====================================================================================================================
// Helpers:
// =====================================================================================================================
//
// GET_MATRIX_LS_FMT - Returns matrix load/store format (stripts element representation info).
//
//
//
// Parameters:
//  - matrix_fmt - [PP] Storage and representation format of the matrix. Allowed values: C, R, HR.
//
// LUT mangled params:
//  - MF<matrix_fmt>
// =====================================================================================================================

#define GET_MATRIX_LS_FMT_MFC_  C
#define GET_MATRIX_LS_FMT_MFR_  R
#define GET_MATRIX_LS_FMT_MFHR_ R


#define GET_MATRIX_LS_FMT_H1_(matrix_fmt)   GET_MATRIX_LS_FMT_MF##matrix_fmt##_
#define GET_MATRIX_LS_FMT(matrix_fmt)       GET_MATRIX_LS_FMT_H1_(matrix_fmt)

// =====================================================================================================================
// =====================================================================================================================

// =====================================================================================================================
// Tile multiply-add:
// =====================================================================================================================

// MAD_TILES - Performs multiplication of tile A with tile B. The result is added to current value of tile C.
//
// MAD_TILES(type, fmt_a, fmt_b, fmt_c, sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,
//           tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)
//
//
// Parameters:
//  - type             - [PP] Type of element in matrix tiles (in tile buffers "tile_buf_a", "tile_buf_b",
//                       "tile_buf_c").
//  - fmt_a            - [PP] Storage and representation format of the matrix tile.
//    fmt_b              Currently allowed values for "fmt_a" and "fmt_b": C, R, HR.
//    fmt_c              Currently allowed values for "fmt_c": C.
//                        * C  - matrix tile is in column-major format (row elements stored across sub-group and
//                               consecutive indices of array; columns stored in groups of indices of array - group size
//                               is equal to GRP_CNT(<tile rows>, sg_size)).
//                        * R  - matrix tile is in row-major format (column elements stored across sub-group and
//                               consecutive indices of array; rows stored in groups of indices of array - group size
//                               is equal to GRP_CNT(<tile columns>, sg_size)).
//                        * HR - matrix tile is in row-major format (as R). Each element of tile is treated as
//                               conjugate of actual value stored in matrix.
//  - sg_size          - [PP] Selected sub-group size or SIMD size. Allowed values: 8, 16, 32.
//  - tile_m           - [PP] Number of rows in m dimesion (number of rows in tile A and tile C).
//  - tile_n           - [PP] Number of columns in n dimension (number of columns in tile B and tile C).
//  - tile_ak          - [PP] Number of columns in k dimesion (number of columns in tile A).
//                       "tile_ak" must be dividable by "tile_bk" or "tile_bk" must be dividable by "tile_ak".
//  - tile_bk          - [PP] Number of rows in k dimesion (number of rows in tile B).
//                       "tile_ak" must be dividable by "tile_bk" or "tile_bk" must be dividable by "tile_ak".
//  - k_part           - [UNIFORM] If "tile_ak" is different than "tile_bk", 0-based index of part of greater tile
//                       in k dimension that should be multiplied with smaller tile in k dimension.
//                       Greater tile in k dimension is a tile with larger tile_k (tile_ak, tile_bk). Remainder tile is
//                       called smaller tile.
//  - tile_buf_a       - [UNIFORM] An rvalue expression of array of type elements. Please use expression without
//    tile_buf_b         side-effects (it will be expanded possibly more than once). Source tile buffers that will
//                       used to perform multiply-add (tile A and B).
//  - tile_buf_c       - [UNIFORM] An lvalue expression of array of type elements. Please use expression without
//                       side-effects (it will be expanded possibly more than once). Source tile buffer that will
//                       used to perform multiply-add (tile C) and destination buffer for MAD results.
//  - idx_a            - [UNIFORM] (0-based) Start position where data should be read from/written to tile buffers
//    idx_b              during calculation of results.
//    idx_c
//
// LUT mangled params:
//  - T<type>
//  - SG<sg_size>
//  - N<tile_n>
//  - M<tile_m>
//  - AK<tile_ak>
//  - BK<tile_bk>
//  - ATF<fmt_a>
//  - BTF<fmt_b>
//  - CTF<fmt_c>
//
// =====================================================================================================================

#if !SG_SUPPORTED
    #error Tile multiply-add currently requires sub-group extension.
#endif


#define MAD_TILES_Txx_SGxx_Mxx_Nxx_AKxx_BKxx_ATFC_BTFC_CTFC_H1_(                                                \
        type, type_op_mad, sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,                                   \
        tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)                                                \
do {                                                                                                            \
    const uint MAD_TILES_tile_k_min_  = (tile_ak) < (tile_bk) ? (tile_ak) : (tile_bk);                          \
    const uint MAD_TILES_tile_k_part_ = (k_part);                                                               \
    const uint MAD_TILES_idx_a_       = (idx_a);                                                                \
    const uint MAD_TILES_idx_b_       = (idx_b);                                                                \
    const uint MAD_TILES_idx_c_       = (idx_c);                                                                \
    uint MAD_TILES_aki_ = (tile_ak) > (tile_bk) ? MAD_TILES_tile_k_part_ * MAD_TILES_tile_k_min_ : 0;           \
    uint MAD_TILES_bki_ = (tile_bk) > (tile_ak) ? MAD_TILES_tile_k_part_ * MAD_TILES_tile_k_min_ : 0;           \
                                                                                                                \
    __attribute__((opencl_unroll_hint))                                                                         \
    for (uint MAD_TILES_ki_ = 0; MAD_TILES_ki_ < MAD_TILES_tile_k_min_; ++MAD_TILES_ki_)                        \
    {                                                                                                           \
        __attribute__((opencl_unroll_hint(tile_n)))                                                             \
        for (uint MAD_TILES_ni_ = 0; MAD_TILES_ni_ < (tile_n); ++MAD_TILES_ni_)                                 \
        {                                                                                                       \
            type MAD_TILES_elem_ = intel_sub_group_shuffle(                                                     \
                (tile_buf_b)[MAD_TILES_ni_ * GRP_CNT(tile_bk, sg_size) +                                        \
                             MAD_TILES_bki_ / (sg_size) + MAD_TILES_idx_b_],                                    \
                MAD_TILES_bki_ % (sg_size));                                                                    \
                                                                                                                \
            __attribute__((opencl_unroll_hint(GRP_CNT(tile_m, sg_size))))                                       \
            for (uint MAD_TILES_mgi_ = 0; MAD_TILES_mgi_ < GRP_CNT(tile_m, sg_size); ++MAD_TILES_mgi_)          \
            {                                                                                                   \
                (tile_buf_c)[MAD_TILES_ni_ * GRP_CNT(tile_m, sg_size) + MAD_TILES_mgi_ + MAD_TILES_idx_c_] =    \
                    type_op_mad(                                                                                \
                        MAD_TILES_elem_,                                                                        \
                        (tile_buf_a)[MAD_TILES_aki_ * GRP_CNT(tile_m, sg_size) +                                \
                                     MAD_TILES_mgi_ + MAD_TILES_idx_a_],                                        \
                        (tile_buf_c)[MAD_TILES_ni_ * GRP_CNT(tile_m, sg_size) +                                 \
                                     MAD_TILES_mgi_ + MAD_TILES_idx_c_]                                         \
                    );                                                                                          \
            }                                                                                                   \
        }                                                                                                       \
        ++MAD_TILES_aki_; ++MAD_TILES_bki_;                                                                     \
    }                                                                                                           \
} while (false)

#define MAD_TILES_Txx_SGxx_Mxx_Nxx_AKxx_BKxx_ATFR_BTFC_CTFC_H1_(                                                \
        type, type_op_mad, sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,                                   \
        tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)                                                \
do {                                                                                                            \
    const uint MAD_TILES_tile_k_min_  = (tile_ak) < (tile_bk) ? (tile_ak) : (tile_bk);                          \
    const uint MAD_TILES_tile_k_part_ = (k_part);                                                               \
    const uint MAD_TILES_idx_a_       = (idx_a);                                                                \
    const uint MAD_TILES_idx_b_       = (idx_b);                                                                \
    const uint MAD_TILES_idx_c_       = (idx_c);                                                                \
    uint MAD_TILES_aki_ = (tile_ak) > (tile_bk) ? MAD_TILES_tile_k_part_ * MAD_TILES_tile_k_min_ : 0;           \
    uint MAD_TILES_bki_ = (tile_bk) > (tile_ak) ? MAD_TILES_tile_k_part_ * MAD_TILES_tile_k_min_ : 0;           \
                                                                                                                \
    __attribute__((opencl_unroll_hint))                                                                         \
    for (uint MAD_TILES_ki_ = 0; MAD_TILES_ki_ < MAD_TILES_tile_k_min_; ++MAD_TILES_ki_)                        \
    {                                                                                                           \
        __attribute__((opencl_unroll_hint(tile_n)))                                                             \
        for (uint MAD_TILES_ni_ = 0; MAD_TILES_ni_ < (tile_n); ++MAD_TILES_ni_)                                 \
        {                                                                                                       \
            type MAD_TILES_elem_ = intel_sub_group_shuffle(                                                     \
                (tile_buf_b)[MAD_TILES_ni_ * GRP_CNT(tile_bk, sg_size) +                                        \
                             MAD_TILES_bki_ / (sg_size) + MAD_TILES_idx_b_],                                    \
                MAD_TILES_bki_ % (sg_size));                                                                    \
                                                                                                                \
            __attribute__((opencl_unroll_hint(GRP_CNT(tile_m, sg_size))))                                       \
            for (uint MAD_TILES_mgi_ = 0; MAD_TILES_mgi_ < GRP_CNT(tile_m, sg_size); ++MAD_TILES_mgi_)          \
            {                                                                                                   \
                (tile_buf_c)[MAD_TILES_ni_ * GRP_CNT(tile_m, sg_size) + MAD_TILES_mgi_ + MAD_TILES_idx_c_] =    \
                    type_op_mad(                                                                                \
                        MAD_TILES_elem_,                                                                        \
                        (tile_buf_a)[MAD_TILES_aki_ * GRP_CNT(tile_m, sg_size) +                                \
                                     MAD_TILES_mgi_ + MAD_TILES_idx_a_],                                        \
                        (tile_buf_c)[MAD_TILES_ni_ * GRP_CNT(tile_m, sg_size) +                                 \
                                     MAD_TILES_mgi_ + MAD_TILES_idx_c_]                                         \
                    );                                                                                          \
            }                                                                                                   \
        }                                                                                                       \
        ++MAD_TILES_aki_; ++MAD_TILES_bki_;                                                                     \
    }                                                                                                           \
} while (false)


#define MAD_TILES_Tfloat_ATFC_BTFC_CTFC_(sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,                                  \
                                         tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)                            \
    MAD_TILES_Txx_SGxx_Mxx_Nxx_AKxx_BKxx_ATFC_BTFC_CTFC_H1_(float, fma, sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,   \
                                                            tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)

#define MAD_TILES_Tdouble_ATFC_BTFC_CTFC_(sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,                                 \
                                          tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)                           \
    MAD_TILES_Txx_SGxx_Mxx_Nxx_AKxx_BKxx_ATFC_BTFC_CTFC_H1_(double, fma, sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,  \
                                                            tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)

#define MAD_TILES_Tfloat_ATFR_BTFC_CTFC_(sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,                                  \
                                         tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)                            \
    MAD_TILES_Txx_SGxx_Mxx_Nxx_AKxx_BKxx_ATFR_BTFC_CTFC_H1_(float, fma, sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,   \
                                                            tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)

#define MAD_TILES_Tdouble_ATFR_BTFC_CTFC_(sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,                                 \
                                          tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)                           \
    MAD_TILES_Txx_SGxx_Mxx_Nxx_AKxx_BKxx_ATFR_BTFC_CTFC_H1_(double, fma, sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,  \
                                                            tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)

#define MAD_TILES_Tfloat_ATFHR_BTFC_CTFC_(sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,                                 \
                                          tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)                           \
    MAD_TILES_Txx_SGxx_Mxx_Nxx_AKxx_BKxx_ATFR_BTFC_CTFC_H1_(float, fma, sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,   \
                                                            tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)

#define MAD_TILES_Tdouble_ATFHR_BTFC_CTFC_(sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,                                \
                                           tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)                          \
    MAD_TILES_Txx_SGxx_Mxx_Nxx_AKxx_BKxx_ATFR_BTFC_CTFC_H1_(double, fma, sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,  \
                                                            tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)


#define MAD_TILES_H1_(type, fmt_a, fmt_b, fmt_c, sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,                    \
                      tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)                                         \
    MAD_TILES_T##type##_ATF##fmt_a##_BTF##fmt_b##_CTF##fmt_c##_(sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,     \
                                                                tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)
#define MAD_TILES(type, fmt_a, fmt_b, fmt_c, sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,   \
                  tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)                        \
    MAD_TILES_H1_(type, fmt_a, fmt_b, fmt_c, sg_size, tile_m, tile_n, tile_ak, tile_bk, k_part,   \
                  tile_buf_a, tile_buf_b, tile_buf_c, idx_a, idx_b, idx_c)
    
// =====================================================================================================================
// =====================================================================================================================

#ifdef GEMM_N3_TEMPLATE_SG_SUPPORTED_UNDEF_MACRO_
    #undef GEMM_N3_TEMPLATE_SG_SUPPORTED_UNDEF_MACRO_
    #undef SG_SUPPORTED
#endif

#endif // GEMM_N3_TEMPLATE_H_DECLS


// =====================================================================================================================
// Parameters (setting default if not set externally):
// =====================================================================================================================
#ifndef TARG_KERNEL_NAME
    #define TARG_KERNEL_NAME          sample_sgemm_AN_BN
    #define GEMM_N3_TEMPLATE_TARG_KERNEL_NAME_UNDEF_MACRO_
#endif

#ifndef TARG_SG_SIZE
    #define TARG_SG_SIZE              16
    #define GEMM_N3_TEMPLATE_TARG_SG_SIZE_UNDEF_MACRO_
#endif

#ifndef TARG_DATA_TYPE
    #define TARG_DATA_TYPE            float
    #define GEMM_N3_TEMPLATE_TARG_DATA_TYPE_UNDEF_MACRO_
#endif

#ifndef TARG_MATRIX_FMT_A
    #define TARG_MATRIX_FMT_A         C
    #define GEMM_N3_TEMPLATE_TARG_MATRIX_FMT_A_UNDEF_MACRO_
#endif

#ifndef TARG_MATRIX_FMT_B
    #define TARG_MATRIX_FMT_B         C
    #define GEMM_N3_TEMPLATE_TARG_MATRIX_FMT_B_UNDEF_MACRO_
#endif

#ifndef TARG_TILE_M
    #define TARG_TILE_M               16
    #define GEMM_N3_TEMPLATE_TARG_TILE_M_UNDEF_MACRO_
#endif

#ifndef TARG_TILE_N
    #define TARG_TILE_N               8
    #define GEMM_N3_TEMPLATE_TARG_TILE_N_UNDEF_MACRO_
#endif

#ifndef TARG_TILE_AK
    #define TARG_TILE_AK              16
    #define GEMM_N3_TEMPLATE_TARG_TILE_AK_UNDEF_MACRO_
#endif

#ifndef TARG_TILE_BK
    #define TARG_TILE_BK              16
    #define GEMM_N3_TEMPLATE_TARG_TILE_BK_UNDEF_MACRO_
#endif

#ifndef TARG_TILE_IDX_GDIM_M
    #define TARG_TILE_IDX_GDIM_M      0
    #define GEMM_N3_TEMPLATE_TARG_TILE_IDX_GDIM_M_UNDEF_MACRO_
#endif

#ifndef TARG_TILE_IDX_GDIM_N
    #define TARG_TILE_IDX_GDIM_N      1
    #define GEMM_N3_TEMPLATE_TARG_TILE_IDX_GDIM_N_UNDEF_MACRO_
#endif

#ifndef TARG_DT_OP_NZT
    #define TARG_DT_OP_NZT(x)         ((x) != 0)
    #define GEMM_N3_TEMPLATE_TARG_DT_OP_NZT_UNDEF_MACRO_
#endif

#ifndef TARG_DT_OP_MUL
    #define TARG_DT_OP_MUL(x, y)      ((x) * (y))
    #define GEMM_N3_TEMPLATE_TARG_DT_OP_MUL_UNDEF_MACRO_
#endif

#ifndef TARG_DT_OP_ADD
    #define TARG_DT_OP_ADD(x, y)      ((x) + (y))
    #define GEMM_N3_TEMPLATE_TARG_DT_OP_ADD_UNDEF_MACRO_
#endif

#ifndef TARG_DT_OP_MAD
    #define TARG_DT_OP_MAD(x, y, z)   fma(x, y, z)
    #define GEMM_N3_TEMPLATE_TARG_DT_OP_MAD_UNDEF_MACRO_
#endif

// =====================================================================================================================
// =====================================================================================================================


__attribute__((intel_reqd_sub_group_size(TARG_SG_SIZE)))
__attribute__((reqd_work_group_size(TARG_SG_SIZE, 1, 1)))
__kernel void TARG_KERNEL_NAME(uint m, uint n, uint k,
                               TARG_DATA_TYPE alpha,
                               DATA_LS_RO_AS const TARG_DATA_TYPE* A, uint lda,
                               DATA_LS_RO_AS const TARG_DATA_TYPE* B, uint ldb,
                               TARG_DATA_TYPE beta,
                               DATA_LS_RW_AS TARG_DATA_TYPE* C, uint ldc)
{    
    // [CONSTEXPR][UNIFORM] Minimal size of tile in "k" dimension.
    const uint tile_k_min   = TARG_TILE_AK < TARG_TILE_BK ? TARG_TILE_AK : TARG_TILE_BK;
    // [CONSTEXPR][UNIFORM] Maximum ratio between sizes of tile in "k" dimension in matrix A and B.
    const uint tile_k_ratio = TARG_TILE_AK < TARG_TILE_BK ? TARG_TILE_BK / TARG_TILE_AK : TARG_TILE_AK / TARG_TILE_BK;
    // [CONSTEXPR][UNIFORM] Size of buffer arrays needed to store tiles.
    const uint tile_buf_size_A = TARG_TILE_M  * TARG_TILE_AK / TARG_SG_SIZE;
    const uint tile_buf_size_B = TARG_TILE_BK * TARG_TILE_N  / TARG_SG_SIZE;
    const uint tile_buf_size_C = TARG_TILE_M  * TARG_TILE_N  / TARG_SG_SIZE;
    
    // [UNIFORM] Indices of the tile calculated by current sub-group.
    const uint gid_m = get_group_id(TARG_TILE_IDX_GDIM_M);
    const uint gid_n = get_group_id(TARG_TILE_IDX_GDIM_N);

    // [UNIFORM] Maximum number of tiles in each iterated dimension.
    const uint tile_cnt_k = (k + tile_k_min - 1) / tile_k_min;
    
    // Tiles (buffer arrays).
    TARG_DATA_TYPE tile_buf_C[tile_buf_size_C] = {};

    // [UNIFORM] Tile byte offsets.
    uint tile_boff_A = SHIFT_TILE_OFFSET(GET_MATRIX_LS_FMT(TARG_MATRIX_FMT_A), TARG_DATA_TYPE, TARG_TILE_M,  TARG_TILE_AK, 0, lda, gid_m, 0);
    uint tile_boff_B = SHIFT_TILE_OFFSET(GET_MATRIX_LS_FMT(TARG_MATRIX_FMT_B), TARG_DATA_TYPE, TARG_TILE_BK, TARG_TILE_N,  0, ldb, 0,     gid_n);
    uint tile_boff_C = SHIFT_TILE_OFFSET(C,                                    TARG_DATA_TYPE, TARG_TILE_M,  TARG_TILE_N,  0, ldc, gid_m, gid_n);

    // [UNIFORM] Indicators that tile on specified dimension can be loaded without limit testing.
    const bool full_m = (gid_m + 1) * TARG_TILE_M <= m;
    const bool full_n = (gid_n + 1) * TARG_TILE_N <= n;
    
    // Control explicitly unroll rate on tile_k_ratio using two nested loops.
    __attribute__((opencl_unroll_hint(1)))
    for (uint ki = 0; ki < tile_cnt_k; ki += tile_k_ratio)
    {
        __attribute__((opencl_unroll_hint))
        for (uint kii = 0; kii < tile_k_ratio; ++kii)
        {
            // Tiles (buffer arrays).
            TARG_DATA_TYPE tile_buf_A[tile_buf_size_A];
            TARG_DATA_TYPE tile_buf_B[tile_buf_size_B];
            
            // [UNIFORM] Indicators that tile on specified dimension can be loaded without limit testing.
            const bool full_ak = (ki + kii + 1) * TARG_TILE_AK <= k;
            const bool full_bk = (ki + kii + 1) * TARG_TILE_BK <= k;
            
            // Support for different ratio of tile loading when tile "k" size used in A and B are different.
            // TARG_TILE_AK > TARG_TILE_BK -> kii is dividable by (TARG_TILE_AK / TARG_TILE_BK)
            if (TARG_TILE_AK <= TARG_TILE_BK || kii % (TARG_TILE_AK / TARG_TILE_BK) == 0)
            {
                if (full_m & full_ak)
                {
                    LOAD_TILE(GET_MATRIX_LS_FMT(TARG_MATRIX_FMT_A), TARG_DATA_TYPE, TARG_SG_SIZE, TARG_TILE_M, TARG_TILE_AK,
                              tile_buf_A, 0, A, m, k, lda, tile_boff_A, gid_m, ki + kii);
                }
                else
                {
                    LOAD_TILE_WITH_LIMIT(GET_MATRIX_LS_FMT(TARG_MATRIX_FMT_A), TARG_DATA_TYPE, TARG_SG_SIZE, TARG_TILE_M, TARG_TILE_AK,
                                         tile_buf_A, 0, A, m, k, lda, tile_boff_A, gid_m, ki + kii);
                }

                tile_boff_A = SHIFT_TILE_OFFSET(GET_MATRIX_LS_FMT(TARG_MATRIX_FMT_A), TARG_DATA_TYPE, TARG_TILE_M,  TARG_TILE_AK, tile_boff_A, lda, 0, 1);
            }

            // Support for different ratio of tile loading when tile "k" size used in A and B are different.
            // TARG_TILE_BK > TARG_TILE_AK -> kii is dividable by (TARG_TILE_BK / TARG_TILE_AK)
            if (TARG_TILE_BK <= TARG_TILE_AK || kii % (TARG_TILE_BK / TARG_TILE_AK) == 0)
            {
                if (full_n & full_bk)
                {
                    LOAD_TILE(GET_MATRIX_LS_FMT(TARG_MATRIX_FMT_B), TARG_DATA_TYPE, TARG_SG_SIZE, TARG_TILE_BK, TARG_TILE_N,
                              tile_buf_B, 0, B, k, n, ldb, tile_boff_B, ki + kii, gid_n);
                }
                else
                {
                    LOAD_TILE_WITH_LIMIT(GET_MATRIX_LS_FMT(TARG_MATRIX_FMT_B), TARG_DATA_TYPE, TARG_SG_SIZE, TARG_TILE_BK, TARG_TILE_N,
                                         tile_buf_B, 0, B, k, n, ldb, tile_boff_B, ki + kii, gid_n);
                }

                tile_boff_B = SHIFT_TILE_OFFSET(GET_MATRIX_LS_FMT(TARG_MATRIX_FMT_B), TARG_DATA_TYPE, TARG_TILE_BK, TARG_TILE_N,  tile_boff_B, ldb, 1, 0);
            }

            MAD_TILES(TARG_DATA_TYPE, TARG_MATRIX_FMT_A, TARG_MATRIX_FMT_B, C, TARG_SG_SIZE,
                      TARG_TILE_M, TARG_TILE_N, TARG_TILE_AK, TARG_TILE_BK, kii,
                      tile_buf_A, tile_buf_B, tile_buf_C, 0, 0, 0);
        }
    }

    // Save or update C.
    if (ldc % 4 == 0)
    {
        if (full_m & full_n)
        {
            if (TARG_DT_OP_NZT(beta))
            {
                // Previous value of C.
                TARG_DATA_TYPE tile_buf_prev_C[tile_buf_size_C];
                LOAD_TILE(C, TARG_DATA_TYPE, TARG_SG_SIZE, TARG_TILE_M, TARG_TILE_N,
                          tile_buf_prev_C, 0, C, m, n, ldc, tile_boff_C, gid_m, gid_n);

                __attribute__((opencl_unroll_hint))
                for (uint ei = 0; ei < tile_buf_size_C; ++ei)
                {
                    tile_buf_C[ei] = TARG_DT_OP_MAD(alpha, tile_buf_C[ei], TARG_DT_OP_MUL(beta, tile_buf_prev_C[ei])); 
                }
            }
            else
            {
                __attribute__((opencl_unroll_hint))
                for (uint ei = 0; ei < tile_buf_size_C; ++ei)
                {
                    tile_buf_C[ei] = TARG_DT_OP_MUL(alpha, tile_buf_C[ei]); 
                }
            }

            STORE_TILE(C, TARG_DATA_TYPE, TARG_SG_SIZE, TARG_TILE_M, TARG_TILE_N, 4,
                       tile_buf_C, 0, C, m, n, ldc, tile_boff_C, gid_m, gid_n);
        }
        else
        {
            if (TARG_DT_OP_NZT(beta))
            {
                // Previous value of C.
                TARG_DATA_TYPE tile_buf_prev_C[tile_buf_size_C];
                LOAD_TILE_WITH_LIMIT(C, TARG_DATA_TYPE, TARG_SG_SIZE, TARG_TILE_M, TARG_TILE_N,
                                     tile_buf_prev_C, 0, C, m, n, ldc, tile_boff_C, gid_m, gid_n);

                __attribute__((opencl_unroll_hint))
                for (uint ei = 0; ei < tile_buf_size_C; ++ei)
                {
                    tile_buf_C[ei] = TARG_DT_OP_MAD(alpha, tile_buf_C[ei], TARG_DT_OP_MUL(beta, tile_buf_prev_C[ei])); 
                }
            }
            else
            {
                __attribute__((opencl_unroll_hint))
                for (uint ei = 0; ei < tile_buf_size_C; ++ei)
                {
                    tile_buf_C[ei] = TARG_DT_OP_MUL(alpha, tile_buf_C[ei]); 
                }
            }

            STORE_TILE_WITH_LIMIT(C, TARG_DATA_TYPE, TARG_SG_SIZE, TARG_TILE_M, TARG_TILE_N, 4,
                                  tile_buf_C, 0, C, m, n, ldc, tile_boff_C, gid_m, gid_n);
        }
    }
    else
    {
        if (full_m & full_n)
        {
            if (TARG_DT_OP_NZT(beta))
            {
                // Previous value of C.
                TARG_DATA_TYPE tile_buf_prev_C[tile_buf_size_C];
                LOAD_TILE(C, TARG_DATA_TYPE, TARG_SG_SIZE, TARG_TILE_M, TARG_TILE_N,
                          tile_buf_prev_C, 0, C, m, n, ldc, tile_boff_C, gid_m, gid_n);

                __attribute__((opencl_unroll_hint))
                for (uint ei = 0; ei < tile_buf_size_C; ++ei)
                {
                    tile_buf_C[ei] = TARG_DT_OP_MAD(alpha, tile_buf_C[ei], TARG_DT_OP_MUL(beta, tile_buf_prev_C[ei])); 
                }
            }
            else
            {
                __attribute__((opencl_unroll_hint))
                for (uint ei = 0; ei < tile_buf_size_C; ++ei)
                {
                    tile_buf_C[ei] = TARG_DT_OP_MUL(alpha, tile_buf_C[ei]); 
                }
            }

            STORE_TILE(C, TARG_DATA_TYPE, TARG_SG_SIZE, TARG_TILE_M, TARG_TILE_N, 1,
                       tile_buf_C, 0, C, m, n, ldc, tile_boff_C, gid_m, gid_n);
        }
        else
        {
            if (TARG_DT_OP_NZT(beta))
            {
                // Previous value of C.
                TARG_DATA_TYPE tile_buf_prev_C[tile_buf_size_C];
                LOAD_TILE_WITH_LIMIT(C, TARG_DATA_TYPE, TARG_SG_SIZE, TARG_TILE_M, TARG_TILE_N,
                                     tile_buf_prev_C, 0, C, m, n, ldc, tile_boff_C, gid_m, gid_n);

                __attribute__((opencl_unroll_hint))
                for (uint ei = 0; ei < tile_buf_size_C; ++ei)
                {
                    tile_buf_C[ei] = TARG_DT_OP_MAD(alpha, tile_buf_C[ei], TARG_DT_OP_MUL(beta, tile_buf_prev_C[ei])); 
                }
            }
            else
            {
                __attribute__((opencl_unroll_hint))
                for (uint ei = 0; ei < tile_buf_size_C; ++ei)
                {
                    tile_buf_C[ei] = TARG_DT_OP_MUL(alpha, tile_buf_C[ei]); 
                }
            }

            STORE_TILE_WITH_LIMIT(C, TARG_DATA_TYPE, TARG_SG_SIZE, TARG_TILE_M, TARG_TILE_N, 1,
                                  tile_buf_C, 0, C, m, n, ldc, tile_boff_C, gid_m, gid_n);
        }
    }
}


// =====================================================================================================================
// Clean up (defaulted parameters):
// =====================================================================================================================
#ifdef GEMM_N3_TEMPLATE_TARG_KERNEL_NAME_UNDEF_MACRO_
    #undef GEMM_N3_TEMPLATE_TARG_KERNEL_NAME_UNDEF_MACRO_
    #undef TARG_KERNEL_NAME
#endif

#ifdef GEMM_N3_TEMPLATE_TARG_SG_SIZE_UNDEF_MACRO_
    #undef GEMM_N3_TEMPLATE_TARG_SG_SIZE_UNDEF_MACRO_
    #undef TARG_SG_SIZE
#endif

#ifdef GEMM_N3_TEMPLATE_TARG_DATA_TYPE_UNDEF_MACRO_
    #undef GEMM_N3_TEMPLATE_TARG_DATA_TYPE_UNDEF_MACRO_
    #undef TARG_DATA_TYPE
#endif

#ifdef GEMM_N3_TEMPLATE_TARG_MATRIX_FMT_A_UNDEF_MACRO_
    #undef GEMM_N3_TEMPLATE_TARG_MATRIX_FMT_A_UNDEF_MACRO_
    #undef TARG_MATRIX_FMT_A
#endif

#ifdef GEMM_N3_TEMPLATE_TARG_MATRIX_FMT_B_UNDEF_MACRO_
    #undef GEMM_N3_TEMPLATE_TARG_MATRIX_FMT_B_UNDEF_MACRO_
    #undef TARG_MATRIX_FMT_B
#endif

#ifdef GEMM_N3_TEMPLATE_TARG_TILE_M_UNDEF_MACRO_
    #undef GEMM_N3_TEMPLATE_TARG_TILE_M_UNDEF_MACRO_
    #undef TARG_TILE_M
#endif

#ifdef GEMM_N3_TEMPLATE_TARG_TILE_N_UNDEF_MACRO_
    #undef GEMM_N3_TEMPLATE_TARG_TILE_N_UNDEF_MACRO_
    #undef TARG_TILE_N
#endif

#ifdef GEMM_N3_TEMPLATE_TARG_TILE_AK_UNDEF_MACRO_
    #undef GEMM_N3_TEMPLATE_TARG_TILE_AK_UNDEF_MACRO_
    #undef TARG_TILE_AK
#endif

#ifdef GEMM_N3_TEMPLATE_TARG_TILE_BK_UNDEF_MACRO_
    #undef GEMM_N3_TEMPLATE_TARG_TILE_BK_UNDEF_MACRO_
    #undef TARG_TILE_BK
#endif

#ifdef GEMM_N3_TEMPLATE_TARG_TILE_IDX_GDIM_M_UNDEF_MACRO_
    #undef GEMM_N3_TEMPLATE_TARG_TILE_IDX_GDIM_M_UNDEF_MACRO_
    #undef TARG_TILE_IDX_GDIM_M
#endif

#ifdef GEMM_N3_TEMPLATE_TARG_TILE_IDX_GDIM_N_UNDEF_MACRO_
    #undef GEMM_N3_TEMPLATE_TARG_TILE_IDX_GDIM_N_UNDEF_MACRO_
    #undef TARG_TILE_IDX_GDIM_N
#endif

#ifdef GEMM_N3_TEMPLATE_TARG_DT_OP_NZT_UNDEF_MACRO_
    #undef GEMM_N3_TEMPLATE_TARG_DT_OP_NZT_UNDEF_MACRO_
    #undef TARG_DT_OP_NZT
#endif

#ifdef GEMM_N3_TEMPLATE_TARG_DT_OP_MUL_UNDEF_MACRO_
    #undef GEMM_N3_TEMPLATE_TARG_DT_OP_MUL_UNDEF_MACRO_
    #undef TARG_DT_OP_MUL
#endif

#ifdef GEMM_N3_TEMPLATE_TARG_DT_OP_ADD_UNDEF_MACRO_
    #undef GEMM_N3_TEMPLATE_TARG_DT_OP_ADD_UNDEF_MACRO_
    #undef TARG_DT_OP_ADD
#endif

#ifdef GEMM_N3_TEMPLATE_TARG_DT_OP_MAD_UNDEF_MACRO_
    #undef GEMM_N3_TEMPLATE_TARG_DT_OP_MAD_UNDEF_MACRO_
    #undef TARG_DT_OP_MAD
#endif

// =====================================================================================================================
// =====================================================================================================================
