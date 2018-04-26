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
// File: tile_load_store.h.cl

#ifndef TILE_LOAD_STORE_H
#define TILE_LOAD_STORE_H

// Includes:
#include <pp_load_store_helpers.h>
#include <data_load_store.h>

// =====================================================================================================================
// Tile offset operations:
// =====================================================================================================================
//
// SHIFT_TILE_OFFSET - Shifts base byte offset ("base_offset") by the specified number of tiles
//                     in row and/or column direction.
//
// SHIFT_TILE_OFFSET(matrix_fmt, type, tile_rc, tile_cc, base_offset, matrix_ld, delta_rt, delta_ct)
//
// Parameters:
//  - matrix_fmt  - [PP] Storage format of the matrix. Currently allowed values: C, R.
//                   * C - matrix is in column-major format (default in BLAS).
//                   * R - matrix is in row-major format (traspose/Hermitian transpose in BLAS).
//  - type        - [PP] Type of element in matrix / matrix tile (to scale byte offset correctly).
//  - tile_rc     - [PP] Number of rows in matrix tile.
//  - tile_cc     - [PP] Number of columns in matrix tile.
//  - base_offset - [UNIFORM] A base byte offset which will be shifted. An unsiged / non-negative value
//                  is recommended.
//  - matrix_ld   - [UNIFORM] Matrix leading dimension (in elements).
//  - delta_rt    - [UNIFORM] Number of tiles to shift in row direction.
//  - delta_ct    - [UNIFORM] Number of tiles to shift in column direction.
//
// LUT mangled params:
//  - MF<matrix_fmt>
//
// =====================================================================================================================

#define SHIFT_TILE_OFFSET_MFC_(type, tile_rc, tile_cc, base_offset, matrix_ld, delta_rt, delta_ct)   \
    SHIFT_TYPE_OFFSET(type, base_offset, matrix_ld * (delta_ct * tile_cc) + (delta_rt * tile_rc))
#define SHIFT_TILE_OFFSET_MFR_(type, tile_rc, tile_cc, base_offset, matrix_ld, delta_rt, delta_ct)   \
    SHIFT_TYPE_OFFSET(type, base_offset, matrix_ld * (delta_rt * tile_rc) + (delta_ct * tile_cc))


#define SHIFT_TILE_OFFSET_H1_(matrix_fmt, type, tile_rc, tile_cc, base_offset, matrix_ld, delta_rt, delta_ct)   \
    SHIFT_TILE_OFFSET_MF##matrix_fmt##_(type, tile_rc, tile_cc, base_offset, matrix_ld, delta_rt, delta_ct)
#define SHIFT_TILE_OFFSET(matrix_fmt, type, tile_rc, tile_cc, base_offset, matrix_ld, delta_rt, delta_ct)       \
    SHIFT_TILE_OFFSET_H1_(matrix_fmt, type, tile_rc, tile_cc, base_offset, matrix_ld, delta_rt, delta_ct)
    
// =====================================================================================================================
// =====================================================================================================================

// =====================================================================================================================
// Loading tile:
// =====================================================================================================================

// LOAD_TILE            - Loads 2D tile of specified size into tile buffer. It does not check limits
//                        (does not check the tile is contained in matrix).
//                        Can use different algorithms to store data depending on load size, type, matrix format, etc.
// LOAD_TILE_WITH_LIMIT - Loads 2D tile of specified size into tile buffer. It checks limits
//                        (whether tile is inside matrix). If element of tile is outside allowed range,
//                        the corresponding value in tile buffer is set to DATA_LS_LIMIT_FILL_VALUE.
//                        Can use different algorithms to store data depending on load size, type, matrix format, etc.
//
// LOAD_TILE(matrix_fmt, type, sg_size, tile_rc, tile_cc, buffer, index,
//           matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)
// LOAD_TILE_WITH_LIMIT(matrix_fmt, type, sg_size, tile_rc, tile_cc, buffer, index,
//                      matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)
//
//
// Parameters:
//  - matrix_fmt       - [PP] Storage format of the matrix. Currently allowed values: C, R.
//                        * C - matrix is in column-major format (default in BLAS).
//                        * R - matrix is in row-major format (traspose/Hermitian transpose in BLAS).
//  - type             - [PP] Type of element in matrix (pointed by "matrix_ptr") / matrix tile ("buffer").
//  - sg_size          - [PP] Selected sub-group size or SIMD size. Allowed values: 8, 16, 32.
//  - tile_rc          - [PP] Number of rows in matrix tile.
//  - tile_cc          - [PP] Number of columns in matrix tile.
//  - buffer           - [UNIFORM] An lvalue expression of array of type elements. Please use expression without
//                       side-effects (it will be expanded possibly more than once). Destination tile buffer.
//  - index            - [UNIFORM] (0-based) Start position where data should be stored/written into tile "buffer".
//  - matrix_ptr       - [UNIFORM] Pointer convertible to "DATA_LS_RO_AS const void*" that represents input
//                       matrix from which data is loaded.
//  - matrix_rc        - [PP] Number of rows in matrix.
//  - matrix_cc        - [PP] Number of columns in matrix.
//  - matrix_ld        - [UNIFORM] Matrix leading dimension (in elements).
//  - tile_boff        - [UNIFORM] Byte offset to the beginning of the tile (to avoid recalculation each time
//                       LOAD_TILE / LOAD_TILE_WITH_LIMIT is invoked). Can be calculated with SHIFT_TILE_OFFSET.
//  - tile_rid         - [UNIFORM] (0-based) Index of tile in row dimension (usually used only when
//                       "tile_boff" is not sufficient, e.g. when calculation limits).
//  - tile_cid         - [UNIFORM] (0-based) Index of tile in column dimension (usually used only when
//                       "tile_boff" is not sufficient, e.g. when calculation limits).
//
// LUT mangled params:
//  - T<type>
//  - SG<sg_size>
//  - TR<tile_rc>
//  - TC<tile_cc>
//  - LC<0|1>        - [PP] Indicates that limit check is enabled. Allowed values: 0, 1.
//  - MF<matrix_fmt>
//
// =====================================================================================================================

#define LOAD_TILE_Txx_SGxx_TRxx_TCxx_LC0_MFC_H1_(type, sg_size, tile_rc, tile_cc, buffer, index,           \
                                                 matrix_ptr, matrix_rc, matrix_cc, matrix_ld,              \
                                                 tile_boff, tile_rid, tile_cid)                            \
do {                                                                                                       \
    uint LOAD_TILE_tmp_idx_         = (index);                                                             \
    uint LOAD_TILE_tile_boff_       = (tile_boff);                                                         \
    const uint LOAD_TILE_matrix_ld_ = (matrix_ld);                                                         \
                                                                                                           \
    __attribute__((opencl_unroll_hint))                                                                    \
    for (uint LOAD_TILE_tmp_li_ = 0; LOAD_TILE_tmp_li_ < (tile_cc); ++LOAD_TILE_tmp_li_)                   \
    {                                                                                                      \
        LOAD_DATA(type, sg_size, tile_rc, buffer, LOAD_TILE_tmp_idx_, matrix_ptr, LOAD_TILE_tile_boff_);   \
        LOAD_TILE_tmp_idx_ += GRP_CNT(tile_rc, sg_size);                                                   \
        LOAD_TILE_tile_boff_ = SHIFT_TYPE_OFFSET(type, LOAD_TILE_tile_boff_, LOAD_TILE_matrix_ld_);        \
    }                                                                                                      \
} while (false)

#define LOAD_TILE_Txx_SGxx_TRxx_TCxx_LC0_MFR_H1_(type, sg_size, tile_rc, tile_cc, buffer, index,           \
                                                 matrix_ptr, matrix_rc, matrix_cc, matrix_ld,              \
                                                 tile_boff, tile_rid, tile_cid)                            \
do {                                                                                                       \
    uint LOAD_TILE_tmp_idx_         = (index);                                                             \
    uint LOAD_TILE_tile_boff_       = (tile_boff);                                                         \
    const uint LOAD_TILE_matrix_ld_ = (matrix_ld);                                                         \
                                                                                                           \
    __attribute__((opencl_unroll_hint))                                                                    \
    for (uint LOAD_TILE_tmp_li_ = 0; LOAD_TILE_tmp_li_ < (tile_rc); ++LOAD_TILE_tmp_li_)                   \
    {                                                                                                      \
        LOAD_DATA(type, sg_size, tile_cc, buffer, LOAD_TILE_tmp_idx_, matrix_ptr, LOAD_TILE_tile_boff_);   \
        LOAD_TILE_tmp_idx_ += GRP_CNT(tile_cc, sg_size);                                                   \
        LOAD_TILE_tile_boff_ = SHIFT_TYPE_OFFSET(type, LOAD_TILE_tile_boff_, LOAD_TILE_matrix_ld_);        \
    }                                                                                                      \
} while (false)

#define LOAD_TILE_Txx_SGxx_TRxx_TCxx_LC1_MFC_H1_(type, sg_size, tile_rc, tile_cc, buffer, index,                   \
                                                 matrix_ptr, matrix_rc, matrix_cc, matrix_ld,                      \
                                                 tile_boff, tile_rid, tile_cid)                                    \
do {                                                                                                               \
    const uint LOAD_TILE_matrix_rc_ = (matrix_rc);                                                                 \
    const uint LOAD_TILE_matrix_cc_ = (matrix_cc);                                                                 \
    const uint LOAD_TILE_matrix_ld_ = (matrix_ld);                                                                 \
    const uint LOAD_TILE_tile_rid_  = (tile_rid);                                                                  \
    const uint LOAD_TILE_tile_cid_  = (tile_cid);                                                                  \
    uint LOAD_TILE_tmp_idx_         = (index);                                                                     \
    uint LOAD_TILE_tile_row_idx_    = (tile_rc) * (LOAD_TILE_tile_rid_);                                           \
    uint LOAD_TILE_tile_col_idx_    = (tile_cc) * (LOAD_TILE_tile_cid_);                                           \
    uint LOAD_TILE_tile_boff_       = (tile_boff);                                                                 \
                                                                                                                   \
    if (LOAD_TILE_tile_row_idx_ + (tile_rc) > LOAD_TILE_matrix_rc_)                                                \
    {                                                                                                              \
        uint LOAD_TILE_tile_blim_ = SHIFT_TILE_OFFSET(C, type, tile_rc, tile_cc, 0, LOAD_TILE_matrix_ld_,          \
                                                      LOAD_TILE_matrix_rc_, LOAD_TILE_tile_cid_);                  \
                                                                                                                   \
        __attribute__((opencl_unroll_hint))                                                                        \
        for (uint LOAD_TILE_tmp_li_ = 0; LOAD_TILE_tmp_li_ < (tile_cc); ++LOAD_TILE_tmp_li_)                       \
        {                                                                                                          \
            if (LOAD_TILE_tile_col_idx_ + LOAD_TILE_tmp_li_ >= LOAD_TILE_matrix_cc_)                               \
            {                                                                                                      \
                LOAD_DATA_WITH_LIMIT(type, sg_size, tile_rc, buffer, LOAD_TILE_tmp_idx_, matrix_ptr, 0, 0);        \
            }                                                                                                      \
            else                                                                                                   \
            {                                                                                                      \
                LOAD_DATA_WITH_LIMIT(type, sg_size, tile_rc, buffer, LOAD_TILE_tmp_idx_, matrix_ptr,               \
                                     LOAD_TILE_tile_boff_, LOAD_TILE_tile_blim_);                                  \
            }                                                                                                      \
            LOAD_TILE_tmp_idx_ += GRP_CNT(tile_rc, sg_size);                                                       \
            LOAD_TILE_tile_boff_ = SHIFT_TYPE_OFFSET(type, LOAD_TILE_tile_boff_, matrix_ld);                       \
            LOAD_TILE_tile_blim_ = SHIFT_TYPE_OFFSET(type, LOAD_TILE_tile_blim_, matrix_ld);                       \
        }                                                                                                          \
    }                                                                                                              \
    else                                                                                                           \
    {                                                                                                              \
        __attribute__((opencl_unroll_hint))                                                                        \
        for (uint LOAD_TILE_tmp_li_ = 0; LOAD_TILE_tmp_li_ < (tile_cc); ++LOAD_TILE_tmp_li_)                       \
        {                                                                                                          \
            if (LOAD_TILE_tile_col_idx_ + LOAD_TILE_tmp_li_ >= LOAD_TILE_matrix_cc_)                               \
            {                                                                                                      \
                LOAD_DATA_WITH_LIMIT(type, sg_size, tile_rc, buffer, LOAD_TILE_tmp_idx_, matrix_ptr, 0, 0);        \
            }                                                                                                      \
            else                                                                                                   \
            {                                                                                                      \
                LOAD_DATA(type, sg_size, tile_rc, buffer, LOAD_TILE_tmp_idx_, matrix_ptr, LOAD_TILE_tile_boff_);   \
            }                                                                                                      \
            LOAD_TILE_tmp_idx_ += GRP_CNT(tile_rc, sg_size);                                                       \
            LOAD_TILE_tile_boff_ = SHIFT_TYPE_OFFSET(type, LOAD_TILE_tile_boff_, matrix_ld);                       \
        }                                                                                                          \
    }                                                                                                              \
} while (false)

#define LOAD_TILE_Txx_SGxx_TRxx_TCxx_LC1_MFR_H1_(type, sg_size, tile_rc, tile_cc, buffer, index,                   \
                                                 matrix_ptr, matrix_rc, matrix_cc, matrix_ld,                      \
                                                 tile_boff, tile_rid, tile_cid)                                    \
do {                                                                                                               \
    const uint LOAD_TILE_matrix_rc_ = (matrix_rc);                                                                 \
    const uint LOAD_TILE_matrix_cc_ = (matrix_cc);                                                                 \
    const uint LOAD_TILE_matrix_ld_ = (matrix_ld);                                                                 \
    const uint LOAD_TILE_tile_rid_  = (tile_rid);                                                                  \
    const uint LOAD_TILE_tile_cid_  = (tile_cid);                                                                  \
    uint LOAD_TILE_tmp_idx_         = (index);                                                                     \
    uint LOAD_TILE_tile_row_idx_    = (tile_rc) * (LOAD_TILE_tile_rid_);                                           \
    uint LOAD_TILE_tile_col_idx_    = (tile_cc) * (LOAD_TILE_tile_cid_);                                           \
    uint LOAD_TILE_tile_boff_       = (tile_boff);                                                                 \
                                                                                                                   \
    if (LOAD_TILE_tile_col_idx_ + (tile_cc) > LOAD_TILE_matrix_cc_)                                                \
    {                                                                                                              \
        uint LOAD_TILE_tile_blim_ = SHIFT_TILE_OFFSET(R, type, tile_rc, tile_cc, 0, LOAD_TILE_matrix_ld_,          \
                                                      LOAD_TILE_tile_rid_, LOAD_TILE_matrix_cc_);                  \
                                                                                                                   \
        __attribute__((opencl_unroll_hint))                                                                        \
        for (uint LOAD_TILE_tmp_li_ = 0; LOAD_TILE_tmp_li_ < (tile_rc); ++LOAD_TILE_tmp_li_)                       \
        {                                                                                                          \
            if (LOAD_TILE_tile_row_idx_ + LOAD_TILE_tmp_li_ >= LOAD_TILE_matrix_rc_)                               \
            {                                                                                                      \
                LOAD_DATA_WITH_LIMIT(type, sg_size, tile_cc, buffer, LOAD_TILE_tmp_idx_, matrix_ptr, 0, 0);        \
            }                                                                                                      \
            else                                                                                                   \
            {                                                                                                      \
                LOAD_DATA_WITH_LIMIT(type, sg_size, tile_cc, buffer, LOAD_TILE_tmp_idx_, matrix_ptr,               \
                                     LOAD_TILE_tile_boff_, LOAD_TILE_tile_blim_);                                  \
            }                                                                                                      \
            LOAD_TILE_tmp_idx_ += GRP_CNT(tile_cc, sg_size);                                                       \
            LOAD_TILE_tile_boff_ = SHIFT_TYPE_OFFSET(type, LOAD_TILE_tile_boff_, matrix_ld);                       \
            LOAD_TILE_tile_blim_ = SHIFT_TYPE_OFFSET(type, LOAD_TILE_tile_blim_, matrix_ld);                       \
        }                                                                                                          \
    }                                                                                                              \
    else                                                                                                           \
    {                                                                                                              \
        __attribute__((opencl_unroll_hint))                                                                        \
        for (uint LOAD_TILE_tmp_li_ = 0; LOAD_TILE_tmp_li_ < (tile_rc); ++LOAD_TILE_tmp_li_)                       \
        {                                                                                                          \
            if (LOAD_TILE_tile_row_idx_ + LOAD_TILE_tmp_li_ >= LOAD_TILE_matrix_rc_)                               \
            {                                                                                                      \
                LOAD_DATA_WITH_LIMIT(type, sg_size, tile_cc, buffer, LOAD_TILE_tmp_idx_, matrix_ptr, 0, 0);        \
            }                                                                                                      \
            else                                                                                                   \
            {                                                                                                      \
                LOAD_DATA(type, sg_size, tile_cc, buffer, LOAD_TILE_tmp_idx_, matrix_ptr, LOAD_TILE_tile_boff_);   \
            }                                                                                                      \
            LOAD_TILE_tmp_idx_ += GRP_CNT(tile_cc, sg_size);                                                       \
            LOAD_TILE_tile_boff_ = SHIFT_TYPE_OFFSET(type, LOAD_TILE_tile_boff_, matrix_ld);                       \
        }                                                                                                          \
    }                                                                                                              \
} while (false)


#define LOAD_TILE_LC0_MFC_(type, sg_size, tile_rc, tile_cc, buffer, index,                                             \
                           matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)                 \
    LOAD_TILE_Txx_SGxx_TRxx_TCxx_LC0_MFC_H1_(type, sg_size, tile_rc, tile_cc, buffer, index,                           \
                                          matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)

#define LOAD_TILE_LC0_MFR_(type, sg_size, tile_rc, tile_cc, buffer, index,                                             \
                           matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)                 \
    LOAD_TILE_Txx_SGxx_TRxx_TCxx_LC0_MFR_H1_(type, sg_size, tile_rc, tile_cc, buffer, index,                           \
                                          matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)
                                          
#define LOAD_TILE_LC1_MFC_(type, sg_size, tile_rc, tile_cc, buffer, index,                                             \
                           matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)                 \
    LOAD_TILE_Txx_SGxx_TRxx_TCxx_LC1_MFC_H1_(type, sg_size, tile_rc, tile_cc, buffer, index,                           \
                                          matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)

#define LOAD_TILE_LC1_MFR_(type, sg_size, tile_rc, tile_cc, buffer, index,                                             \
                           matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)                 \
    LOAD_TILE_Txx_SGxx_TRxx_TCxx_LC1_MFR_H1_(type, sg_size, tile_rc, tile_cc, buffer, index,                           \
                                          matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)

                                          
#define LOAD_TILE_H1_(matrix_fmt, type, sg_size, tile_rc, tile_cc, buffer, index,                                 \
                      matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)                 \
    LOAD_TILE_LC0_MF##matrix_fmt##_(type, sg_size, tile_rc, tile_cc, buffer, index,                               \
                                    matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)
#define LOAD_TILE(matrix_fmt, type, sg_size, tile_rc, tile_cc, buffer, index,                                     \
                  matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)                     \
    LOAD_TILE_H1_(matrix_fmt, type, sg_size, tile_rc, tile_cc, buffer, index,                                     \
                  matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)

#define LOAD_TILE_WITH_LIMIT_H1_(matrix_fmt, type, sg_size, tile_rc, tile_cc, buffer, index,                      \
                                 matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)      \
    LOAD_TILE_LC1_MF##matrix_fmt##_(type, sg_size, tile_rc, tile_cc, buffer, index,                               \
                                    matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)
#define LOAD_TILE_WITH_LIMIT(matrix_fmt, type, sg_size, tile_rc, tile_cc, buffer, index,                          \
                             matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)          \
    LOAD_TILE_WITH_LIMIT_H1_(matrix_fmt, type, sg_size, tile_rc, tile_cc, buffer, index,                          \
                             matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)

// =====================================================================================================================
// =====================================================================================================================

// =====================================================================================================================
// Storing tile:
// =====================================================================================================================

// STORE_TILE            - Stores 2D tile of specified size into tile buffer. It does not check limits
//                         (does not check the tile is contained in matrix).
//                         Can use different algorithms to store data depending on load size, type, matrix format, etc.
// STORE_TILE_WITH_LIMIT - Stores 2D tile of specified size into tile buffer. It checks limits (whether tile is
//                         inside matrix). If element of tile is outside allowed range, the write of this element is
//                         discarded.
//                         Can use different algorithms to store data depending on load size, type, matrix format, etc.
//
// STORE_TILE(matrix_fmt, type, sg_size, tile_rc, tile_cc, align, buffer, index,
//            matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)
// STORE_TILE_WITH_LIMIT(matrix_fmt, type, sg_size, tile_rc, tile_cc, align, buffer, index,
//                       matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)
//
//
// Parameters:
//  - matrix_fmt       - [PP] Storage format of the matrix. Currently allowed values: C, R.
//                        * C - matrix is in column-major format (default in BLAS).
//                        * R - matrix is in row-major format (traspose/Hermitian transpose in BLAS).
//  - type             - [PP] Type of element in matrix (pointed by "matrix_ptr") / matrix tile ("buffer").
//  - sg_size          - [PP] Selected sub-group size or SIMD size. Allowed values: 8, 16, 32.
//  - tile_rc          - [PP] Number of rows in matrix tile.
//  - tile_cc          - [PP] Number of columns in matrix tile.
//  - align            - [PP] Ensured / Provided alignment of "matrix_ptr". The value provided is treated as muliply of
//                       element size, e.g. 1 means sizeof(type), 2 means 2 * sizeof(type), etc.
//                       Must be positive and power of 2 and it is limited to 128.
//                       (Effectively usually it is sufficient that "matrix_ptr" is aligned and "matrix_ld"
//                       is dividable by "align" to ensure correct alignment.)
//  - buffer           - [UNIFORM] An rvalue expression of array of type elements. Please use expression without
//                       side-effects (it will be expanded possibly more than once). Source tile buffer that will
//                       be stored into "matrix_ptr".
//  - index            - [UNIFORM] (0-based) Start position where data should be read from tile "buffer" and stored
//                       into "matrix_ptr".
//  - matrix_ptr       - [UNIFORM] Pointer convertible to "DATA_LS_RW_AS void*" that represents output
//                       matrix to which data is stored.
//  - matrix_rc        - [PP] Number of rows in matrix.
//  - matrix_cc        - [PP] Number of columns in matrix.
//  - matrix_ld        - [UNIFORM] Matrix leading dimension (in elements).
//  - tile_boff        - [UNIFORM] Byte offset to the beginning of the tile (to avoid recalculation each time
//                       LOAD_TILE / LOAD_TILE_WITH_LIMIT is invoked). Can be calculated with SHIFT_TILE_OFFSET.
//  - tile_rid         - [UNIFORM] (0-based) Index of tile in row dimension (usually used only when
//                       "tile_boff" is not sufficient, e.g. when calculation limits).
//  - tile_cid         - [UNIFORM] (0-based) Index of tile in column dimension (usually used only when
//                       "tile_boff" is not sufficient, e.g. when calculation limits).
//
// LUT mangled params:
//  - T<type>
//  - SG<sg_size>
//  - TR<tile_rc>
//  - TC<tile_cc>
//  - LC<0|1>        - [PP] Indicates that limit check is enabled. Allowed values: 0, 1.
//  - MF<matrix_fmt>
//
// =====================================================================================================================

#define STORE_TILE_Txx_SGxx_TRxx_TCxx_LC0_MFC_H1_(type, sg_size, tile_rc, tile_cc, align, buffer, index,   \
                                                  matrix_ptr, matrix_rc, matrix_cc, matrix_ld,             \
                                                  tile_boff, tile_rid, tile_cid)                           \
do {                                                                                                       \
    uint STORE_TILE_tmp_idx_         = (index);                                                            \
    uint STORE_TILE_tile_boff_       = (tile_boff);                                                        \
    const uint STORE_TILE_matrix_ld_ = (matrix_ld);                                                        \
                                                                                                           \
    __attribute__((opencl_unroll_hint))                                                                    \
    for (uint STORE_TILE_tmp_li_ = 0; STORE_TILE_tmp_li_ < (tile_cc); ++STORE_TILE_tmp_li_)                \
    {                                                                                                      \
        STORE_DATA(type, sg_size, tile_rc, align, buffer, STORE_TILE_tmp_idx_,                             \
                   matrix_ptr, STORE_TILE_tile_boff_);                                                     \
        STORE_TILE_tmp_idx_ += GRP_CNT(tile_rc, sg_size);                                                  \
        STORE_TILE_tile_boff_ = SHIFT_TYPE_OFFSET(type, STORE_TILE_tile_boff_, STORE_TILE_matrix_ld_);     \
    }                                                                                                      \
} while (false)

#define STORE_TILE_Txx_SGxx_TRxx_TCxx_LC0_MFR_H1_(type, sg_size, tile_rc, tile_cc, align, buffer, index,   \
                                                  matrix_ptr, matrix_rc, matrix_cc, matrix_ld,             \
                                                  tile_boff, tile_rid, tile_cid)                           \
do {                                                                                                       \
    uint STORE_TILE_tmp_idx_         = (index);                                                            \
    uint STORE_TILE_tile_boff_       = (tile_boff);                                                        \
    const uint STORE_TILE_matrix_ld_ = (matrix_ld);                                                        \
                                                                                                           \
    __attribute__((opencl_unroll_hint))                                                                    \
    for (uint STORE_TILE_tmp_li_ = 0; STORE_TILE_tmp_li_ < (tile_rc); ++STORE_TILE_tmp_li_)                \
    {                                                                                                      \
        STORE_DATA(type, sg_size, tile_cc, align, buffer, STORE_TILE_tmp_idx_,                             \
                   matrix_ptr, STORE_TILE_tile_boff_);                                                     \
        STORE_TILE_tmp_idx_ += GRP_CNT(tile_cc, sg_size);                                                  \
        STORE_TILE_tile_boff_ = SHIFT_TYPE_OFFSET(type, STORE_TILE_tile_boff_, STORE_TILE_matrix_ld_);     \
    }                                                                                                      \
} while (false)

#define STORE_TILE_Txx_SGxx_TRxx_TCxx_LC1_MFC_H1_(type, sg_size, tile_rc, tile_cc, align, buffer, index,      \
                                                  matrix_ptr, matrix_rc, matrix_cc, matrix_ld,                \
                                                  tile_boff, tile_rid, tile_cid)                              \
do {                                                                                                          \
    const uint STORE_TILE_matrix_rc_ = (matrix_rc);                                                           \
    const uint STORE_TILE_matrix_cc_ = (matrix_cc);                                                           \
    const uint STORE_TILE_matrix_ld_ = (matrix_ld);                                                           \
    const uint STORE_TILE_tile_rid_  = (tile_rid);                                                            \
    const uint STORE_TILE_tile_cid_  = (tile_cid);                                                            \
    uint STORE_TILE_tmp_idx_         = (index);                                                               \
    uint STORE_TILE_tile_row_idx_    = (tile_rc) * (STORE_TILE_tile_rid_);                                    \
    uint STORE_TILE_tile_col_idx_    = (tile_cc) * (STORE_TILE_tile_cid_);                                    \
    uint STORE_TILE_tile_boff_       = (tile_boff);                                                           \
                                                                                                              \
    if (STORE_TILE_tile_row_idx_ + (tile_rc) > STORE_TILE_matrix_rc_)                                         \
    {                                                                                                         \
        uint STORE_TILE_tile_blim_ = SHIFT_TILE_OFFSET(C, type, tile_rc, tile_cc, 0, STORE_TILE_matrix_ld_,   \
                                                      STORE_TILE_matrix_rc_, STORE_TILE_tile_cid_);           \
                                                                                                              \
        __attribute__((opencl_unroll_hint))                                                                   \
        for (uint STORE_TILE_tmp_li_ = 0; STORE_TILE_tmp_li_ < (tile_cc); ++STORE_TILE_tmp_li_)               \
        {                                                                                                     \
            if (STORE_TILE_tile_col_idx_ + STORE_TILE_tmp_li_ >= STORE_TILE_matrix_cc_)                       \
                break;                                                                                        \
                                                                                                              \
            STORE_DATA_WITH_LIMIT(type, sg_size, tile_rc, align, buffer, STORE_TILE_tmp_idx_, matrix_ptr,     \
                                  STORE_TILE_tile_boff_, STORE_TILE_tile_blim_);                              \
                                                                                                              \
            STORE_TILE_tmp_idx_ += GRP_CNT(tile_rc, sg_size);                                                 \
            STORE_TILE_tile_boff_ = SHIFT_TYPE_OFFSET(type, STORE_TILE_tile_boff_, matrix_ld);                \
            STORE_TILE_tile_blim_ = SHIFT_TYPE_OFFSET(type, STORE_TILE_tile_blim_, matrix_ld);                \
        }                                                                                                     \
    }                                                                                                         \
    else                                                                                                      \
    {                                                                                                         \
        __attribute__((opencl_unroll_hint))                                                                   \
        for (uint STORE_TILE_tmp_li_ = 0; STORE_TILE_tmp_li_ < (tile_cc); ++STORE_TILE_tmp_li_)               \
        {                                                                                                     \
            if (STORE_TILE_tile_col_idx_ + STORE_TILE_tmp_li_ >= STORE_TILE_matrix_cc_)                       \
                break;                                                                                        \
                                                                                                              \
            STORE_DATA(type, sg_size, tile_rc, align, buffer, STORE_TILE_tmp_idx_,                            \
                       matrix_ptr, STORE_TILE_tile_boff_);                                                    \
                                                                                                              \
            STORE_TILE_tmp_idx_ += GRP_CNT(tile_rc, sg_size);                                                 \
            STORE_TILE_tile_boff_ = SHIFT_TYPE_OFFSET(type, STORE_TILE_tile_boff_, matrix_ld);                \
        }                                                                                                     \
    }                                                                                                         \
} while (false)

#define STORE_TILE_Txx_SGxx_TRxx_TCxx_LC1_MFR_H1_(type, sg_size, tile_rc, tile_cc, align, buffer, index,      \
                                                  matrix_ptr, matrix_rc, matrix_cc, matrix_ld,                \
                                                  tile_boff, tile_rid, tile_cid)                              \
do {                                                                                                          \
    const uint STORE_TILE_matrix_rc_ = (matrix_rc);                                                           \
    const uint STORE_TILE_matrix_cc_ = (matrix_cc);                                                           \
    const uint STORE_TILE_matrix_ld_ = (matrix_ld);                                                           \
    const uint STORE_TILE_tile_rid_  = (tile_rid);                                                            \
    const uint STORE_TILE_tile_cid_  = (tile_cid);                                                            \
    uint STORE_TILE_tmp_idx_         = (index);                                                               \
    uint STORE_TILE_tile_row_idx_    = (tile_rc) * (STORE_TILE_tile_rid_);                                    \
    uint STORE_TILE_tile_col_idx_    = (tile_cc) * (STORE_TILE_tile_cid_);                                    \
    uint STORE_TILE_tile_boff_       = (tile_boff);                                                           \
                                                                                                              \
    if (STORE_TILE_tile_col_idx_ + (tile_cc) > STORE_TILE_matrix_cc_)                                         \
    {                                                                                                         \
        uint STORE_TILE_tile_blim_ = SHIFT_TILE_OFFSET(R, type, tile_rc, tile_cc, 0, STORE_TILE_matrix_ld_,   \
                                                      STORE_TILE_tile_rid_, STORE_TILE_matrix_cc_);           \
                                                                                                              \
        __attribute__((opencl_unroll_hint))                                                                   \
        for (uint STORE_TILE_tmp_li_ = 0; STORE_TILE_tmp_li_ < (tile_rc); ++STORE_TILE_tmp_li_)               \
        {                                                                                                     \
            if (STORE_TILE_tile_row_idx_ + STORE_TILE_tmp_li_ >= STORE_TILE_matrix_rc_)                       \
                break;                                                                                        \
                                                                                                              \
            STORE_DATA_WITH_LIMIT(type, sg_size, tile_cc, align, buffer, STORE_TILE_tmp_idx_, matrix_ptr,     \
                                  STORE_TILE_tile_boff_, STORE_TILE_tile_blim_);                              \
                                                                                                              \
            STORE_TILE_tmp_idx_ += GRP_CNT(tile_cc, sg_size);                                                 \
            STORE_TILE_tile_boff_ = SHIFT_TYPE_OFFSET(type, STORE_TILE_tile_boff_, matrix_ld);                \
            STORE_TILE_tile_blim_ = SHIFT_TYPE_OFFSET(type, STORE_TILE_tile_blim_, matrix_ld);                \
        }                                                                                                     \
    }                                                                                                         \
    else                                                                                                      \
    {                                                                                                         \
        __attribute__((opencl_unroll_hint))                                                                   \
        for (uint STORE_TILE_tmp_li_ = 0; STORE_TILE_tmp_li_ < (tile_rc); ++STORE_TILE_tmp_li_)               \
        {                                                                                                     \
            if (STORE_TILE_tile_row_idx_ + STORE_TILE_tmp_li_ >= STORE_TILE_matrix_rc_)                       \
                break;                                                                                        \
                                                                                                              \
            STORE_DATA(type, sg_size, tile_cc, align, buffer, STORE_TILE_tmp_idx_,                            \
                       matrix_ptr, STORE_TILE_tile_boff_);                                                    \
                                                                                                              \
            STORE_TILE_tmp_idx_ += GRP_CNT(tile_cc, sg_size);                                                 \
            STORE_TILE_tile_boff_ = SHIFT_TYPE_OFFSET(type, STORE_TILE_tile_boff_, matrix_ld);                \
        }                                                                                                     \
    }                                                                                                         \
} while (false)


#define STORE_TILE_LC0_MFC_(type, sg_size, tile_rc, tile_cc, align, buffer, index,                                     \
                           matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)                 \
    STORE_TILE_Txx_SGxx_TRxx_TCxx_LC0_MFC_H1_(type, sg_size, tile_rc, tile_cc, align, buffer, index,                   \
                                          matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)

#define STORE_TILE_LC0_MFR_(type, sg_size, tile_rc, tile_cc, align, buffer, index,                                     \
                           matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)                 \
    STORE_TILE_Txx_SGxx_TRxx_TCxx_LC0_MFR_H1_(type, sg_size, tile_rc, tile_cc, align, buffer, index,                   \
                                          matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)
                                          
#define STORE_TILE_LC1_MFC_(type, sg_size, tile_rc, tile_cc, align, buffer, index,                                     \
                           matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)                 \
    STORE_TILE_Txx_SGxx_TRxx_TCxx_LC1_MFC_H1_(type, sg_size, tile_rc, tile_cc, align, buffer, index,                   \
                                          matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)

#define STORE_TILE_LC1_MFR_(type, sg_size, tile_rc, tile_cc, align, buffer, index,                                     \
                           matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)                 \
    STORE_TILE_Txx_SGxx_TRxx_TCxx_LC1_MFR_H1_(type, sg_size, tile_rc, tile_cc, align, buffer, index,                   \
                                          matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)

                                          
#define STORE_TILE_H1_(matrix_fmt, type, sg_size, tile_rc, tile_cc, align, buffer, index,                         \
                      matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)                 \
    STORE_TILE_LC0_MF##matrix_fmt##_(type, sg_size, tile_rc, tile_cc, align, buffer, index,                       \
                                    matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)
#define STORE_TILE(matrix_fmt, type, sg_size, tile_rc, tile_cc, align, buffer, index,                             \
                   matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)                    \
    STORE_TILE_H1_(matrix_fmt, type, sg_size, tile_rc, tile_cc, align, buffer, index,                             \
                  matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)

#define STORE_TILE_WITH_LIMIT_H1_(matrix_fmt, type, sg_size, tile_rc, tile_cc, align, buffer, index,              \
                                 matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)      \
    STORE_TILE_LC1_MF##matrix_fmt##_(type, sg_size, tile_rc, tile_cc, align, buffer, index,                       \
                                    matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)
#define STORE_TILE_WITH_LIMIT(matrix_fmt, type, sg_size, tile_rc, tile_cc, align, buffer, index,                  \
                             matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)          \
    STORE_TILE_WITH_LIMIT_H1_(matrix_fmt, type, sg_size, tile_rc, tile_cc, align, buffer, index,                  \
                             matrix_ptr, matrix_rc, matrix_cc, matrix_ld, tile_boff, tile_rid, tile_cid)

// =====================================================================================================================
// =====================================================================================================================

#endif // TILE_LOAD_STORE_H
