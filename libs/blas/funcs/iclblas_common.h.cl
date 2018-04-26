/* Copyright (c) 2017-2018 Intel Corporation
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

#ifndef ICLBLAS_COMMON_H
#define ICLBLAS_COMMON_H

#ifdef __cplusplus // C++
#   include <cstdint>
#   define ENUM_INNER_TYPE std::int32_t
#elif defined (__OPENCL_C_VERSION__) // OpenCL
#   define ENUM_INNER_TYPE int
#else // Assume C
#   include <stdint.h>
#   define ENUM_INNER_TYPE int32_t
#endif

typedef ENUM_INNER_TYPE iclblasFillMode_t;
#define ICLBLAS_FILL_MODE_UPPER (0)
#define ICLBLAS_FILL_MODE_LOWER (1)

typedef ENUM_INNER_TYPE iclblasOperation_t;
#define ICLBLAS_OP_N (0)
#define ICLBLAS_OP_T (1)
#define ICLBLAS_OP_C (2)

typedef ENUM_INNER_TYPE iclblasDiagType_t;
#define ICLBLAS_DIAG_NON_UNIT (0)
#define ICLBLAS_DIAG_UNIT (1)

typedef ENUM_INNER_TYPE iclblasSideMode_t;
#define ICLBLAS_SIDE_LEFT (0)
#define ICLBLAS_SIDE_RIGHT (1)

/*  Column-major indexing (Fortran) */
#define IDXF(row, col, ld) ((col) * (ld) + (row))
/*  Column-major (Fortran) banded matrix indexing */
#define IDXF_B(row, col, ld, ku) IDX((row) + (ku) - (col), col, ld)
/*  Row-major indexing (C/C++) */
#define IDXC(row, col, ld) ((row) * (ld) + (col))
/*  Row-major (C/C++) banded matrix indexing */
#define IDXC_B(row, col, ld, kl) IDX(row, (col) + (kl) - (row), ld)

#ifdef ICLBLAS_ROW_MAJOR
#   define IDX(row, col, ld) IDXC(row, col, ld)
#   define IDX_B(row, col, ld, kl) IDXC_B(row, col, ld, kl)
#else
#   define IDX(row, col, ld) IDXF(row, col, ld)
#   define IDX_B(row, col, ld, ku) IDXF_B(row, col, ld, ku)
#endif

#undef ENUM_INNER_TYPE

#endif /* ICLBLAS_COMMON_H */
