// Copyright (c) 2017-2018 Intel Corporation
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//      http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include "test_helpers.hpp"
#include "test_trsv.hpp"

#include "functions/Strsv.hpp"
#include "functions/Ctrsv.hpp"

namespace iclgpu { namespace tests {

using ::testing::Combine;
using ::testing::Values;

template<>
struct func_traits<iclgpu::functions::Strsv>
{
    using data_type = float;
    static void reference(iclgpu::functions::Strsv::params& params){}
};

using test_Strsv = test_trsv<iclgpu::functions::Strsv>;

TEST_P(test_Strsv, basic)
{

    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Strsv>(params, impl_name));
    for (size_t i = 0; i < b.size(); i++)
    {
        EXPECT_NEAR(x[i], b[i], 1.0e-6);
    }
}

INSTANTIATE_TEST_CASE_P(
    S256,
    test_Strsv,
    Combine(
        Values(""),     // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER, ICLBLAS_FILL_MODE_LOWER),   // uplo
        Values(ICLBLAS_OP_N, ICLBLAS_OP_T),// trans
        Values(ICLBLAS_DIAG_NON_UNIT, ICLBLAS_DIAG_UNIT),   // diag
        Values(2 << 7), // num
        Values(0, 13),  // lda_add
        Values(1, 3)    // incx
    ),
    testing::internal::DefaultParamName<test_Strsv::ParamType>
);

INSTANTIATE_TEST_CASE_P(
    S256_upper_ntrans,
    test_Strsv,
    Combine(
        Values("simd16x16_upper_ntrans"),                   // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER),                    // uplo
        Values(ICLBLAS_OP_N),                               // trans
        Values(ICLBLAS_DIAG_NON_UNIT, ICLBLAS_DIAG_UNIT),   // diag
        Values(2 << 7, (2 << 7) + 13),                      // num
        Values(0, 13),                                      // lda_add
        Values(1, 3)                                        // incx
    ),
    testing::internal::DefaultParamName<test_Strsv::ParamType>
);

INSTANTIATE_TEST_CASE_P(
    S256_upper_ntrans_noinc,
    test_Strsv,
    Combine(
        Values("simd16x16_upper_ntrans_noinc", "simd16x16_upper_ntrans_noinc_aligned"),   // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER),                    // uplo
        Values(ICLBLAS_OP_N),                               // trans
        Values(ICLBLAS_DIAG_NON_UNIT, ICLBLAS_DIAG_UNIT),   // diag
        Values(2 << 7, (2 << 7) + 13),                      // num
        Values(0, 13),                                      // lda_add
        Values(1)                                           // incx
    ),
    testing::internal::DefaultParamName<test_Strsv::ParamType>
);

INSTANTIATE_TEST_CASE_P(
    S256_lower_ntrans,
    test_Strsv,
    Combine(
        Values("simd16x16_lower_ntrans"),                   // impl_name
        Values(ICLBLAS_FILL_MODE_LOWER),                    // uplo
        Values(ICLBLAS_OP_N),                               // trans
        Values(ICLBLAS_DIAG_NON_UNIT, ICLBLAS_DIAG_UNIT),   // diag
        Values(2 << 7, (2 << 7) + 13),                      // num
        Values(0, 13),                                      // lda_add
        Values(1, 3)                                        // incx
    ),
    testing::internal::DefaultParamName<test_Strsv::ParamType>
);

INSTANTIATE_TEST_CASE_P(
    S256_lower_ntrans_noinc,
    test_Strsv,
    Combine(
        Values("simd16x16_lower_ntrans_noinc", "simd16x16_lower_ntrans_noinc_aligned"),   // impl_name
        Values(ICLBLAS_FILL_MODE_LOWER),                    // uplo
        Values(ICLBLAS_OP_N),                               // trans
        Values(ICLBLAS_DIAG_NON_UNIT, ICLBLAS_DIAG_UNIT),   // diag
        Values(2 << 7, (2 << 7) + 13),                      // num
        Values(0, 13),                                      // lda_add
        Values(1)                                           // incx
    ),
    testing::internal::DefaultParamName<test_Strsv::ParamType>
);

template<>
struct func_traits<iclgpu::functions::Ctrsv>
{
    using data_type = iclgpu::complex_t;
    static void reference(iclgpu::functions::Ctrsv::params& params) {}
};

using test_Ctrsv = test_trsv<iclgpu::functions::Ctrsv>;

TEST_P(test_Ctrsv, basic)
{

    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Ctrsv>(params, impl_name));
    for (size_t i = 0; i < b.size(); i++)
    {
        EXPECT_COMPLEX_NEAR(x[i], b[i], 1.0e-6);
    }
}

INSTANTIATE_TEST_CASE_P(
    C256,
    test_Ctrsv,
    Combine(
        Values(""),     // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER, ICLBLAS_FILL_MODE_LOWER),   // uplo
        Values(ICLBLAS_OP_N, ICLBLAS_OP_T, ICLBLAS_OP_C),// trans
        Values(ICLBLAS_DIAG_NON_UNIT, ICLBLAS_DIAG_UNIT),   // diag
        Values(2 << 7), // num
        Values(0, 13),  // lda_add
        Values(1, 3)    // incx
    ),
    testing::internal::DefaultParamName<test_Ctrsv::ParamType>
);

}}
