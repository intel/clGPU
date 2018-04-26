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
#include "test_syr.hpp"

#include <functions/Ssyr.hpp>
#include <functions/Csyr.hpp>
#include <functions/Cher.hpp>

namespace iclgpu { namespace tests {

using ::testing::Combine;
using ::testing::Values;

template<typename _data_type, typename _alpha_data_type, typename func, bool is_hermitian = false>
struct syr_traits
{
    using data_type = _data_type;
    using alpha_data_type = _alpha_data_type;

    static void reference(typename func::params& params)
    {
        cpu_syr<data_type, alpha_data_type, is_hermitian>(params.uplo, params.n, params.alpha, params.x, params.incx, params.A, params.lda);
    }
};

template<>
struct func_traits<iclgpu::functions::Ssyr> : syr_traits<float, float, iclgpu::functions::Ssyr>
{};

using test_Ssyr = test_syr<iclgpu::functions::Ssyr>;

TEST_P(test_Ssyr, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Ssyr>(params, impl_name));
    for (size_t i = 0; i < A.size(); ++i)
    {
        ASSERT_FLOAT_EQ(A_ref[i], A[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    S256,
    test_Ssyr,
    Combine(
        Values("naive", "early_return"),                            // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER, ICLBLAS_FILL_MODE_LOWER),   // uplo
        Values(2 << 7, (2 << 7) + 13),                              // n
        Values(1, 3),                                               // incx
        Values(0, 13)                                               // lda_add
    ),
    testing::internal::DefaultParamName<test_Ssyr::ParamType>
);

INSTANTIATE_TEST_CASE_P(
    S256_lda4,
    test_Ssyr,
    Combine(
        Values("early_return_float4"),                              // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER, ICLBLAS_FILL_MODE_LOWER),   // uplo
        Values(2 << 7, (2 << 7) + 12),                              // n
        Values(1, 3),                                               // incx
        Values(0, 12)                                               // lda_add
    ),
    testing::internal::DefaultParamName<test_Ssyr::ParamType>
);

INSTANTIATE_TEST_CASE_P(
    S256_upper_lda4,
    test_Ssyr,
    Combine(
        Values("simd16x1x1_upper", "simd16x4x4_upper", "early_return_simd16x1x1_upper"),// impl_name
        Values(ICLBLAS_FILL_MODE_UPPER),                            // uplo
        Values(2 << 7, (2 << 7) + 12),                              // n
        Values(1, 3),                                               // incx
        Values(0, 12)                                               // lda_add
    ),
    testing::internal::DefaultParamName<test_Ssyr::ParamType>
);

INSTANTIATE_TEST_CASE_P(
    S256_lower_lda4,
    test_Ssyr,
    Combine(
        Values("simd16x1x1_lower", "simd16x4x4_lower", "early_return_simd16x1x1_lower"),// impl_name
        Values(ICLBLAS_FILL_MODE_LOWER),   // uplo
        Values(2 << 7, (2 << 7) + 12),                              // n
        Values(1, 3),                                               // incx
        Values(0, 12)                                               // lda_add
    ),
    testing::internal::DefaultParamName<test_Ssyr::ParamType>
);

template<>
struct func_traits<iclgpu::functions::Csyr> : syr_traits<iclgpu::complex_t, iclgpu::complex_t, iclgpu::functions::Csyr>
{};

using test_Csyr = test_syr<iclgpu::functions::Csyr>;

TEST_P(test_Csyr, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Csyr>(params, impl_name));
    for (size_t i = 0; i < A.size(); ++i)
    {
        EXPECT_COMPLEX_EQ(A_ref[i], A[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    C256,
    test_Csyr,
    Combine(
        Values(""),                                                 // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER, ICLBLAS_FILL_MODE_LOWER),   // uplo
        Values(2 << 7),                                             // n
        Values(1, 3),                                               // incx
        Values(0, 13)                                               // lda_add
    ),
    testing::internal::DefaultParamName<test_Csyr::ParamType>
);

template<>
struct func_traits<iclgpu::functions::Cher> : syr_traits<iclgpu::complex_t, float, iclgpu::functions::Cher, true>
{};

using test_Cher = test_syr<iclgpu::functions::Cher>;

TEST_P(test_Cher, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Cher>(params, impl_name));
    for (size_t i = 0; i < A.size(); ++i)
    {
        EXPECT_COMPLEX_EQ(A_ref[i], A[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    C256,
    test_Cher,
    Combine(
        Values("early_return"),                                     // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER, ICLBLAS_FILL_MODE_LOWER),   // uplo
        Values(2 << 7, (2 << 7) + 13),                              // n
        Values(1, 3),                                               // incx
        Values(0, 13)                                               // lda_add
    ),
    testing::internal::DefaultParamName<test_Cher::ParamType>
);

INSTANTIATE_TEST_CASE_P(
    C256_lda2,
    test_Cher,
    Combine(
        Values("early_return_float4"),                              // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER, ICLBLAS_FILL_MODE_LOWER),   // uplo
        Values(2 << 7, (2 << 7) + 12),                              // n
        Values(1, 3),                                               // incx
        Values(0, 12)                                               // lda_add
    ),
    testing::internal::DefaultParamName<test_Cher::ParamType>
);

INSTANTIATE_TEST_CASE_P(
    C256_upper,
    test_Cher,
    Combine(
        Values("early_return_simd16x1x1_upper"),                    // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER),                            // uplo
        Values(2 << 7, (2 << 7) + 13),                              // n
        Values(1, 3),                                               // incx
        Values(0, 13)                                               // lda_add
    ),
    testing::internal::DefaultParamName<test_Cher::ParamType>
);

INSTANTIATE_TEST_CASE_P(
    C256_lower,
    test_Cher,
    Combine(
        Values("early_return_simd16x1x1_lower"),                    // impl_name
        Values(ICLBLAS_FILL_MODE_LOWER),                            // uplo
        Values(2 << 7, (2 << 7) + 13),                              // n
        Values(1, 3),                                               // incx
        Values(0, 13)                                               // lda_add
    ),
    testing::internal::DefaultParamName<test_Cher::ParamType>
);

} } // iclgpu::tests
