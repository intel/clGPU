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
#include "test_syr2.hpp"

#include <functions/Ssyr2.hpp>
#include <functions/Cher2.hpp>

namespace iclgpu { namespace tests {

using ::testing::Combine;
using ::testing::Values;

template<typename _data_type, typename func, bool is_hermitian = false>
struct syr2_traits
{
    using data_type = _data_type;

    static void reference(typename func::params& params)
    {
        cpu_syr2<data_type, is_hermitian>(params.uplo, params.n, params.alpha, params.x, params.incx, params.y, params.incy, params.A, params.lda);
    }
};

template<>
struct func_traits<iclgpu::functions::Ssyr2> : syr2_traits<float, iclgpu::functions::Ssyr2>
{};

using test_Ssyr2 = test_syr2<iclgpu::functions::Ssyr2>;

TEST_P(test_Ssyr2, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Ssyr2>(params, impl_name));
    for (size_t i = 0; i < A.size(); ++i)
    {
        ASSERT_FLOAT_EQ(A_ref[i], A[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    S256_upper,
    test_Ssyr2,
    Combine(
        Values("naive_upper", "opt_async_upper"),   // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER),            // uplo
        Values(2 << 7, (2 << 7) + 13),              // n
        Values(1, 3),                               // incx
        Values(1, 3),                               // incy
        Values(0, 13)                               // lda_add
    ),
    testing::internal::DefaultParamName<test_Ssyr2::ParamType>
);

INSTANTIATE_TEST_CASE_P(
    S256_lower,
    test_Ssyr2,
    Combine(
        Values("naive_lower", "opt_async_lower"),   // impl_name
        Values(ICLBLAS_FILL_MODE_LOWER),            // uplo
        Values(2 << 7, (2 << 7) + 13),              // n
        Values(1, 3),                               // incx
        Values(1, 3),                               // incy
        Values(0, 13)                               // lda_add
    ),
    testing::internal::DefaultParamName<test_Ssyr2::ParamType>
);

template<>
struct func_traits<iclgpu::functions::Cher2> : syr2_traits<iclgpu::complex_t, iclgpu::functions::Cher2, true>
{};

using test_Cher2 = test_syr2<iclgpu::functions::Cher2>;

TEST_P(test_Cher2, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Cher2>(params, impl_name));
    for (size_t i = 0; i < A.size(); ++i)
    {
        EXPECT_COMPLEX_EQ(A_ref[i], A[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    C256,
    test_Cher2,
    Combine(
        Values(""),                                                 // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER, ICLBLAS_FILL_MODE_LOWER),   // uplo
        Values(2 << 7, (2 << 7) + 13),                              // n
        Values(1, 3),                                               // incx
        Values(1, 3),                                               // incy
        Values(0, 13)                                               // lda_add
    ),
    testing::internal::DefaultParamName<test_Cher2::ParamType>
);

} } // namespace iclgpu::tests
