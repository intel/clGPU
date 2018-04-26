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
#include "test_symv.hpp"

#include <functions/Ssymv.hpp>
#include <functions/Chemv.hpp>

namespace iclgpu { namespace tests {

using ::testing::Combine;
using ::testing::Values;

template<>
struct func_traits<iclgpu::functions::Ssymv>
{
    using data_type = float;

    static void reference(iclgpu::functions::Ssymv::params& params)
    {
        cpu_symv<data_type>(params.uplo, params.n, params.alpha, params.A, params.lda, params.x, params.incx, params.beta, params.y, params.incy);
    }
};

using test_Ssymv = test_symv<iclgpu::functions::Ssymv>;

TEST_P(test_Ssymv, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Ssymv>(params, impl_name));
    for (size_t i = 0; i < y.size(); i++)
    {
        EXPECT_FLOAT_EQ(y_ref[i], y[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    C256,
    test_Ssymv,
    Combine(
        Values(""),                                                 // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER, ICLBLAS_FILL_MODE_LOWER),   // uplo
        Values(2 << 7, (2 << 7) + 13),                              // n
        Values(1, 3),                                               // incx
        Values(1, 3),                                               // incy
        Values(0, 13)                                               // lda_add
    ),
    testing::internal::DefaultParamName<test_Ssymv::ParamType>
);

template<>
struct func_traits<iclgpu::functions::Chemv>
{
    using data_type = iclgpu::complex_t;

    static void reference(iclgpu::functions::Chemv::params& params)
    {
        cpu_symv<data_type, true>(params.uplo, params.n, params.alpha, params.A, params.lda, params.x, params.incx, params.beta, params.y, params.incy);
    }
};

using test_Chemv = test_symv<iclgpu::functions::Chemv>;

TEST_P(test_Chemv, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Chemv>(params, impl_name));
    for (size_t i = 0; i < y.size(); i++)
    {
        EXPECT_COMPLEX_EQ(y_ref[i], y[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    C256,
    test_Chemv,
    Combine(
        Values(""),                                                 // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER, ICLBLAS_FILL_MODE_LOWER),   // uplo
        Values(2 << 7, (2 << 7) + 13),                              // n
        Values(1, 3),                                               // incx
        Values(1, 3),                                               // incy
        Values(0, 13)                                               // lda_add
    ),
    testing::internal::DefaultParamName<test_Chemv::ParamType>
);

} } // namespace iclgpu::tests
