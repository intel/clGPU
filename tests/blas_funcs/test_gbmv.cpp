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
#include "test_gbmv.hpp"

#include <functions/Sgbmv.hpp>
#include <functions/Cgbmv.hpp>

namespace iclgpu { namespace tests {

using ::testing::Combine;
using ::testing::Values;

template<typename _data_type, typename func_type>
struct gbmv_traits
{
    using data_type = _data_type;

    static void reference(typename func_type::params& params)
    {
        cpu_gbmv<data_type>(
            params.trans, params.m, params.n,
            params.kl, params.ku, params.alpha, params.A, params.lda,
            params.x, params.incx, params.beta, params.y, params.incy);
    }
};

template<>
struct func_traits<iclgpu::functions::Sgbmv> : gbmv_traits<float, iclgpu::functions::Sgbmv>
{};

using test_Sgbmv = test_gbmv<iclgpu::functions::Sgbmv>;

TEST_P(test_Sgbmv, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Sgbmv>(params, impl_name));
    for (size_t i = 0; i < y.size(); i++)
    {
        EXPECT_EQ(y_ref[i], y[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    S512,
    test_Sgbmv,
    Combine(
        Values(""),                         // impl_name
        Values(ICLBLAS_OP_N, ICLBLAS_OP_T), // trans
        Values(2 << 6, 2 << 8),             // m
        Values(2 << 6, 2 << 8),             // n
        Values(2 << 3, 2 << 5),             // kl
        Values(2 << 3, 2 << 5),             // ku
        Values(0, 13),                      // lda_add
        Values(1, 3),                       // incx
        Values(1, 3)                        // incy
    ),
    testing::internal::DefaultParamName<test_Sgbmv::ParamType>
);

template<>
struct func_traits<iclgpu::functions::Cgbmv> : gbmv_traits<iclgpu::complex_t, iclgpu::functions::Cgbmv>
{};

using test_Cgbmv = test_gbmv<iclgpu::functions::Cgbmv>;

TEST_P(test_Cgbmv, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Cgbmv>(params, impl_name));
    for (size_t i = 0; i < y.size(); i++)
    {
        EXPECT_COMPLEX_EQ(y_ref[i], y[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    C512,
    test_Cgbmv,
    Combine(
        Values(""),                                         // impl_name
        Values(ICLBLAS_OP_N, ICLBLAS_OP_T, ICLBLAS_OP_C),   // trans
        Values(2 << 6, 2 << 8),                             // m
        Values(2 << 6, 2 << 8),                             // n
        Values(2 << 3, 2 << 5),                             // kl
        Values(2 << 3, 2 << 5),                             // ku
        Values(0, 13),                                      // lda_add
        Values(1, 3),                                       // incx
        Values(1, 3)                                        // incy
    ),
    testing::internal::DefaultParamName<test_Cgbmv::ParamType>
);

}}
