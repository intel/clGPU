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
#include "test_axpy.hpp"

#include <functions/Saxpy.hpp>
#include <functions/Caxpy.hpp>

namespace iclgpu { namespace tests {

using ::testing::Combine;
using ::testing::Values;

template<typename _data_type, typename func_type>
struct axpy_traits
{
    using data_type = _data_type;
    static void reference(typename func_type::params& params)
    {
        cpu_axpy<data_type>(params.n, params.alpha, params.x, params.incx, params.y, params.incy);
    }
};

template<>
struct func_traits<iclgpu::functions::Saxpy> : axpy_traits<float, iclgpu::functions::Saxpy>
{};

using test_Saxpy = test_axpy<iclgpu::functions::Saxpy>;

TEST_P(test_Saxpy, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Saxpy>(params, impl_name));
    for (size_t i = 0; i < y.size(); i++)
    {
        EXPECT_FLOAT_EQ(y_ref[i], y[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    S2K,
    test_Saxpy,
    Combine(
        Values(""),
        Values(2 << 10),
        Values(1, 3),
        Values(1, 3)
    ),
    testing::internal::DefaultParamName<test_Saxpy::ParamType> // workaround for gTest + GCC -Wpedantic incompatibility
);

// Too slow:
//INSTANTIATE_TEST_CASE_P(
//    S2M,
//    test_Saxpy,
//    Combine(
//        Values(""),
//        Values(2 << 20),
//        Values(1, 3),
//        Values(1, 3)
//    ),
//    testing::internal::DefaultParamName<test_Saxpy::ParamType> // workaround for gTest + GCC -Wpedantic incompatibility
//);

template<>
struct func_traits<iclgpu::functions::Caxpy> : axpy_traits<iclgpu::complex_t, iclgpu::functions::Caxpy>
{};

using test_Caxpy = test_axpy<iclgpu::functions::Caxpy>;

TEST_P(test_Caxpy, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Caxpy>(params, impl_name));
    for (size_t i = 0; i < y.size(); i++)
    {
        EXPECT_COMPLEX_EQ(y_ref[i], y[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    C2K,
    test_Caxpy,
    Combine(
        Values(""),
        Values(2 << 10),
        Values(1, 3),
        Values(1, 3)
    ),
    testing::internal::DefaultParamName<test_Caxpy::ParamType> // workaround for gTest + GCC -Wpedantic incompatibility
);

// Too slow:
//INSTANTIATE_TEST_CASE_P(
//    C2M,
//    test_Caxpy,
//    Combine(
//        Values(""), 
//        Values(2 << 20),
//        Values(1, 3),
//        Values(1, 3)
//    ),
//    testing::internal::DefaultParamName<test_Caxpy::ParamType> // workaround for gTest + GCC -Wpedantic incompatibility
//);

}}
