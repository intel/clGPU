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
#include "test_amin.hpp"

#include <functions/Isamin.hpp>
#include <functions/Icamin.hpp>

namespace iclgpu { namespace tests {

using ::testing::Combine;
using ::testing::Values;

template<typename _data_type, typename func_type>
struct amin_traits
{
    using data_type = _data_type;

    static void reference(typename func_type::params& params)
    {
        cpu_amin<data_type>(params.n, params.x, params.incx, params.result);
    }
};

template<>
struct func_traits<iclgpu::functions::Isamin> : amin_traits<float, iclgpu::functions::Isamin>
{};

using test_Isamin = test_amin<iclgpu::functions::Isamin>;

TEST_P(test_Isamin, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Isamin>(params, impl_name));
    EXPECT_EQ(result_ref, result);
}

INSTANTIATE_TEST_CASE_P(
    S2K,
    test_Isamin,
    Combine(
        Values(""),
        Values(2 << 10),
        Values(1, 3)
    ),
    testing::internal::DefaultParamName<test_Isamin::ParamType> // workaround for gTest + GCC -Wpedantic incompatibility
);

template<>
struct func_traits<iclgpu::functions::Icamin> : amin_traits<iclgpu::complex_t, iclgpu::functions::Icamin>
{};

using test_Icamin = test_amin<iclgpu::functions::Icamin>;

TEST_P(test_Icamin, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Icamin>(params, impl_name));
    EXPECT_EQ(result_ref, result);
}

INSTANTIATE_TEST_CASE_P(
    C2K,
    test_Icamin,
    Combine(
        Values(""),
        Values(2 << 10),
        Values(1, 3)
    ),
    testing::internal::DefaultParamName<test_Icamin::ParamType> // workaround for gTest + GCC -Wpedantic incompatibility
);

}}
