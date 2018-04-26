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
#include "test_scal.hpp"

#include <functions/Sscal.hpp>
#include <functions/Cscal.hpp>

namespace iclgpu { namespace tests {

using ::testing::Combine;
using ::testing::Values;

template<typename _data_type, typename _alpha_type, typename func_type>
struct scal_traits
{
    using data_type = _data_type;
    using alpha_type = _alpha_type;

    static void reference(typename func_type::params& params)
    {
        cpu_scal<data_type, alpha_type>(params.n, params.alpha, params.x, params.incx);
    }
};

template<>
struct func_traits<iclgpu::functions::Sscal> : scal_traits<float, float, iclgpu::functions::Sscal>
{};

using test_Sscal = test_scal<iclgpu::functions::Sscal>;

TEST_P(test_Sscal, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Sscal>(params, impl_name));
    for (size_t i = 0; i < x.size(); ++i)
    {
        EXPECT_FLOAT_EQ(x_ref[i], x[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    S5K_13,
    test_Sscal,
    Combine(
        Values("naive", "packed"),
        Values(5 << 10, (5 << 10) + 13),
        Values(1, 3)
    ),
    testing::internal::DefaultParamName<test_Sscal::ParamType> // workaround for gTest + GCC -Wpedantic incompatibility
);

INSTANTIATE_TEST_CASE_P(
    S5K_13_noinc,
    test_Sscal,
    Combine(
        Values("noinc", "packed_noinc", "block_read"),
        Values(5 << 10, (5 << 10) + 13),
        Values(1)
    ),
    testing::internal::DefaultParamName<test_Sscal::ParamType> // workaround for gTest + GCC -Wpedantic incompatibility
);

template<>
struct func_traits<iclgpu::functions::Cscal> : scal_traits<iclgpu::complex_t, iclgpu::complex_t, iclgpu::functions::Cscal>
{};

using test_Cscal = test_scal<iclgpu::functions::Cscal>;

TEST_P(test_Cscal, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Cscal>(params, impl_name));
    for (size_t i = 0; i < x.size(); ++i)
    {
        EXPECT_COMPLEX_EQ(x_ref[i], x[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    C3K_13,
    test_Cscal,
    Combine(
        Values(""),
        Values(3 << 10, (3 << 10) + 13),
        Values(1, 3)
    ),
    testing::internal::DefaultParamName<test_Cscal::ParamType> // workaround for gTest + GCC -Wpedantic incompatibility
);

}}
