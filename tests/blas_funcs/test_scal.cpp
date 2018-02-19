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

template<>
struct func_traits<iclgpu::functions::Sscal>
{
    using data_type = float;
    using alpha_type = float;

    static void reference(iclgpu::functions::Sscal::params& params)
    {
        const auto alpha = static_cast<double>(params.alpha);
        int index = 0;
        for (int i = 0; i < params.n; i++)
        {
            auto this_x = static_cast<double>(params.x[index]);
            this_x *= alpha;
            params.x[index] = static_cast<float>(this_x);

            index += params.incx;
        }
    }
};

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
        Values("naive"),
        Values(5 << 10, (5 << 10) + 13),
        Values(1, 3)
    ),
    testing::internal::DefaultParamName<test_Sscal::ParamType> // workaround for gTest + GCC -Wpedantic incompatibility
);

INSTANTIATE_TEST_CASE_P(
    S5K_13_noinc,
    test_Sscal,
    Combine(
        Values("noinc"),
        Values(5 << 10, (5 << 10) + 13),
        Values(1)
    ),
    testing::internal::DefaultParamName<test_Sscal::ParamType> // workaround for gTest + GCC -Wpedantic incompatibility
);

template<>
struct func_traits<iclgpu::functions::Cscal>
{
    using data_type = iclgpu::complex_t;
    using alpha_type = iclgpu::complex_t;

    static void reference(iclgpu::functions::Cscal::params& params)
    {
        const auto alpha = static_cast<std::complex<double>>(params.alpha);
        int index = 0;
        for (int i = 0; i < params.n; i++)
        {
            auto this_x = static_cast<std::complex<double>>(params.x[index]);
            this_x *= alpha;
            params.x[index] = static_cast<iclgpu::complex_t>(this_x);

            index += params.incx;
        }
    }
};

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
