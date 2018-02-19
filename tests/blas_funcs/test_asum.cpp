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
#include "test_asum.hpp"

#include <functions/Sasum.hpp>

namespace iclgpu { namespace tests {

using ::testing::Combine;
using ::testing::Values;

template<>
struct func_traits<iclgpu::functions::Sasum>
{
    using data_type = float;
    using result_type = float;

    static void reference(iclgpu::functions::Sasum::params& params)
    {
        double result = 0.0;

        for (int i = 0; i < params.n; i++)
        {
            const auto this_x = params.x[i * params.incx];
            result += static_cast<double>(std::abs(this_x));
        }

        params.result[0] = static_cast<float>(result);
    }
};

using test_Sasum = test_asum<iclgpu::functions::Sasum>;

TEST_P(test_Sasum, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Sasum>(params, impl_name));
    EXPECT_FLOAT_EQ(result_ref, result);
}

INSTANTIATE_TEST_CASE_P(
    S1K,
    test_Sasum,
    Combine(
        Values("naive"),
        Values(1 << 10, (5 << 10) + 13),
        Values(1, 3)
    ),
    testing::internal::DefaultParamName<test_Sasum::ParamType> // workaround for gTest + GCC -Wpedantic incompatibility
);

INSTANTIATE_TEST_CASE_P(
    S5K_13,
    test_Sasum,
    Combine(
        Values("simd16_single_thread", "simd16_two_stage", "simd16x16", "slm_reduction"),
        Values(5 << 10, (5 << 10) + 13),
        Values(1, 3)
    ),
    testing::internal::DefaultParamName<test_Sasum::ParamType> // workaround for gTest + GCC -Wpedantic incompatibility
);

INSTANTIATE_TEST_CASE_P(
    S5K_13_noinc,
    test_Sasum,
    Combine(
        Values("simd16_two_stage_noinc"),
        Values(5 << 10, (5 << 10) + 13),
        Values(1)
    ),
    testing::internal::DefaultParamName<test_Sasum::ParamType> // workaround for gTest + GCC -Wpedantic incompatibility
);

}}
