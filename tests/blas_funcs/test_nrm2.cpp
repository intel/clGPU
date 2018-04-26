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
#include "test_nrm2.hpp"

#include <functions/Snrm2.hpp>
#include <functions/Scnrm2.hpp>

namespace iclgpu { namespace tests {

using ::testing::Combine;
using ::testing::Values;

template<typename _data_type, typename func_type>
struct nrm2_traits
{
    using data_type = _data_type;

    static void reference(typename func_type::params& params)
    {
        cpu_nrm2<data_type>(params.n, params.x, params.incx, params.result);
    }
};

template<>
struct func_traits<iclgpu::functions::Snrm2> : nrm2_traits<float, iclgpu::functions::Snrm2>
{};

using test_Snrm2 = test_nrm2<iclgpu::functions::Snrm2>;

TEST_P(test_Snrm2, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Snrm2>(params, impl_name));
    EXPECT_FLOAT_EQ(result_ref, result);
}

INSTANTIATE_TEST_CASE_P(
    S2K,
    test_Snrm2,
    Combine(
        Values(""),
        Values(2 << 10),
        Values(1, 3)
    ),
    testing::internal::DefaultParamName<test_Snrm2::ParamType> // workaround for gTest + GCC -Wpedantic incompatibility
);

template<>
struct func_traits<iclgpu::functions::Scnrm2> : nrm2_traits<iclgpu::complex_t, iclgpu::functions::Scnrm2>
{};

using test_Scnrm2 = test_nrm2<iclgpu::functions::Scnrm2>;

TEST_P(test_Scnrm2, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Scnrm2>(params, impl_name));
    EXPECT_COMPLEX_EQ(result_ref, result);
}

INSTANTIATE_TEST_CASE_P(
    C2K,
    test_Scnrm2,
    Combine(
        Values(""),
        Values(2 << 10),
        Values(1, 3)
    ),
    testing::internal::DefaultParamName<test_Scnrm2::ParamType> // workaround for gTest + GCC -Wpedantic incompatibility
);

}}
