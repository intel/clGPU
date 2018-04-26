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
#include "test_rotm.hpp"

#include "functions/Srotm.hpp"

namespace iclgpu { namespace tests {

using ::testing::Combine;
using ::testing::Values;

template<>
struct func_traits<iclgpu::functions::Srotm>
{
    using data_type = float;
    static void reference(iclgpu::functions::Srotm::params& params)
    {
        cpu_rotm<data_type>(params.n, params.x, params.incx, params.y, params.incy, params.param);
    }
};

using test_Srotm = test_rotm<iclgpu::functions::Srotm>;


TEST_P(test_Srotm, basic)
{

    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Srotm>(params, impl_name));
    for (size_t i = 0; i < x.size(); i++)
    {
        EXPECT_FLOAT_EQ(x_ref[i], x[i]);
    }
    for (size_t i = 0; i < y.size(); i++)
    {
        EXPECT_FLOAT_EQ(y_ref[i], y[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    S2K,
    test_Srotm,
    Combine(
        Values("naive"),        // impl_name
        Values(2 << 10),        // num
        Values(1, 3),           // incx
        Values(1, 3),           // incy
        Values(-1.f, 0.f, 1.f)  // flag
    ),
    testing::internal::DefaultParamName<test_Srotm::ParamType>
);

INSTANTIATE_TEST_CASE_P(
    S2K_full,
    test_Srotm,
    Combine(
        Values("packed_full", "async_full"),    // impl_name
        Values(2 << 10),                        // num
        Values(1, 3),                           // incx
        Values(1, 3),                           // incy
        Values(-1.f)                            // flag
    ),
    testing::internal::DefaultParamName<test_Srotm::ParamType>
);

INSTANTIATE_TEST_CASE_P(
    S2K_diagonal_ones,
    test_Srotm,
    Combine(
        Values("packed_diagonal_ones", "async_diagonal_ones"),  // impl_name
        Values(2 << 10),                                        // num
        Values(1, 3),                                           // incx
        Values(1, 3),                                           // incy
        Values(0.f)                                             // flag
    ),
    testing::internal::DefaultParamName<test_Srotm::ParamType>
);

INSTANTIATE_TEST_CASE_P(
    S2K_anti_diagonal_ones,
    test_Srotm,
    Combine(
        Values("packed_anti_diagonal_ones", "async_anti_diagonal_ones"),    // impl_name
        Values(2 << 10),                                                    // num
        Values(1, 3),                                                       // incx
        Values(1, 3),                                                       // incy
        Values(1.f)                                                         // flag
    ),
    testing::internal::DefaultParamName<test_Srotm::ParamType>
);

INSTANTIATE_TEST_CASE_P(
    S2K_full_noinc,
    test_Srotm,
    Combine(
        Values("noinc_full"),   // impl_name
        Values(2 << 10),        // num
        Values(1),              // incx
        Values(1),              // incy
        Values(-1.f)            // flag
    ),
    testing::internal::DefaultParamName<test_Srotm::ParamType>
);

INSTANTIATE_TEST_CASE_P(
    S2K_diagonal_ones_noinc,
    test_Srotm,
    Combine(
        Values("noinc_diagonal_ones"),  // impl_name
        Values(2 << 10),                // num
        Values(1),                      // incx
        Values(1),                      // incy
        Values(0.f)                     // flag
    ),
    testing::internal::DefaultParamName<test_Srotm::ParamType>
);

INSTANTIATE_TEST_CASE_P(
    S2K_anti_diagonal_ones_noinc,
    test_Srotm,
    Combine(
        Values("noinc_anti_diagonal_ones"), // impl_name
        Values(2 << 10),                    // num
        Values(1),                          // incx
        Values(1),                          // incy
        Values(1.f)                         // flag
    ),
    testing::internal::DefaultParamName<test_Srotm::ParamType>
);

}}
