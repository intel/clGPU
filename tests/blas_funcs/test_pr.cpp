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
#include "test_pr.hpp"

#include <functions/Chpr.hpp>
#include <functions/Sspr.hpp>

namespace iclgpu { namespace tests {

using ::testing::Combine;
using ::testing::Values;

template<>
struct func_traits<iclgpu::functions::Chpr>
{
    using data_type = iclgpu::complex_t;
    using alpha_data_type = float;

    static void reference(iclgpu::functions::Chpr::params& params)
    {
        const auto alpha = static_cast<double>(params.alpha);
        const bool ltriangle = params.uplo == ICLBLAS_FILL_MODE_LOWER;

        if (ltriangle)
        {
            int packed_counter = 0;
            for (int col = 0; col < params.n; col++)
            {
                auto this_a = static_cast<std::complex<double>>(params.AP[packed_counter]);
                const auto right_x = static_cast<std::complex<double>>(params.x[col * params.incx]);

                this_a += right_x * std::conj(right_x) * alpha;
                this_a.imag(0.f);
                params.AP[packed_counter] = static_cast<iclgpu::complex_t>(this_a);

                packed_counter++;
                for (int row = col + 1; row < params.n; row++)
                {
                    auto this_a = static_cast<std::complex<double>>(params.AP[packed_counter]);
                    const auto left_x = static_cast<std::complex<double>>(params.x[row * params.incx]);

                    this_a += left_x * std::conj(right_x) * alpha;
                    params.AP[packed_counter] = static_cast<iclgpu::complex_t>(this_a);

                    packed_counter++;
                }
            }
        }
        else {
            int packed_counter = 0;
            for (int col = 0; col < params.n; col++)
            {
                const auto right_x = static_cast<std::complex<double>>(params.x[col * params.incx]);
                for (int row = 0; row < col; row++)
                {
                    auto this_a = static_cast<std::complex<double>>(params.AP[packed_counter]);
                    const auto left_x = static_cast<std::complex<double>>(params.x[row * params.incx]);

                    this_a += left_x * std::conj(right_x) * alpha;
                    params.AP[packed_counter] = static_cast<iclgpu::complex_t>(this_a);

                    packed_counter++;
                }
                auto this_a = static_cast<std::complex<double>>(params.AP[packed_counter]);

                this_a += right_x * std::conj(right_x) * alpha;
                this_a.imag(0.f);
                params.AP[packed_counter] = this_a;

                packed_counter++;
            }
        }
    }
};

using test_Chpr = test_pr<iclgpu::functions::Chpr>;

TEST_P(test_Chpr, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Chpr>(params, impl_name));
    for (size_t i = 0; i < ap.size(); i++)
    {
        EXPECT_COMPLEX_EQ(ap_ref[i], ap[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    C512,
    test_Chpr,
    Combine(
        Values(""),                                                 // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER, ICLBLAS_FILL_MODE_LOWER),   // uplo
        Values(2 << 8),                                             // num
        Values(1, 3)                                                // incx
    ),
    testing::internal::DefaultParamName<test_Chpr::ParamType>
);

template<>
struct func_traits<iclgpu::functions::Sspr>
{
    using data_type = float;
    using alpha_data_type = float;

    static void reference(iclgpu::functions::Sspr::params& params)
    {
        const auto alpha = static_cast<double>(params.alpha);
        const bool ltriangle = params.uplo == ICLBLAS_FILL_MODE_LOWER;

        if (ltriangle)
        {
            int packed_counter = 0;
            for (int col = 0; col < params.n; col++)
            {
                const auto right_x = static_cast<double>(params.x[col * params.incx]);
                for (int row = col; row < params.n; row++)
                {
                    auto this_a = static_cast<double>(params.AP[packed_counter]);
                    const auto left_x = static_cast<double>(params.x[row * params.incx]);

                    this_a += left_x * right_x * alpha;
                    params.AP[packed_counter] = static_cast<float>(this_a);

                    packed_counter++;
                }
            }
        }
        else {
            int packed_counter = 0;
            for (int col = 0; col < params.n; col++)
            {
                const auto right_x = static_cast<double>(params.x[col * params.incx]);
                for (int row = 0; row <= col; row++)
                {
                    auto this_a = static_cast<double>(params.AP[packed_counter]);
                    const auto left_x = static_cast<double>(params.x[row * params.incx]);

                    this_a += left_x * right_x * alpha;
                    params.AP[packed_counter] = static_cast<float>(this_a);

                    packed_counter++;
                }
            }
        }
    }
};

using test_Sspr = test_pr<iclgpu::functions::Sspr>;

TEST_P(test_Sspr, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Sspr>(params, impl_name));
    for (size_t i = 0; i < ap.size(); i++)
    {
        EXPECT_FLOAT_EQ(ap_ref[i], ap[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    S512,
    test_Sspr,
    Combine(
        Values(""),                                                 // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER, ICLBLAS_FILL_MODE_LOWER),   // uplo
        Values(2 << 8),                                             // num
        Values(1, 3)                                                // incx
    ),
    testing::internal::DefaultParamName<test_Sspr::ParamType>
);

}}
