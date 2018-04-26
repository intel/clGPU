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
#include "test_pr2.hpp"

#include <functions/Chpr2.hpp>
#include <functions/Sspr2.hpp>

namespace iclgpu { namespace tests {

using ::testing::Combine;
using ::testing::Values;

template<>
struct func_traits<iclgpu::functions::Chpr2>
{
    using data_type = iclgpu::complex_t;

    static void reference(iclgpu::functions::Chpr2::params& params)
    {
        const auto alpha = static_cast<std::complex<double>>(params.alpha);
        const bool ltriangle = params.uplo == ICLBLAS_FILL_MODE_LOWER;
        if (ltriangle)
        {
            int packed_counter = 0;
            for (int col = 0; col < params.n; col++)
            {
                const auto right_x = static_cast<std::complex<double>>(params.x[col * params.incx]);
                const auto right_y = static_cast<std::complex<double>>(params.y[col * params.incy]);

                for (int row = col; row < params.n; row++)
                {
                    const auto left_x = static_cast<std::complex<double>>(params.x[row * params.incx]);
                    const auto left_y = static_cast<std::complex<double>>(params.y[row * params.incy]);
                    auto this_a = static_cast<std::complex<double>>(params.AP[packed_counter]);

                    this_a += left_x * std::conj(right_y) * alpha;
                    this_a += left_y * std::conj(right_x) * std::conj(alpha);
                    if (row == col) this_a.imag(0.f);

                    params.AP[packed_counter] = static_cast<iclgpu::complex_t>(this_a);
                    packed_counter++;
                }
            }
        }
        else
        {
            int packed_counter = 0;
            for (int col = 0; col < params.n; col++)
            {
                const auto right_x = static_cast<std::complex<double>>(params.x[col * params.incx]);
                const auto right_y = static_cast<std::complex<double>>(params.y[col * params.incy]);

                for (int row = 0; row <= col; row++)
                {
                    const auto left_x = static_cast<std::complex<double>>(params.x[row * params.incx]);
                    const auto left_y = static_cast<std::complex<double>>(params.y[row * params.incy]);
                    auto this_a = static_cast<std::complex<double>>(params.AP[packed_counter]);

                    this_a += left_x * std::conj(right_y) * alpha;
                    this_a += left_y * std::conj(right_x) * std::conj(alpha);
                    if (row == col) this_a.imag(0.f);

                    params.AP[packed_counter] = static_cast<iclgpu::complex_t>(this_a);
                    packed_counter++;
                }
            }
        }
    }
};

using test_Chpr2 = test_pr2<iclgpu::functions::Chpr2>;

TEST_P(test_Chpr2, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Chpr2>(params, impl_name));
    for (size_t i = 0; i < ap.size(); i++)
    {
        EXPECT_COMPLEX_EQ(ap_ref[i], ap[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    C256,
    test_Chpr2,
    Combine(
        Values(""),                                                 // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER, ICLBLAS_FILL_MODE_LOWER),   // uplo
        Values(2 << 7),                                             // num
        Values(1, 3),                                               // incx
        Values(1, 3)                                                // incy
    ),
    testing::internal::DefaultParamName<test_Chpr2::ParamType>
);

template<>
struct func_traits<iclgpu::functions::Sspr2>
{
    using data_type = float;

    static void reference(iclgpu::functions::Sspr2::params& params)
    {
        const auto alpha = static_cast<double>(params.alpha);
        const bool ltriangle = params.uplo == ICLBLAS_FILL_MODE_LOWER;

        if (ltriangle)
        {
            int packed_counter = 0;
            for (int col = 0; col < params.n; col++)
            {
                const auto right_x = static_cast<double>(params.x[col * params.incx]);
                const auto right_y = static_cast<double>(params.y[col * params.incy]);

                for (int row = col; row < params.n; row++)
                {
                    const auto left_x = static_cast<double>(params.x[row * params.incx]);
                    const auto left_y = static_cast<double>(params.y[row * params.incy]);
                    auto this_a = static_cast<double>(params.AP[packed_counter]);

                    this_a += left_x * right_y * alpha;
                    this_a += left_y * right_x * alpha;

                    params.AP[packed_counter] = static_cast<float>(this_a);
                    packed_counter++;
                }
            }
        }
        else
        {
            int packed_counter = 0;
            for (int col = 0; col < params.n; col++)
            {
                const auto right_x = static_cast<double>(params.x[col * params.incx]);
                const auto right_y = static_cast<double>(params.y[col * params.incy]);

                for (int row = 0; row <= col; row++)
                {
                    const auto left_x = static_cast<double>(params.x[row * params.incx]);
                    const auto left_y = static_cast<double>(params.y[row * params.incy]);
                    auto this_a = static_cast<double>(params.AP[packed_counter]);

                    this_a += left_x * right_y * alpha;
                    this_a += left_y * right_x * alpha;

                    params.AP[packed_counter] = static_cast<float>(this_a);
                    packed_counter++;
                }
            }
        }
    }
};

using test_Sspr2 = test_pr2<iclgpu::functions::Sspr2>;

TEST_P(test_Sspr2, basic)
{
    ASSERT_NO_FATAL_FAILURE(run_function<iclgpu::functions::Sspr2>(params, impl_name));
    for (size_t i = 0; i < ap.size(); i++)
    {
        EXPECT_FLOAT_EQ(ap_ref[i], ap[i]);
    }
}

INSTANTIATE_TEST_CASE_P(
    S256,
    test_Sspr2,
    Combine(
        Values(""),                                                 // impl_name
        Values(ICLBLAS_FILL_MODE_UPPER, ICLBLAS_FILL_MODE_LOWER),   // uplo
        Values(2 << 7),                                             // num
        Values(1, 3),                                               // incx
        Values(1, 3)                                                // incy
    ),
    testing::internal::DefaultParamName<test_Sspr2::ParamType>
);

}}
