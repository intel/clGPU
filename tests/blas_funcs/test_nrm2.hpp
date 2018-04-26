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

#pragma once

#include <gtest/gtest.h>
#include "test_helpers.hpp"

namespace iclgpu { namespace tests {

template<typename data_type, typename acc_type = accumulator_type_t<data_type>>
void cpu_nrm2(
    const int n,
    const data_type* x,
    const int incx,
    absolute_type_t<data_type>* result)
{
    using std::abs;
    using std::sqrt;

    const auto abs_zero = absolute_type_t<acc_type>(0);
    const auto abs_one = absolute_type_t<acc_type>(1);

    auto scale = abs_zero; // abs of max element so far
    auto ssq = abs_one; // sum of squares

    for (int i = 0; i < n; ++i)
    {
        auto this_x = static_cast<acc_type>(x[i * incx]);

        auto this_absx = abs(this_x);
        if (this_absx != abs_zero)
        {
            if (scale < this_absx)
            {
                auto scale_inv_normx = (scale / this_absx) * (scale / this_absx);
                ssq = abs_one + scale_inv_normx * ssq;
                scale = this_absx;
            }
            else
            {
                auto scaled_normx = (this_absx / scale) * (this_absx / scale);
                ssq += scaled_normx;
            }
        }
    }

    auto nrm2 = scale * sqrt(ssq);

    *result = static_cast<absolute_type_t<data_type>>(nrm2);
}

template<class Func>
struct test_nrm2 : test_base_VS<Func, absolute_type_t<typename func_traits<Func>::data_type>>
{
    using result_type = typename test_base_VS<Func, absolute_type_t<typename func_traits<Func>::data_type>>::result_type;

    void init_values() override
    {
        this->result = get_random_scalar<result_type>();
        this->result_ref = this->result;
    }
};

}}
