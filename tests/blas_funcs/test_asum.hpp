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
void cpu_asum(
    const int n,
    const data_type* x,
    const int incx,
    absolute_type_t<data_type>* result)
{
    auto abs_sum = absolute_type_t<acc_type>(0);
    for (int i = 0; i < n; ++i)
    {
        auto this_x = static_cast<acc_type>(x[i * incx]);
        abs_sum += blas_abs(this_x);
    }
    *result = static_cast<absolute_type_t<data_type>>(abs_sum);
}

template<class Func>
struct test_asum : test_base_VS<Func, typename func_traits<Func>::result_type>
{
    using result_type = typename test_base_VS<Func, typename func_traits<Func>::result_type>::result_type;

    void init_values() override
    {
        this->result = get_random_scalar<result_type>();
        this->result_ref = this->result;
    }
};

}}
