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
void cpu_axpy(
    const int n,
    const data_type alpha,
    const data_type* x,
    const int incx,
    data_type* y,
    const int incy)
{
    auto acc_alpha = static_cast<acc_type>(alpha);
    auto x_ptr = x;
    auto y_ptr = y;
    if (incx < 0)
    {
        x_ptr -= incx * n;
    }
    if (incy < 0)
    {
        y_ptr -= incy * n;
    }
    for (int i = 0; i < n; ++i)
    {
        auto this_x = static_cast<acc_type>(*x_ptr);
        auto this_y = static_cast<acc_type>(*y_ptr);
        auto result = this_y + acc_alpha * this_x;
        *y_ptr = static_cast<data_type>(result);
        x_ptr += incx;
        y_ptr += incy;
    }
}

template<class Func>
struct test_axpy : test_base_VV<Func>
{
    using data_type = typename test_base_VV<Func>::data_type;
    using data_arr_type = typename test_base_VV<Func>::data_arr_type;
    data_type alpha;
    data_arr_type y_ref;

    void init_values() override
    {
        alpha = get_random_scalar<data_type>();
        y_ref = this->y;
    }

    typename Func::params get_params() override
    {
        return
        {
            this->num,
            this->alpha,
            this->x.data(),
            this->incx,
            this->y.data(),
            this->incy
        };
    }
    typename Func::params get_params_ref() override
    {
        return
        {
            this->num,
            this->alpha,
            this->x.data(),
            this->incx,
            this->y_ref.data(),
            this->incy
        };
    }
};

}}
