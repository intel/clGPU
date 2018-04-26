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
void cpu_rotm(
    const int n,
    data_type* x,
    const int incx,
    data_type* y,
    const int incy,
    const data_type* param)
{
    auto x_ptr = x;
    if (incx < 0)
    {
        x_ptr -= incx * n;
    }
    auto y_ptr = y;
    if (incy < 0)
    {
        y_ptr -= incy * n;
    }
    for (int i = 0; i < n; ++i)
    {
        auto this_x = static_cast<acc_type>(*x_ptr);
        auto this_y = static_cast<acc_type>(*y_ptr);
        auto new_x = this_x;
        auto new_y = this_y;
        if (param[0] == -1)
        {
            new_x = param[1] * this_x + param[2] * this_y;
            new_y = param[3] * this_x + param[4] * this_y;
        }
        else if (param[0] == 0)
        {
            new_x = this_x + param[2] * this_y;
            new_y = param[3] * this_x + this_y;
        }
        else if (param[0] == 1)
        {
            new_x = param[1] * this_x + this_y;
            new_y = -this_x + param[4] * this_y;
        }

        *x_ptr = static_cast<data_type>(new_x);
        *y_ptr = static_cast<data_type>(new_y);

        x_ptr += incx;
        y_ptr += incy;
    }
}

template<class Func>
struct test_rotm : test_base_VVS<Func>
{
    using data_arr_type = typename test_base_VVS<Func>::data_arr_type;

    data_arr_type x_ref;
    data_arr_type y_ref;

    void init_values() override
    {
        x_ref = this->x;
        y_ref = this->y;
    }

    typename Func::params get_params() override
    {
        return
        {
            this->num,
            this->x.data(),
            this->incx,
            this->y.data(),
            this->incy,
            this->param.data()
        };
    }

    typename Func::params get_params_ref() override
    {
        return
        {
            this->num,
            this->x_ref.data(),
            this->incx,
            this->y_ref.data(),
            this->incy,
            this->param.data()
        };
    }
};

}}
