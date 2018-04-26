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

template<typename data_type, bool conj = false, typename acc_type = accumulator_type_t<data_type>>
void cpu_dot(
    const int n,
    const data_type* x,
    const int incx,
    const data_type* y,
    const int incy,
    data_type* result)
{
    auto product = acc_type(0);
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
        if (conj)
        {
            this_x = blas_conj(this_x);
        }
        product += this_x * this_y;
        x_ptr += incx;
        y_ptr += incy;
    }

    *result = static_cast<data_type>(product);
}

template<class Func>
struct test_dot : test_base_VV<Func>
{
    using data_type = typename test_base_VV<Func>::data_type;

    data_type result;
    data_type result_ref;

    typename Func::params get_params() override
    {
        return
        {
            this->num,
            this->x.data(),
            this->incx,
            this->y.data(),
            this->incy,
            &result
        };
    }

    typename Func::params get_params_ref() override
    {
        return
        {
            this->num,
            this->x.data(),
            this->incx,
            this->y.data(),
            this->incy,
            &result_ref
        };
    }
};

}}
