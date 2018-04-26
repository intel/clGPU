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
#include "blas_reference.hpp"

namespace iclgpu { namespace tests {

template<typename data_type>
void cpu_amax(
    const int n,
    const data_type* x,
    const int incx,
    int* result)
{
    auto current_amax = blas_abs(x[0]);
    int current_iamax = 0;
    for (int i = 1; i < n; ++i)
    {
        auto this_ax = blas_abs(x[i * incx]);
        if (this_ax > current_amax)
        {
            current_amax = this_ax;
            current_iamax = i;
        }
    }
    *result = current_iamax;
}

template<>
inline void cpu_amax<float>(
    const int n,
    const float* x,
    const int incx,
    int* result)
{
    //TODO clarify index base (0 or 1) for implementations
    *result = Isamax_reference(n, array_ref<float>(const_cast<float*>(x), n*incx), incx) - 1;
}

template<class Func>
using test_amax = test_base_VS<Func, int>;

}}
