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

#include "functions/Sspr.hpp"

static const char* module_name = "Sspr_naive_lower";
static const char* kernel_name = "Sspr_naive_lower";

namespace iclgpu { namespace functions { namespace implementations {

bool Sspr_naive_lower::accept(const Sspr::params& params, Sspr::score& score)
{
    if (params.uplo == 1)
    {
        score.uplo = 2.0f;
        return true;
    }
    else
    {
        return false;
    }
}

event Sspr_naive_lower::execute(const Sspr::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);

    auto matrix_a_size = params.n * (params.n + 1) / 2;
    auto vector_size = params.n * params.incx;

    //N
    kernel->set_arg(0, params.n);

    //Alpha
    kernel->set_arg(1, params.alpha);

    //Vector X
    auto buf_vector_x = engine->get_input_buffer(params.x, vector_size);
    kernel->set_arg(2, buf_vector_x);

    kernel->set_arg(3, params.incx);

    //Matrix AP
    auto buf_matrix_a = engine->get_inout_buffer(params.AP, matrix_a_size);
    kernel->set_arg(4, buf_matrix_a);

    auto gws = nd_range(params.n, params.n);
    auto lws = null_range;

    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
