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

#include "functions/Sgemv.hpp"

static const char* module_name = "Sgemv_naive_async";
static const char* kernel_name = "Sgemv_naive_async";

namespace iclgpu { namespace functions { namespace implementations {

bool Sgemv_naive_async::accept(const Sgemv::params& params, Sgemv::score& score)
{
    score.n = 2.f;
    return true;
}

event Sgemv_naive_async::execute(const Sgemv::params& params, const std::vector<event>& dep_events)
{ 
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);

    size_t buf_matrix_size = params.lda * params.n;
    size_t buf_vector_x = params.incx;
    size_t buf_vector_y = params.incy;

    /* For non-transpose matrix A*/
    if (params.trans == 0)
    {
        buf_vector_x *= params.n;
        buf_vector_y *= params.m;
    }
    /* For (conj.)-transpose matrix A*/
    else
    {
        buf_vector_x *= params.m;
        buf_vector_y *= params.n;
    }


    kernel->set_arg(0, params.trans);
    kernel->set_arg(1, params.m);
    kernel->set_arg(2, params.n);
    kernel->set_arg(3, params.alpha);
    auto buf_A = engine->get_input_buffer(params.A, buf_matrix_size);
    kernel->set_arg(4, buf_A);
    kernel->set_arg(5, params.lda);
    auto buf_x = engine->get_input_buffer(params.x, buf_vector_x);
    kernel->set_arg(6, buf_x);
    kernel->set_arg(7, params.incx);
    kernel->set_arg(8, params.beta);
    auto buf_y = engine->get_inout_buffer(params.y, buf_vector_y);
    kernel->set_arg(9, buf_y);
    kernel->set_arg(10, params.incy);

    nd_range gws((params.trans == 0) ? params.m : params.n);
    auto lws = null_range;

    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
