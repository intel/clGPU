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

#include "functions/Sger.hpp"

static const char* module_name = "Sger_no_inc";
static const char* kernel_name = "Sger_no_inc";

namespace iclgpu { namespace functions { namespace implementations {

bool Sger_no_inc::accept(const Sger::params& params, Sger::score& score)
{
    if (params.incx != 1 || params.incy != 1)
        return false;
    score.incx = 1.1f;
    return true;
}

event Sger_no_inc::execute(const Sger::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);

    size_t x_buf_size = params.m*params.incx;
    size_t y_buf_size = params.n*params.incy;
    size_t a_buf_size = params.n*params.lda;

    kernel->set_arg(0, params.alpha);
    auto buf_x = engine->get_input_buffer(params.x, x_buf_size);
    kernel->set_arg(1, buf_x);
    auto buf_y = engine->get_input_buffer(params.y, y_buf_size);
    kernel->set_arg(2, buf_y);
    auto buf_a = engine->get_inout_buffer(params.A, a_buf_size);
    kernel->set_arg(3, buf_a);
    kernel->set_arg(4, params.lda);


    auto gws = nd_range(params.m, params.n);
    auto lws = null_range;

    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
