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

#include "functions/Csrot.hpp"

static const char* module_name = "Csrot_async";
static const char* kernel_name = "Csrot_async";

namespace iclgpu { namespace functions { namespace implementations {

bool Csrot_async::accept(const Csrot::params& params, Csrot::score& score)
{
    score.n = 1.25f;

    return true;
}

event Csrot_async::execute(const Csrot::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    
    auto x_buf_size = params.n * params.incx;
    auto y_buf_size = params.n * params.incy;

    auto buf_x = engine->get_inout_buffer(params.x, x_buf_size);
    kernel->set_arg(0, buf_x);
    kernel->set_arg(1, params.incx);
    auto buf_y = engine->get_inout_buffer(params.y, y_buf_size);
    kernel->set_arg(2, buf_y);
    kernel->set_arg(3, params.incy);
    kernel->set_arg(4, params.c);
    kernel->set_arg(5, params.s);

    auto gws = nd_range(params.n);
    auto lws = null_range;

    auto options = kernel_options(gws, lws);
    kernel->set_options(options);

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
