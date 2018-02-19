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

#include "functions/Srotm.hpp"

static const char* module_name = "Srotm_naive";
static const char* kernel_name = "Srotm_naive";

namespace iclgpu { namespace functions { namespace implementations {

bool Srotm_naive::accept(const Srotm::params& params, Srotm::score& score)
{
    return true;
}

event Srotm_naive::execute(const Srotm::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    
    auto buf_size = (params.incx <= params.incy) ? params.n * params.incx : params.n * params.incy;
    auto param_size = 5;

    kernel->set_arg(0, params.n);
    auto buf_x = engine->get_inout_buffer(params.x, buf_size);
    kernel->set_arg(1, buf_x);
    kernel->set_arg(2, params.incx);
    auto buf_y = engine->get_inout_buffer(params.y, buf_size);
    kernel->set_arg(3, buf_y);
    kernel->set_arg(4, params.incy);
    auto buf_param = engine->get_input_buffer(params.param, param_size);
    kernel->set_arg(5, buf_param);


    auto gws = nd_range(1);
    auto lws = null_range;

    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
