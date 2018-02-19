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

#include "functions/Crotg.hpp"

static const char* module_name = "Crotg_naive";
static const char* kernel_name = "Crotg_naive";

namespace iclgpu { namespace functions { namespace implementations {

bool Crotg_naive::accept(const Crotg::params& params, Crotg::score& score)
{
    return true;
}

event Crotg_naive::execute(const Crotg::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    
    auto buf_size = 1;

    auto buf_a = engine->get_inout_buffer(params.a, buf_size);
    kernel->set_arg(0, buf_a);
    auto buf_b = engine->get_inout_buffer(params.b, buf_size);
    kernel->set_arg(1, buf_b);
    auto buf_c = engine->get_output_buffer(params.c, buf_size);
    kernel->set_arg(2, buf_c);
    auto buf_s = engine->get_output_buffer(params.s, buf_size);
    kernel->set_arg(3, buf_s);


    auto gws = nd_range(1);
    auto lws = null_range;

    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
