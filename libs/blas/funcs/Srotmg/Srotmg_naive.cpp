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

#include "functions/Srotmg.hpp"

static const char* module_name = "Srotmg_naive";
static const char* kernel_name = "Srotmg_naive";

namespace iclgpu { namespace functions { namespace implementations {

bool Srotmg_naive::accept(const Srotmg::params& params, Srotmg::score& score)
{
    return true;
}

event Srotmg_naive::execute(const Srotmg::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    size_t result_buf_size = 5;
    size_t buf_size = 1;

    auto buf_d1 = engine->get_inout_buffer(params.d1, buf_size);
    kernel->set_arg(0, buf_d1);
    auto buf_d2 = engine->get_inout_buffer(params.d2, buf_size);
    kernel->set_arg(1, buf_d2);
    auto buf_b1 = engine->get_inout_buffer(params.b1, buf_size);
    kernel->set_arg(2, buf_b1);
    kernel->set_arg(3, params.b2);
    auto buf_result = engine->get_output_buffer(params.result, result_buf_size);
    kernel->set_arg(4, buf_result);


    auto gws = nd_range(1);
    auto lws = null_range;

    kernel->set_options({ gws, lws });

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
